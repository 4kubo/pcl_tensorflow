import gym
from gym import wrappers
from collections import namedtuple
import numpy as np
import tensorflow as tf
import six.moves.queue as queue
from argparse import ArgumentParser
import threading
import scipy.signal

from policy import LSTMPolicy, LinearPolicy


def arg_parse():
    parser = ArgumentParser()
    parser.add_argument("-v", "--visualise",
                        action="store_true")
    parser.add_argument("-s", "--n_total_step",
                        type=int,
                        default=10000)
    parser.add_argument("--gamma",
                        type=float,
                        default=0.99)
    parser.add_argument("--tau",
                        type=float,
                        default=0.1)
    parser.add_argument("--d",
                        type=int,
                        default=3)
    args = parser.parse_args()
    return args


def main(_):
    args = arg_parse()

    env = gym.make("CartPole-v0")
    args.max_step_per_episode = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    total_step = 0

    model = PCL(env, args)

    init_all_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_all_op)
        # model.start(sess)

        # env = wrappers.Monitor(env, "/Users/kubo-a/tmp/cart")
        while total_step <= args.n_total_step:
            visualise = True if total_step % 500 == 0 else False
            model.process(sess, visualise)
            total_step += 1

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def consistency(values, rewards, log_pies, T, d, gamma, tau):
    # Indexes for actions and rewards
    index1 = [np.arange(d) + t for t in range(T - d + 1)]
    index2 = [np.arange(d - t - 1) + T - d + t + 1 for t in range(d - 2)]

    gammas = [gamma ** i for i in range(d)]
    r1 = np.vstack([rewards[i] for i in index1])
    r2 = np.vstack([np.r_[rewards[i], np.zeros(d - i.size)] for i in index2])
    r = np.vstack((r1, r2))
    discounted_r = r.dot(np.array(gammas))

    # log pies
    lp1 = np.vstack([log_pies[i] for i in index1])
    lp2 = np.vstack([np.r_[log_pies[i], np.zeros(d - i.size)] for i in index2])
    lp = np.vstack((lp1, lp2))
    g = lp.dot(np.array(gammas))

    exp = [d] * (T - d + 1) + [d - t - 1 for t in range(d - 2)]
    v_ = np.vstack([values[[i[0], i[-1]], 0] for i in index1 + index2])
    v = np.sum(v_ * np.array([[-1, e] for e in exp]), axis=1)
    return v + discounted_r - tau * g


Batch = namedtuple("Batch", ["state", "action", "consistency", "terminal"])


def process_rollout(rollout, d, gamma, tau, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    :param rollout:
    :param d:
    :param gamma: discount ratio
    :param lambda_:
    :return:
    """
    batch_states = np.asarray(rollout.states)
    batch_actions = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])
    values = np.asarray(rollout.values)
    log_pies = np.asarray(rollout.log_pies)

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    batch_consistency = consistency(values, rewards, log_pies, rollout.T,
                                    d, gamma, tau)

    features = rollout.features[0]
    return Batch(batch_states, batch_actions, batch_consistency, rollout.terminal)


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """

    def __init__(self):
        self.states = []
        self.log_pies = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.T = 0

    def add(self, state, log_pi, action, reward, value, terminal, features):
        self.states += [state]
        self.log_pies += [log_pi]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        self.T += 1

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.log_pies.extend(other.log_pies)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        self.T += other.T


def log_softmax(action_logit, one_hot_action):
    log_distrib = action_logit - np.log(np.sum(np.exp(action_logit)))
    log_pi = log_distrib[0, one_hot_action.argmax()]
    return log_pi


def env_runner(sess, env, policy, summary_writer, visualize, max_step_per_episode):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()
    step = 0
    episode_reward = 0
    terminal_end = False
    rollout = PartialRollout()

    while not terminal_end:
        step += 1

        action_logit, action, value_ = policy.act(last_state, *last_features)
        # action_distrib = action_logit / np.sum(action_logit)
        # log_pi = np.log(action_distrib[0, np.argmax(action)])
        log_pi = log_softmax(action_logit, action)
        # argmax to convert from one-hot
        state, reward, terminal, info = env.step(action.argmax())
        if visualize:
            env.render()

        # collect the experience
        rollout.add(state, log_pi, action, reward, value_, terminal, last_features)

        episode_reward += reward
        last_state = state
        # last_features = features

        # if info:
        #     summary = tf.Summary()
        #     for k, v in info.items():
        #         summary.value.add(tag=k, simple_value=float(v))
        #     summary_writer.add_summary(summary, policy.global_step.eval())
        #     summary_writer.flush()

        if terminal or step >= max_step_per_episode:
            terminal_end = True
            if step >= max_step_per_episode or not env.metadata.get('semantics.autoreset'):
                last_state = env.reset()
            last_features = policy.get_initial_features()
            # print("reward: %d. step: %d" % (episode_reward, step))
            break

    # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
    return rollout


def tf_stack(max_, inner_body, init, dim):
    """
    utility function for stacking tensor like python list comprehension
    :param max_:
    :param inner_body:
    :param init:
    :param dim:
    :return:
    """
    cond = lambda i, m: tf.less(i, max_)
    body = lambda i, m: [i + 1,
                         tf.concat([m, tf.reshape(inner_body(i), (1, dim))], axis=0)]
    i = tf.constant(1)
    init = tf.reshape(init, (1, dim))
    _, stacked_tensor = tf.while_loop(cond, body, loop_vars=[i, init],
                                      shape_invariants=[i.get_shape(), tf.TensorShape([None, dim])])
    return stacked_tensor


class PCL(object):
    def __init__(self, env, args):
        self.env = env
        self.max_step_per_episode = args.max_step_per_episode
        self.visualise = args.visualise
        self.pi = pi = LinearPolicy(env.observation_space.shape, env.action_space.n)
        self.action = tf.placeholder(tf.float32, [None, env.action_space.n], name="action")
        # self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.consistency = tf.placeholder(tf.float32, [None],
                                          name="consistency")
        self.values = pi.values
        # self.r = tf.placeholder(tf.float32, [None], name="r")
        self.queue = queue.Queue(5)
        self.d = args.d
        self.gamma = args.gamma
        self.tau = args.tau

        log_prob_tf = tf.nn.log_softmax(pi.logits)
        prob_tf = tf.nn.softmax(pi.logits)

        log_pi = tf.reduce_sum(log_prob_tf * self.action, [1])
        T = tf.shape(self.consistency)[0] + 1

        # Making sliding slice indexes
        # cond1 = lambda i, m:\
        #     i < T - self.d + 1
        # body1 = lambda i, m:\
        #     [i + 1,
        #      tf.concat([m, tf.reshape(tf.range(self.d) + i, (1, self.d))], axis=0)]
        # i1 = tf.constant(1)
        # m1 = tf.reshape(tf.range(self.d), (1, self.d))
        # _, index1 = tf.while_loop(cond1, body1, loop_vars=[i1, m1],
        #                         shape_invariants=[i1.get_shape(), tf.TensorShape([None, self.d])])
        body1 = lambda i: tf.range(self.d) + i
        index1 = tf_stack(T - self.d + 1, body1, tf.range(self.d), self.d)

        # cond2 = lambda i, m: i < self.d - 2
        # body2 = lambda i, m: [i + 1,
        #                       tf.concat([m,
        #                                  tf.expand_dims(tf.concat(
        #                                      [tf.range(self.d - i - 1) + T - self.d + i + 1, -tf.ones((i + 1,), tf.int32)]
        #                                      , axis=0)
        #                                                 , axis=0)]
        #                                 , axis=0)]
        # i2 = tf.constant(1)
        # m2 = tf.expand_dims(tf.concat([tf.range(self.d - 1) + T - self.d + 1, -tf.ones(1, tf.int32)], axis=0), axis=0)
        # _, index2 = tf.while_loop(cond2, body2, loop_vars=[i2, m2],
        #                         shape_invariants=[i2.get_shape(), tf.TensorShape([None, self.d])])
        body2 = lambda i: \
            tf.concat([tf.range(self.d - i - 1) + T - self.d + i + 1, -tf.ones((i + 1,),
                                                                               tf.int32)], axis=0)
        init_m2 = tf.concat([tf.range(self.d - 1) + T - self.d + 1, -tf.ones(1, tf.int32)], axis=0)
        index2 = tf_stack(self.d - 2, body2, init_m2, self.d)

        # log_pies1 = tf.stack([tf.gather(log_pi, i) for i in index1])
        # log_pies2 = tf.stack([tf.concat([tf.gather(log_pi, i),
        #                                  tf.zeros(self.d - i.size)], 0)
        #                       for i in index2])
        log_body1 = lambda i: tf.gather(log_pi, tf.gather(index1, i))
        init_m1 = tf.gather(log_pi, tf.gather(index1, 0))
        log_pies1 = tf_stack(tf.shape(index1)[0], log_body1, init_m1, self.d)

        log_body2 = lambda i: \
            tf.concat([tf.gather(log_pi, tf.gather(tf.gather(index2, i), tf.range(self.d - i - 1))),
                       tf.zeros((i + 1,))
                       ], 0)

        init_m2 = tf.concat([tf.gather(log_pi, tf.gather(tf.gather(index2, 0), tf.range(self.d - 1))),
                             tf.zeros(1)], axis=0)
        log_pies2 = tf_stack(tf.shape(index2)[0], log_body2, init_m2, self.d)
        log_pies = tf.concat([log_pies1, log_pies2], 0)

        # Calculation of pi loss
        gammas = tf.constant([[args.gamma ** i] for i in range(self.d)])
        g = tf.matmul(log_pies, gammas)
        pi_loss = tf.reduce_sum(self.consistency * g)

        exp = [self.d] * (T - self.d + 1) \
              + [self.d - t - 1 for t in range(self.d - 2)]
        # gamma_t1 = tf.tile(np.array([[self.gamma ** self.d]]), (T, 1))
        gamma_t1 = tf.cast(tf.tile(np.array([[self.gamma ** self.d]]), [T - self.d + 1, 1]), tf.float32)
        gamma_body = lambda i: tf.reshape(self.gamma ** tf.cast(self.d - i - 1, tf.float32),
                                          (1, 1))
        gamma_init = self.gamma ** (self.d - 1)
        gamma_t2 = tf_stack(self.d - 2, gamma_body, gamma_init, 1)
        gamma = tf.concat([gamma_t1, gamma_t2], axis=0)
        gamma_t = tf.concat([tf.ones((T - 1, 1)), gamma], axis=1)
        # v_ = tf.stack([tf.gather(self.values, [[i[0], i[-1]]])
        #                 for i in index1 + index2])
        v1_init = tf.gather(self.values, [0, self.d - 1])
        v1_body = lambda i: \
            tf.gather(self.values, [tf.gather(index1, i)[0], tf.gather(index1, i)[self.d - 1]])
        v1_ = tf_stack(T - self.d + 1, v1_body, v1_init, 2)
        v2_init = tf.gather(self.values, [T - 2, T - 2 + self.d - 2])
        v2_body = lambda i: \
            tf.gather(self.values, [tf.gather(index2, i)[0], tf.gather(index2, i)[self.d - 1 - i]])
        v2_ = tf_stack(self.d - 2, v2_body, v2_init, 2)
        v_ = tf.concat([v1_, v2_], axis=0)
        v = tf.reduce_sum(v_ * gamma_t, axis=1)
        value_loss = tf.reduce_sum(self.consistency * v)

        grad_theta = tf.gradients(pi_loss, pi.theta)
        grad_phi = tf.gradients(value_loss, pi.phi)
        grads = grad_theta + grad_phi
        grads_and_vars = list(zip(grads, pi.theta + pi.phi))

        # the "policy gradients" loss:  its derivative is precisely the policy gradient
        # notice that self.ac is a placeholder that is provided externally.
        # adv will contain the advantages, as calculated in process_rollout
        # pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)
        #
        # # loss of value function
        # vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
        # entropy = - tf.reduce_sum(prob_tf * log_prob_tf)
        #
        # bs = tf.to_float(tf.shape(pi.x)[0])
        # self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

        # 20 represents the number of "local steps":  the number of timesteps
        # we run the policy before we update the parameters.
        # The larger local steps is, the lower is the variance in our policy gradients estimate
        # on the one hand;  but on the other hand, we get less frequent parameter updates, which
        # slows down learning.  In this code, we found that making local steps be much
        # smaller than 20 makes the algorithm more difficult to tune and to get to work.

        # self.runner = RunnerThread(env, pi, 20, args.visualise)

        # grads = tf.gradients(self.loss, pi.var_list)

        # tf.summary.scalar("model/policy_loss", pi_loss / bs)
        # tf.summary.scalar("model/value_loss", vf_loss / bs)
        # tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.image("model/state", pi.x)
        tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        # tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
        self.summary_op = tf.summary.merge_all()

        # grads, _ = tf.clip_by_global_norm(grads, 40.0)
        # grads_and_vars = list(zip(grads, self.pi.var_list))

        # each worker has a different set of adam optimizer parameters
        opt = tf.train.AdamOptimizer(1e-4)
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.summary_writer = None
        self.local_steps = 0

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.queue.get(timeout=5000.0)
        # Retrieve one episode
        while not rollout.terminal:
            try:
                rollout.extend(self.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess, visualise):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        rollout = env_runner(sess, self.env, self.pi, self.summary_writer,
                             visualise, self.max_step_per_episode)

        # self.queue.put(rollout, timeout=1.0)
        #
        # rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, self.d, self.gamma, self.tau, lambda_=1.0)

        feed_dict = {
            self.pi.x: batch.state,
            self.action: batch.action,
            self.consistency: batch.consistency
            # self.r: batch.r
        }

        fetched = sess.run(self.train_op, feed_dict=feed_dict)

        # if should_compute_summary:
        #     self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
        #     self.summary_writer.flush()
        loss = np.sum(batch.consistency ** 2)
        if visualise:
            print("reward : {0:.3}, loss : {1:.3}".format(np.sum(rollout.rewards), loss))
        self.local_steps += 1


if __name__ == "__main__":
    tf.app.run()
