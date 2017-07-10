import gym
# from gym import wrappers
from collections import namedtuple
import numpy as np
import tensorflow as tf
import six.moves.queue as queue
import os
from argparse import ArgumentParser
# import threading
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
                        default=0.9)
    parser.add_argument("--tau",
                        type=float,
                        default=0.01)
    parser.add_argument("--d",
                        type=int,
                        default=3)
    parser.add_argument("--task",
                        type=int,
                        default=0)
    parser.add_argument("--logdir",
                        type=str,
                        default="/tmp/pcl")
    parser.add_argument("--summary",
                        action="store_true")
    parser.add_argument("--target_task",
                        type=str,
                        default="CartPole-v0")
    parser.add_argument("--is_lstm",
                        action="store_true")
    parser.add_argument("--save_model",
                        type=str,
                        default=None)
    parser.add_argument("--restore_model",
                        type=str,
                        default=None,
                        help="Specify the location from which restore model")
    args = parser.parse_args()
    return args


def main(_):
    args = arg_parse()

    env = gym.make(args.target_task)
    args.max_step_per_episode = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    total_step = 0

    model = PCL(env, args)

    init_all_op = tf.global_variables_initializer()
    logdir = os.path.join(args.logdir, "train")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir + "_{}".format(args.task), sess.graph)\
            if args.summary else None
        sess.run(init_all_op)
        model.start(sess, summary_writer)
        if args.restore_model is not None:
            ckpt = tf.train.get_checkpoint_state(args.restore_model)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded from {0}".format(ckpt.model_checkpoint_path))
            else:
                print("Does not exist : {}".format(ckpt.model_checkpoint_path))

        # env = wrappers.Monitor(env, "/Users/kubo-a/tmp/cart")
        while total_step <= args.n_total_step:
            if args.visualise:
                visualise = True if total_step % 1000 == 0 else False
            else:
                visualise = False
            if total_step % 100 == 0:
                report = True
                if args.save_model is not None:
                    saver.save(sess, args.save_model + "/pcl_model.ckpt",
                               global_step=total_step)
            else:
                report = False
            model.process(sess, total_step, visualise, report)
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
    gamma_d = np.power(gamma, exp)
    v_ = np.vstack([values[[i[0], i[-1]], 0] for i in index1 + index2])
    v = np.sum(v_*np.c_[-np.ones(T-1), gamma_d], axis=1)
    return v + discounted_r - tau * g


Batch = namedtuple("Batch", ["state", "action", "consistency", "terminal", "features"])


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
    values = np.asarray(rollout.values)
    log_pies = np.asarray(rollout.log_pies)
    batch_consistency = consistency(values, rewards, log_pies, rollout.T,
                                    d, gamma, tau)

    features = rollout.features[0]
    return Batch(batch_states, batch_actions, batch_consistency, rollout.terminal, features)


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


def env_runner(sess, env, policy, max_step_per_episode,
               summary_writer=None, visualize=False):
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
    features = None

    while not terminal_end:
        step += 1

        fetches = policy.act(last_state, *last_features)
        action_logit, action, value_ = fetches[0], fetches[1], fetches[2:]
        if isinstance(value_, list):
            features = value_[1]
            value_ = value_[0]
        log_pi = log_softmax(action_logit, action)
        # argmax to convert from one-hot
        action_code = policy.action_decoder(action)
        state, reward, terminal, info = env.step(action_code)
        if visualize:
            env.render()

        # collect the experience
        rollout.add(last_state, log_pi, action, reward, value_, terminal, last_features)

        episode_reward += reward
        last_state = state
        last_features = features

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
        self.is_lstm = args.is_lstm
        self.max_step_per_episode = args.max_step_per_episode
        self.visualise = args.visualise
        if args.is_lstm:
            self.pi = pi = LSTMPolicy(env.observation_space, env.action_space)
        else:
            self.pi = pi = LinearPolicy(env.observation_space, env.action_space)
        self.action = tf.placeholder(tf.float32, [None, pi.action_dim], name="action")
        self.consistency = tf.placeholder(tf.float32, [None],
                                          name="consistency")
        self.values = pi.values
        self.queue = queue.Queue(5)
        self.d = args.d
        self.gamma = args.gamma
        self.tau = args.tau
        self.local_steps = 0

        log_prob_tf = tf.nn.log_softmax(self.pi.logits)
        log_pi = tf.reduce_sum(log_prob_tf * self.action, [1])
        T = tf.shape(self.consistency)[0] + 1

        # Making sliding slice indexes
        body1 = lambda i: tf.range(self.d) + i
        index1 = tf_stack(T - self.d + 1, body1, tf.range(self.d), self.d)

        body2 = lambda i: \
            tf.concat([tf.range(self.d - i - 1) + T - self.d + i + 1, -tf.ones((i + 1,),
                                                                               tf.int32)], axis=0)
        init_m2 = tf.concat([tf.range(self.d - 1) + T - self.d + 1, -tf.ones(1, tf.int32)], axis=0)
        index2 = tf_stack(self.d - 2, body2, init_m2, self.d)

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

        gamma_t1 = tf.cast(tf.tile(np.array([[self.gamma ** self.d]]), [T - self.d + 1, 1]), tf.float32)
        gamma_body = lambda i: tf.reshape(self.gamma ** tf.cast(self.d - i - 1, tf.float32),
                                          (1, 1))
        gamma_init = self.gamma ** (self.d - 1)
        gamma_t2 = tf_stack(self.d - 2, gamma_body, gamma_init, 1)
        gamma = tf.concat([gamma_t1, gamma_t2], axis=0)
        gamma_t = tf.concat([tf.ones((T - 1, 1)), -gamma], axis=1)
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

        opt = tf.train.AdamOptimizer(1e-4)
        # grad_theta = tf.gradients(pi_loss, pi.theta)
        # grad_phi = tf.gradients(value_loss, pi.phi)
        grad_theta_and_vars = opt.compute_gradients(pi_loss)
        grad_phi_and_vars = opt.compute_gradients(value_loss)
        grads_and_vars = grad_theta_and_vars + grad_phi_and_vars
        grads, vars = list(zip(*grads_and_vars))
        # grads, _ = tf.clip_by_global_norm(grads, 40.0)
        # grads_and_vars = list(zip(grads, pi.theta + pi.phi))

        # bs = tf.to_float(tf.shape(pi.x)[0])

        tf.summary.scalar("loss", tf.reduce_sum(self.consistency**2, axis=0))
        self.summary_op = tf.summary.merge_all()

        # each worker has a different set of adam optimizer parameters
        self.train_op = opt.apply_gradients(grads_and_vars)

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

    def process(self, sess, step, visualise, report):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        rollout = env_runner(sess, self.env, self.pi, self.max_step_per_episode,
                              self.summary_writer, visualise)

        # self.queue.put(rollout, timeout=1.0)
        # rollout = self.pull_batch_from_queue()
        # In the too short step case
        if rollout.T <= self.d:
            return
        batch = process_rollout(rollout, self.d, self.gamma, self.tau, lambda_=1.0)

        feed_dict = {
            self.pi.x: batch.state,
            self.action: batch.action,
            self.consistency: batch.consistency
        }

        if self.is_lstm:
            feed_dict[self.pi.state_in[0]] = batch.features[0]
            feed_dict[self.pi.state_in[1]] = batch.features[1]

        fetches = [self.train_op, self.summary_op] if report else [self.train_op]

        fetched = sess.run(fetches, feed_dict=feed_dict)

        # if should_compute_summary:
        #     self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
        #     self.summary_writer.flush()
        loss = np.mean(batch.consistency ** 2)
        if visualise or report:
            if self.summary_writer is not None:
                self.summary_writer.add_summary(fetched[1])
            print("@{2}; reward : {0:.3}, loss : {1:.3}".format(np.sum(rollout.rewards),
                                                                loss, step))
        self.local_steps += 1

    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer


if __name__ == "__main__":
    tf.app.run()
