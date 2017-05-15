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
    args = parser.parse_args()
    return args


def main(_):
    args = arg_parse()

    env = gym.make("CartPole-v0")
    args.max_step_per_episode = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

    n_total_step = 10000
    total_step = 0

    model = PCL(env, args)

    init_all_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_all_op)
        model.start(sess)

        # env = wrappers.Monitor(env, "/Users/kubo-a/tmp/cart")
        while not should_stop(env, n_total_step, total_step):
            model.process(sess)
            total_step += 1


def should_stop(env, n_total_step, total_step):
    if n_total_step <= total_step:
        return True
    else:
        return False


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


def process_rollout(rollout, gamma, lambda_=1.0):
    """
    given a rollout, compute its returns and the advantage
    """
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_.  This means that there should be a thread
    that would constantly interact with the environment and tell it what to do.  This thread is here.
    """

    def __init__(self, env, policy, visualise, max_step_per_episode):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.env = env
        self.last_features = None
        self.policy = policy
        self.summary_writer = None
        self.visualise = visualise
        self.max_step_per_episode = max_step_per_episode

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default() as sess:
            self._run(sess)

    def _run(self, sess):
        rollout_provider = env_runner(sess, self.env, self.policy, self.summary_writer,
                                      self.visualise, self.max_step_per_episode)

        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(rollout_provider.__next__(), timeout=1.0)


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

    while not terminal_end:
        step += 1
        rollout = PartialRollout()

        fetched = policy.act(last_state, *last_features)
        action, value_, features = fetched[0], fetched[1], fetched[2:]
        # argmax to convert from one-hot
        state, reward, terminal, info = env.step(action.argmax())
        if visualize:
            env.render()

        # collect the experience
        rollout.add(last_state, action, reward, value_, terminal, last_features)

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
            print("Episode finished. Sum of reward: %d. step: %d" % (episode_reward, step))
            break

    # if not terminal_end:
    #     rollout.r = policy.value(last_state, *last_features)

    # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
    yield rollout


class PCL(object):
    def __init__(self, env, args):
        self.env = env
        self.max_step_per_episode = args.max_step_per_episode
        self.pi = pi = LinearPolicy(env.observation_space.shape, env.action_space.n)
        self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")

        log_prob_tf = tf.nn.log_softmax(pi.logits)
        prob_tf = tf.nn.softmax(pi.logits)

        # the "policy gradients" loss:  its derivative is precisely the policy gradient
        # notice that self.ac is a placeholder that is provided externally.
        # adv will contain the advantages, as calculated in process_rollout
        pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

        # loss of value function
        vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
        entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

        bs = tf.to_float(tf.shape(pi.x)[0])
        self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

        # 20 represents the number of "local steps":  the number of timesteps
        # we run the policy before we update the parameters.
        # The larger local steps is, the lower is the variance in our policy gradients estimate
        # on the one hand;  but on the other hand, we get less frequent parameter updates, which
        # slows down learning.  In this code, we found that making local steps be much
        # smaller than 20 makes the algorithm more difficult to tune and to get to work.
        self.runner = RunnerThread(env, pi, 20, args.visualise)

        grads = tf.gradients(self.loss, pi.var_list)

        tf.summary.scalar("model/policy_loss", pi_loss / bs)
        tf.summary.scalar("model/value_loss", vf_loss / bs)
        tf.summary.scalar("model/entropy", entropy / bs)
        tf.summary.image("model/state", pi.x)
        tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
        tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
        self.summary_op = tf.summary.merge_all()

        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        grads_and_vars = list(zip(grads, self.pi.var_list))

        # each worker has a different set of adam optimizer parameters
        opt = tf.train.AdamOptimizer(1e-4)
        self.train_op = opt.apply_gradients(grads_and_vars)
        self.summary_writer = None
        self.local_steps = 0

    def start(self, sess, summary_writer=None):
        self.runner.start_runner(sess, summary_writer)
        # self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
        self explanatory:  take a rollout from the queue of the thread runner.
        """
        rollout = self.runner.queue.get(timeout=5000.0)
        # Retrieve one episode
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """

        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        feed_dict = {
            self.pi.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
        }

        fetched = sess.run(self.train_op, feed_dict=feed_dict)

        # if should_compute_summary:
        #     self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
        #     self.summary_writer.flush()
        self.local_steps += 1


if __name__ == "__main__":
    tf.app.run()
