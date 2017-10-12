import gym
import gym_maze
from collections import namedtuple
import numpy as np
import tensorflow as tf
import six.moves.queue as queue
import os
from argparse import ArgumentParser

from policy import LSTMPolicy, LinearPolicy
from replay_buffer import ReplayBuffer


def arg_parse():
    parser = ArgumentParser()
    # Hyper parameters
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
                        default=10)
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        default=100)
    parser.add_argument("-r", "--actor_learning_rate",
                        type=float,
                        default=7e-4)
    parser.add_argument("-c", "--critic_weight",
                        type=float,
                        default=0.1)
    parser.add_argument("-a", "--alpha",
                        type=float,
                        default=0.5)
    parser.add_argument("--clip_min",
                        type=float,
                        default=1e-10)
    parser.add_argument("--cut_end",
                        action="store_false")
    # Configuration
    parser.add_argument("--task",
                        type=int,
                        default=0)
    parser.add_argument("-v", "--visualise",
                        action="store_true")
    parser.add_argument("--step_to_report",
                        type=int,
                        default=100)
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
    # For saving video
    if args.visualise and args.save_model is not None:
        from gym import wrappers
        env = wrappers.Monitor(env, args.logdir, force=True)
    args.max_step_per_episode = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    if args.max_step_per_episode is None:
        args.max_step_per_episode = env.spec.max_episode_steps

    total_step = 0

    model = PCL(env, d=args.d, gamma=args.gamma, tau=args.tau,
                actor_learning_rate=args.actor_learning_rate,
                critic_weight=args.critic_weight, alpha=args.alpha, is_lstm=args.is_lstm,
                visualise=args.visualise, max_step_per_episode=args.max_step_per_episode,
                cut_end=args.cut_end, clip_min=args.clip_min)

    init_all_op = tf.global_variables_initializer()
    logdir = os.path.join(args.logdir, "train")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir + "_{}".format(args.task), sess.graph) \
            if args.summary else None
        sess.run(init_all_op)
        model.start(sess, summary_writer)
        # Restoring a model
        if args.restore_model is not None:
            ckpt = tf.train.get_checkpoint_state(args.restore_model)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded from {0}".format(ckpt.model_checkpoint_path))
            else:
                print("Does not exist : {}".format(ckpt.model_checkpoint_path))

        # Main loop
        while total_step <= args.n_total_step:
            if args.visualise:
                visualise = True if total_step % args.step_to_report == 0 else False
            else:
                visualise = False
            if total_step % args.step_to_report == 0:
                report = True
                if args.save_model is not None:
                    saver.save(sess, args.save_model + "/pcl_model.ckpt",
                               global_step=total_step)
            else:
                report = False
            model.process(sess, total_step, visualise, report,
                          is_lstm=args.is_lstm, batch_size=args.batch_size)
            total_step += 1


def consistency(values, rewards, log_pies, T, d, gamma, tau, is_terminal, cut_end=True):
    """
    Calculate path consistency
    Here, we use end of samples
    :param values:
    :param rewards:
    :param log_pies:
    :param T:
    :param d:
    :param gamma:
    :param tau:
    :return:
    """
    d = d if d < T else T

    # Gamma discounted rewards
    discount_m = np.tril(np.triu(np.ones((T, T))), k=d - 1)
    gamma1 = [[gamma ** i for i in range(d)] for t in range(T - d + 1)]
    gamma2 = [[gamma ** i for i in range(T - t)] for t in range(T - d + 1, T)]
    gammas = reduce(lambda x, y: x + y, gamma1 + gamma2)
    discount_m[0 < discount_m] = gammas
    discounted_rewards = discount_m.dot(rewards)

    # Gamma discounted values
    value_m = -np.eye(T, T + 1) + np.eye(T, T + 1, k=d)
    value_m[1 == value_m] = gamma ** d
    value_m[T - d:, -1] = 0 if is_terminal else [gamma ** (d - i) for i in range(d)]
    # value_m[T - d:, -1] = [gamma ** (d - i) for i in range(d)]
    if cut_end:
        discount_m = discount_m[:T - d + 1, :]
        value_m = value_m[:T - d + 1, :]
        discounted_rewards = discounted_rewards[:T - d + 1]

    discounted_values = value_m.dot(values[:, :, 0, 0])

    # Gamma discounted log pies
    # discount_m[T - d:, :] = 0
    g = discount_m.dot(log_pies)

    consistency = discounted_values[:, 0] + discounted_rewards - tau * g
    return consistency, discounted_rewards, discount_m, value_m


Batch = namedtuple("Batch", ["state", "action", "consistency", "terminal", "feature_p",
                             "feature_v", "reward", "discounted_r", "discount_m", "value_m"])



def process_rollout(rollout, d, gamma, tau, cut_end=True):
    """
    Given a rollout, compute its returns and the advantage
    :param rollout:
    :param d:
    :param gamma: discount ratio
    :return:
    """
    batch_states = np.asarray(rollout.states)
    batch_actions = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    values = np.asarray(rollout.values)
    log_pies = np.asarray(rollout.log_pies)
    batch_consistency, discounted_r, discount_m, value_m \
        = consistency(values, rewards, log_pies, rollout.T, d, gamma, tau, rollout.terminal, cut_end)

    feature_p, feature_v = rollout.features
    return Batch(batch_states, batch_actions, batch_consistency, rollout.terminal,
                 feature_p, feature_v, rewards, discounted_r, discount_m, value_m)


class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """

    def __init__(self, initial_state, initial_value, feature_p, feature_v):
        self.states = [initial_state]
        self.log_pies = []
        self.actions = []
        self.rewards = []
        self.values = [initial_value]
        self.r = 0.0
        self.terminal = False
        self.features = [feature_p, feature_v]
        self.T = 0

    def add(self, state, log_pi, action, reward, value, terminal):
        self.states += [state]
        self.log_pies += [log_pi]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
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
        self.T += other.T


def sample_log_pi(action_logit, action_dim, clip_min=1e-10):
    # Clipping to avoid log 0
    np.clip(action_logit, clip_min, 1.0, out=action_logit)
    action_logit = np.squeeze(action_logit)
    # Normalization
    sum_ = np.sum(action_logit)
    pi = action_logit / sum_
    # Sample action from current policy
    action_id = np.random.choice(np.arange(len(pi)), p=pi)
    log_pi = np.log(pi)
    log_pi = log_pi[action_id]
    # Coding of action
    one_hot_action = np.zeros(action_dim)
    one_hot_action[action_id] = 1
    return log_pi, one_hot_action


def env_runner(sess, env, policy_net, value_net, max_step_per_episode,
               summary_writer=None, visualize=False, is_lstm=False):
    """
    The logic of the thread runner.  In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    if visualize:
        env.render()
    last_feature_p = policy_net.get_initial_features()
    last_feature_v = value_net.get_initial_features()
    initial_value = value_net.value(last_state, *last_feature_v)["value"]
    step = 0
    episode_reward = 0
    terminal_end = False
    rollout = PartialRollout(last_state, initial_value, last_feature_p, last_feature_v)
    feature_p = [None, None]
    feature_v = [None, None]

    while not terminal_end:
        step += 1

        fetches = policy_net.act(last_state, *last_feature_p)
        action_logit = fetches["logit"]
        if is_lstm:
            feature_p = fetches["feature"]
        log_pi, action = sample_log_pi(action_logit, policy_net.action_dim)
        # argmax to convert from one-hot
        action_code = policy_net.action_decoder(action)
        state, reward, terminal, info = env.step(action_code)

        fetches = value_net.value(state, *last_feature_v)
        value = fetches["value"]
        if is_lstm:
            feature_v = fetches["feature"]

        if visualize:
            env.render()

        # collect the experience
        state_to_input = state if isinstance(state, int) else state.copy()
        rollout.add(state_to_input, log_pi, action, reward, value, terminal)

        episode_reward += reward

        # For next step
        last_state = state_to_input
        last_feature_p = feature_p
        last_feature_v = feature_v

        if terminal or step >= max_step_per_episode:
            terminal_end = True
            if step >= max_step_per_episode or not env.metadata.get('semantics.autoreset'):
                last_state = env.reset()
            last_feature_p = policy_net.get_initial_features()
            last_feature_v = value_net.get_initial_features()
            break

    # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
    return rollout


class PCL(object):
    def __init__(self, env, d=10, gamma=1.0, tau=0.01, actor_learning_rate=1e-4,
                 critic_weight=0.1, alpha=0.5, is_lstm=False, visualise=False,
                 max_step_per_episode=1000, cut_end=True, clip_min=1e-10):
        self.env = env
        self.d = d
        self.gamma = gamma
        self.tau = tau
        self.actor_learning_rate = actor_learning_rate
        self.is_lstm = is_lstm
        self.max_step_per_episode = max_step_per_episode
        self.visualise = visualise
        self.cut_end = cut_end
        self.clip_min = clip_min
        self.obs_space = env.observation_space
        if self.is_lstm:
            self.policy_network = LSTMPolicy(env.observation_space, env.action_space,
                                     is_policy_network=True)
            self.value_network =  LSTMPolicy(env.observation_space, env.action_space,
                                     is_policy_network=False)
        else:
            self.policy_network = LinearPolicy(env.observation_space, env.action_space,
                                       is_policy_network=True)
            self.value_network = LinearPolicy(env.observation_space, env.action_space,
                                   is_policy_network=False)
        self.action_ph = tf.placeholder(tf.float32, [None, None, self.policy_network.action_dim],
                                        name="action")
        self.discount_m_ph = tf.placeholder(tf.float32, [None, None, None], name="discount_m")
        self.value_m_ph = tf.placeholder(tf.float32, [None, None, None], name="value_m")
        self.reward_ph = tf.placeholder(tf.float32, [None, None], name="reward")
        self.discounted_r_ph = tf.placeholder(tf.float32, [None, None], name="discounted_r")
        self.values = self.value_network.values[:, :, 0]
        self.queue = queue.Queue(5)
        self.local_steps = 0
        self.replay_buffer = ReplayBuffer(alpha=alpha)

        # The length of one episode
        T = tf.shape(self.action_ph)[1]
        d = tf.cond(tf.constant(self.d) < T, lambda: tf.constant(self.d), lambda: T)
        # Calculate log pi from sampled actions
        log_prob_tf = tf.log(tf.clip_by_value(self.policy_network.logits[:, :-1, :],
                                              self.clip_min, 1.0))
        log_pi = tf.reduce_sum(log_prob_tf * self.action_ph, [-1])
        # Discounted action distribution
        g = tf.einsum("ijk,ik->ij", self.discount_m_ph, log_pi)
        # Discounted values
        # discounted_values = tf.reshape(tf.matmul(self.value_m_ph, self.values[:, None]), [-1])
        discounted_values = tf.einsum("ijk,ik->ij", self.value_m_ph, self.values)
        # Path Consistency
        consistency = discounted_values + self.discounted_r_ph - self.tau*g
        self.c = consistency

        # Calculation of entropy for report
        entropy = log_prob_tf * self.policy_network.logits[:, :-1, :]
        entropy = -tf.reduce_mean(tf.reduce_sum(entropy, axis=1))

        # Calculation of losses
        self.pi_loss = tf.reduce_mean(consistency ** 2, axis=1)
        self.v_loss = tf.reduce_mean(consistency ** 2, axis=1) \
                      # + (self.values[-1] - self.reward_ph[-1]) ** 2

        # Entropy regularized reward of sample (err): e.q. (15)
        gammas = tf.pow(gamma, tf.cast(tf.range(T), tf.float32))
        err = tf.reduce_mean(tf.reduce_sum(gammas[None, :] * (self.reward_ph - tau * log_pi), axis=1))

        # Optimizer for policy and value function
        opt_pi = tf.train.AdamOptimizer(actor_learning_rate)
        opt_value = tf.train.AdamOptimizer(actor_learning_rate * critic_weight)

        # Summary
        tf.summary.scalar("loss", tf.reduce_mean(self.pi_loss))
        tf.summary.scalar("reward", tf.reduce_sum(self.reward_ph))
        tf.summary.scalar("entropy regularized reward", err)
        self.summary_op = tf.summary.merge_all()

        self.train_op = [opt_pi.minimize(self.pi_loss, var_list=self.policy_network.variable),
                         opt_value.minimize(self.v_loss, var_list=self.value_network.variable)]

        self.report = {"entropy": entropy, "loss": self.v_loss, "err": err}
        self.summary_op = tf.summary.merge_all()

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

    def process(self, sess, step, visualise, report, is_lstm=False, batch_size=100):
        """
        process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        rollout = env_runner(sess, self.env, self.policy_network, self.value_network,
                             self.max_step_per_episode, self.summary_writer,
                             visualise, is_lstm=is_lstm)


        # self.queue.put(rollout, timeout=1.0)
        # rollout = self.pull_batch_from_queue()

        batch, fetched = self.train(rollout, report, sess)

        # if should_compute_summary:
        #     self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[-1])
        #     self.summary_writer.flush()
        if visualise or report:
            d = self.d if self.d < rollout.T else rollout.T
            loss = fetched["report"]["loss"]
            loss = np.mean(loss) / d
            entropy = fetched["report"]["entropy"]
            erer = fetched["report"]["err"]
            print(
            "@{2: >5}; reward : {0: >8.3}, entropy regularized reward : {4: >8.3}, loss : {1: >8.3}, entropy : {3: >8.3}"
            .format(np.sum(rollout.rewards), loss, step, entropy, erer))

        self.replay_buffer.add(rollout)
        if self.replay_buffer.trainable:
            rollouts = self.replay_buffer.sample(batch_size)
            self._batch_train(rollouts, report, sess)
            # for rollout in rollouts:
            #     batch, fetched = self.train(rollout, False, sess)

        if self.summary_writer is not None:
            self.summary_writer.add_summary(fetched["summary_op"])
        self.local_steps += 1

    def train(self, rollout, report, sess):
        batch = process_rollout(rollout, self.d, self.gamma, self.tau, self.cut_end)

        feed_dict = {
            self.value_network.x: [batch.state],
            self.policy_network.x: [batch.state],
            self.action_ph: [batch.action],
            self.reward_ph: [batch.reward],
            self.discounted_r_ph: [batch.discounted_r],
            self.discount_m_ph: [batch.discount_m],
            self.value_m_ph: [batch.value_m]
        }

        if self.is_lstm:
            feed_dict[self.policy_network.state_in[0]] = batch.feature_p[0]
            feed_dict[self.policy_network.state_in[1]] = batch.feature_p[1]
            feed_dict[self.value_network.state_in[0]] = batch.feature_v[0]
            feed_dict[self.value_network.state_in[1]] = batch.feature_v[1]

        fetches = {"train_op": self.train_op}
        if report or self.summary_writer is not None:
            fetches["report"] = self.report
            fetches["summary_op"] = self.summary_op

        fetched = sess.run(fetches, feed_dict=feed_dict)
        return batch, fetched

    def _batch_train(self, rollouts, report, sess):
        batch_state, batch_action, batch_reward, batch_discounted_r, batch_discount_m, batch_value_m =\
            self._make_batches(rollouts)

        feed_dict = {
            self.value_network.x: batch_state,
            self.policy_network.x: batch_state,
            self.action_ph: batch_action,
            self.reward_ph: batch_reward,
            self.discounted_r_ph: batch_discounted_r,
            self.discount_m_ph: batch_discount_m,
            self.value_m_ph: batch_value_m
        }

        if self.is_lstm:
            feed_dict[self.policy_network.state_in[0]] = batch.feature_p[0]
            feed_dict[self.policy_network.state_in[1]] = batch.feature_p[1]
            feed_dict[self.value_network.state_in[0]] = batch.feature_v[0]
            feed_dict[self.value_network.state_in[1]] = batch.feature_v[1]

        fetches = {"train_op": self.train_op}
        if report or self.summary_writer is not None:
            fetches["report"] = self.report
            fetches["summary_op"] = self.summary_op

        fetched = sess.run(fetches, feed_dict=feed_dict)
        # return batch, fetched

    def _make_batches(self, rollouts):
        batch_size = len(rollouts)
        max_t = max(r.T for r in rollouts)
        obs_shape = list(rollouts[0].states[0].shape)

        batch_list = [process_rollout(rollout, self.d, self.gamma, self.tau, self.cut_end)
                      for rollout in rollouts]

        # Initialization
        batch_state = np.zeros([batch_size, max_t+1] + obs_shape)
        batch_action = np.zeros((batch_size, max_t, self.policy_network.action_dim))
        batch_reward = np.zeros((batch_size, max_t))
        batch_discounted_r = np.zeros((batch_size, max_t))
        batch_discount_m = np.zeros((batch_size, max_t, max_t))
        batch_value_m = np.zeros((batch_size, max_t, max_t+1))

        batch_state = self._padd_1d_batches(batch_state, [b.state for b in batch_list])
        batch_action = self._padd_1d_batches(batch_action, [b.action for b in batch_list])
        batch_reward = self._padd_1d_batches(batch_reward, [b.reward[:, None] for b in batch_list])
        batch_discounted_r = self._padd_1d_batches(batch_discounted_r,
                                                   [b.discounted_r[:, None] for b in batch_list])
        batch_discount_m = self._padd_2d_batches(batch_discount_m,
                                                 [b.discount_m for b in batch_list])
        batch_value_m = self._padd_2d_batches(batch_value_m,
                                                 [b.value_m for b in batch_list])
        return batch_state, batch_action, batch_reward, batch_discounted_r,\
               batch_discount_m, batch_value_m

    def _padd_1d_batches(self, zeros, batch_list):
        """

        :param zeros(np.ndarray): An array to be padded with values in `batch_list`
        :param batch_list:
        :return:
        """
        n = np.prod(zeros.shape[2:]).astype(int)
        t_max = zeros.shape[1] # The maximum length of the episodes in `batch_list`
        # exec("grids=np.meshgrid({})".format(",".join("range({0})".format(i) for i in dims)))
        indexes = np.concatenate([n*t_max*i+np.arange(n*t.shape[0])
                                  for i, t in enumerate(batch_list)])

        data = np.concatenate([np.ravel(b) for b in batch_list])
        batches = np.ravel(zeros)
        batches[indexes] = data
        return batches.reshape(zeros.shape)

    def _padd_2d_batches(self, zeros, batch_list):
        """

        :param zeros(np.ndarray): An array to be padded with values in `batch_list`
        :param batch_list:
        :return:
        """
        _, N, M = zeros.shape
        # n = np.prod(zeros.shape[2:]).astype(int)
        # t_max = zeros.shape[1] # The maximum length of the episodes in `batch_list`

        indexes = np.concatenate([M*N*i + M*n + np.arange(batch.shape[1])
                                  for i, batch in enumerate(batch_list)
                                  for n in range(batch.shape[0])])


        data = np.concatenate([np.ravel(b) for b in batch_list])
        batches = np.ravel(zeros)
        batches[indexes] = data
        return batches.reshape(zeros.shape)

    def start(self, sess, summary_writer):
        self.summary_writer = summary_writer


if __name__ == "__main__":
    tf.app.run()
