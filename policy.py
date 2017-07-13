import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils

use_tf100_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('1.0.0')


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b


def categorical_sample(logits, d):
    # 1st input to tf.multinomial, logits is the unnormalized log probabilities for all classes
    value = tf.squeeze(
        # tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1),
        tf.multinomial(tf.nn.log_softmax(logits - tf.reduce_max(logits, [1], keep_dims=True))*5, 1),
        [1])
    return tf.one_hot(value, d)


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, size=256):
        with tf.variable_scope("common"):
            self.action_dim, self.action_decoder = get_action_space(ac_space)
            self.x, x = preprocess_observation_space(ob_space)

            if use_tf100_api:
                lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)
            else:
                lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = lstm.state_size
            step_size = tf.shape(self.x)[:1]

            c_init = np.zeros((1, lstm.state_size.c), np.float32)
            h_init = np.zeros((1, lstm.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm, x, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.reshape(lstm_outputs, [-1, size])

        self.logits = linear(x, self.action_dim, "theta", normalized_columns_initializer(0.01))
        self.values = tf.reshape(linear(x, 1, "phi", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        # one-hot vector
        self.sample = categorical_sample(self.logits, self.action_dim)[0, :]
        common_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "common")
        self.theta = common_var + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "theta")
        self.phi = common_var + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phi")
        # self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def get_initial_features(self):
        return self.state_init

    def act(self, state, c, h):
        sess = tf.get_default_session()
        return sess.run([self.logits, self.sample, self.values, self.state_out],
                        {self.x: [state], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.values, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class LinearPolicy(object):
    def __init__(self, ob_space, ac_space, n_hidden = 10):
        self.n_hidden = n_hidden
        # First dimension is the number of steps in an episode
        self.action_dim, self.action_decoder = get_action_space(ac_space)

        self.x, x = self.preprocess_observation_space(ob_space)
        self.logits = linear(x, self.action_dim, "theta",
                             normalized_columns_initializer(0.01))
        self.values = tf.reshape(linear(x, 1, "phi",
                                        normalized_columns_initializer(1.0)), [-1])
        self.sample = categorical_sample(self.logits, self.action_dim)[0, :]
        self.theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "theta")
        self.phi = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phi")

    def get_initial_features(self):
        return [None, None]

    def act(self, state, *_):
        sess = tf.get_default_session()
        return sess.run([self.logits, self.sample, self.values], {self.x: [state]})

    def value(self, state, *_):
        sess = tf.get_default_session()
        return sess.run(self.values, {self.x: [state]})[0]

    def preprocess_observation_space(self, observation_space):
        if isinstance(observation_space, gym.spaces.box.Box):
            # If observation is an image, add CNN
            if len(observation_space.shape) is 3 and observation_space.shape[-1] is 3:
                x_placeholder = tf.placeholder(tf.float32, [None] + list(observation_space.shape),
                                               name="observation")
                x = x_placeholder
                for i in range(4):
                    x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
                # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
                x = flatten(x)
                return x_placeholder, x
            # e.g. CartPole
            elif len(observation_space.shape) is 1:
                print("observation dim :", observation_space.shape[0])
                x = x_placeholder = tf.placeholder(tf.float32, [None, observation_space.shape[0]])
                x = linear(x, self.n_hidden, 'common')
                return x_placeholder, x
            else:
                print("Not implemented yet!")
        # Discrete
        else:
            obs_dim = observation_space.n
            print("observation dim :", obs_dim)
            x_placeholder = tf.placeholder(tf.int32, [None], name="observation")
            x = tf.one_hot(x_placeholder, obs_dim, axis=1)
            for l in range(3):
                x = linear(x, self.n_hidden, "common{}".format(l))
            return x_placeholder, x


def get_action_space(action_space):
    # If action space is multidimension
    if isinstance(action_space, gym.spaces.tuple_space.Tuple):
        action_space_dims = [d.n for d in action_space.spaces]
        action_space_dim = np.prod(action_space_dims)

        def action_decoder(action):
            return np.unravel_index(action.argmax(), action_space_dims)

        return action_space_dim, action_decoder
    else:
        action_space_dim = action_space.n

        def action_decoder(action):
            return action.argmax()
        return action_space_dim, action_decoder

def preprocess_observation_space(observation_space, n_hidden=32):
    # If observation is an image, add CNN
    if isinstance(observation_space, gym.spaces.box.Box):
        if len(observation_space.shape) is 3 and\
            observation_space.shape[-1] is 3:
            x_placeholder = tf.placeholder(tf.float32, [None] + list(observation_space.shape),
                                           name="observation")
            x = x_placeholder
            for i in range(4):
                x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
            # introduce a "fake" batch dimension of 1 after flatten so that we can do LSTM over time dim
            x = tf.expand_dims(flatten(x), [0])
            return x_placeholder, x
        # e.g. CartPole
        elif len(observation_space.shape) is 1:
            print("observation dim :", observation_space.shape[0])
            x = x_placeholder = tf.placeholder(tf.float32, [None, observation_space.shape[0]])
            x = linear(x, n_hidden, 'common')
            x = tf.expand_dims(x, [0])
            return x_placeholder, x
        else:
            print("Not implemented yet!")

    else:
        obs_dim = observation_space.n
        print("observation dim :", obs_dim)
        x_placeholder = tf.placeholder(tf.int32, [None], name="observation")
        # one hot action vector
        x = tf.one_hot(x_placeholder, obs_dim, axis=1)
        for l in range(2):
            x = linear(x, n_hidden, "common{}".format(l))
        x = tf.expand_dims(x, [0])
        return x_placeholder, x