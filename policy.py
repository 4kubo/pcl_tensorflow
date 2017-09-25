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

def relu(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    y = tf.matmul(x, w) + b
    return tf.nn.relu(y, "relu")


class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, size=128):
        with tf.variable_scope("common"):
            self.action_dim, self.action_decoder = get_action_space(ac_space)
            self.x, x = preprocess_observation_space(ob_space)
            # introduce a "fake" batch dimension of 1 after flatten so that
            #  we can do LSTM over time dim
            # x is converted in shape of [1, time step, ob_space]
            x = tf.expand_dims(x, [0])

            if use_tf100_api:
                cell = rnn.BasicLSTMCell(size, state_is_tuple=True)
            else:
                cell = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
            self.state_size = cell.state_size

            # batch size is always 1
            c_init = np.zeros((1, cell.state_size.c), np.float32)
            h_init = np.zeros((1, cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            c_in = tf.placeholder(tf.float32, [1, cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, cell.state_size.h])
            self.state_in = [c_in, h_in]

            if use_tf100_api:
                state_in = rnn.LSTMStateTuple(c_in, h_in)
            else:
                state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                cell, x, initial_state=state_in,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            x = tf.squeeze(lstm_outputs, axis=0)

        with tf.variable_scope("theta"):
            hidden_theta = relu(x, 50, "dense_0")
            i = 0
            for i in range(0):
                hidden_theta = relu(hidden_theta, 50, "hidden{}".format(i+1))
            last = relu(hidden_theta, self.action_dim, "dense_1")
            self.logits = tf.nn.softmax(last, name="softmax")

        with tf.variable_scope("phi"):
            hidden_phi = relu(x, 50, "dense_0")
            for i in range(1):
                hidden_phi = relu(hidden_phi, 50, "hidden{}".format(i+1))
            self.values = tf.reshape(relu(hidden_phi, 1, "dense_1"), [-1])
        self.features = [lstm_c[:1, :], lstm_h[:1, :]]
        common_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "common")
        self.theta = common_var + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "theta")
        self.phi = common_var + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phi")

    def get_initial_features(self):
        return self.state_init

    def act(self, state, c, h):
        sess = tf.get_default_session()
        return sess.run({"logits": self.logits, "features": self.features},
                        {self.x: [state], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.values,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})


class LinearPolicy(object):
    def __init__(self, ob_space, ac_space, n_hidden = 50):
        self.n_hidden = n_hidden
        # First dimension is the number of steps in an episode
        with tf.variable_scope("common"):
            self.action_dim, self.action_decoder = get_action_space(ac_space)
            self.x, x = preprocess_observation_space(ob_space, n_hidden)

        with tf.variable_scope("theta"):
            hidden_pi = relu(x, 50, "hidden0")
            i = 0
            for i in range(0):
                hidden_pi = relu(hidden_pi, 50, "hidden{}".format(i+1))
            hidden_pi = relu(hidden_pi, self.action_dim, "hidden{}".format(i+1))
            self.logits = tf.nn.softmax(hidden_pi, name="softmax")

        with tf.variable_scope("phi"):
            hidden_v = relu(x, 50, "hidden0")
            for i in range(0):
                hidden_v = relu(hidden_v, 50, "hidden{}".format(i+1))
            self.values = tf.reshape(relu(hidden_v, 1, "value"), [-1])

        # Collecting trainable variables
        common_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "common")
        self.theta = common_var + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "theta")
        self.phi = common_var + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "phi")

    def get_initial_features(self):
        return [None, None]

    def act(self, state, *_):
        sess = tf.get_default_session()
        return sess.run({"logits": self.logits, "values": self.values}, {self.x: [state]})

    def value(self, state, *_):
        sess = tf.get_default_session()
        return sess.run(self.values, {self.x: [state]})


def get_action_space(action_space):
    # If action space is multidimension
    if isinstance(action_space, gym.spaces.tuple_space.Tuple):
        action_space_dims = [d.n for d in action_space.spaces]
        action_space_dim = np.prod(action_space_dims)

        def action_decoder(action):
            return np.unravel_index(action.argmax(), action_space_dims)

        return action_space_dim, action_decoder
    # For Pendulum
    elif isinstance(action_space, gym.spaces.box.Box)\
        and action_space.shape == (1,):
        print("action space : Box(1,)")
        action_space_dim = 20
        bins = np.linspace(action_space.low, action_space.high, action_space_dim)

        def action_decoder(action):
            return np.array([bins[action.argmax()]])
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
            x = flatten(x)
            return x_placeholder, x
        # e.g. CartPole
        elif len(observation_space.shape) is 1:
            print("observation dim :", observation_space.shape[0])
            x = x_placeholder = tf.placeholder(tf.float32, [None, observation_space.shape[0]])
            for i in range(2):
                x = relu(x, n_hidden, 'common{}'.format(i))
            return x_placeholder, x
        else:
            print("Not implemented yet!")
    # Discrete
    else:
        obs_dim = observation_space.n
        print("observation dim :", obs_dim)
        x_placeholder = tf.placeholder(tf.int32, [None], name="observation")
        # one hot action vector
        x = tf.one_hot(x_placeholder, obs_dim, axis=1)
        for l in range(2):
            x = relu(x, n_hidden, "common{}".format(l))
        return x_placeholder, x