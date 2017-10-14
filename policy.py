import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import distutils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, InputLayer, Input
from keras import backend as K
from keras.layers import Input, Lambda

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


class LinearPolicy(object):
    def __init__(self, ob_space, ac_space, n_hidden = 50, is_policy_network=True):
        self.n_hidden = n_hidden
        if is_policy_network:
            self.action_dim, self.action_decoder = get_action_space(ac_space)
            with tf.variable_scope("policy"):
                # First dimension is the number of steps in an episode
                model, self.obs_shape = preprocess_observation_space(ob_space)
                self._build_policy_network(model)
        else:
            with tf.variable_scope("value"):
                model, self.obs_shape = preprocess_observation_space(ob_space)
                self._build_value_network(model)
        self.x = model.input
        self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def get_initial_features(self, _):
        return [None, None]

    def act(self, obsevation, *_):
        sess = tf.get_default_session()
        return sess.run({"logit": self.logits}, {self.x: obsevation})

    def value(self, obsevation, *_):
        sess = tf.get_default_session()
        return sess.run({"value": self.values}, {self.x: obsevation})

    def _build_value_network(self, model):
        model.add(Dense(self.n_hidden))
        model.add(Activation("relu"))
        # hidden_v = relu(x, 50, "hidden0", normalized_columns_initializer())
        for i in range(1):
            model.add(Dense(self.n_hidden, activation="relu"))
            # hidden_v = linear(hidden_v, 50, "hidden{}".format(i+1), normalized_columns_initializer())
        # self.values = tf.reshape(linear(hidden_v, 1, "value", normalized_columns_initializer()), [-1])
        model.add(Dense(1, activation="linear"))
        self.values = model.output

    def _build_policy_network(self, model):
        # hidden_pi = relu(x, 50, "hidden0", normalized_columns_initializer())
        # i = 0
        model.add(Dense(self.n_hidden, activation="relu"))
        for i in range(0):
            model.add(Dense(self.n_hidden, activation="relu"))
        model.add(Dense(self.action_dim, activation="softmax"))
        self.logits = model.output


class LSTMPolicy(LinearPolicy):
    def __init__(self, ob_space, ac_space, n_hidden=50, size=128, is_policy_network=True):
        self.n_hidden = n_hidden
        if is_policy_network:
            self.action_dim, self.action_decoder = get_action_space(ac_space)
            with tf.variable_scope("policy"):
                model, self.obs_shape = preprocess_observation_space(ob_space)
                x = self._build_lstm_network(model, size)

                m = Sequential()
                m.add(InputLayer(input_tensor=x))
                self._build_policy_network(m)
        else:
            with tf.variable_scope("value"):
                model, self.obs_shape = preprocess_observation_space(ob_space)
                x = self._build_lstm_network(model, size)

                m = Sequential()
                m.add(InputLayer(input_tensor=x))
                self._build_value_network(m)
        self.x = model.input
        self.variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def _build_lstm_network(self, model, size):
        x = model.output

        if use_tf100_api:
            cell = rnn.BasicLSTMCell(size, state_is_tuple=True)
        else:
            cell = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = cell.state_size
        # The first dimension is batch size
        c_in = tf.placeholder(tf.float32, [None, cell.state_size.c])
        h_in = tf.placeholder(tf.float32, [None, cell.state_size.h])
        self.state_in = [c_in, h_in]

        if use_tf100_api:
            state_in = rnn.LSTMStateTuple(c_in, h_in)
        else:
            state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, self.features = tf.nn.dynamic_rnn(
            cell, x, initial_state=state_in,
            time_major=False)
        return x

    def get_initial_features(self, batch_size):
        # batch size is always 1
        c_init = np.zeros((batch_size, self.state_size.c), np.float32)
        h_init = np.zeros((batch_size, self.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        return self.state_init

    def act(self, observation, c, h):
        # obs_code = self.obs_encoder(observation)
        sess = tf.get_default_session()
        return sess.run({"logit": self.logits, "feature": self.features},
                        {self.x: observation, self.state_in[0]: c, self.state_in[1]: h})

    def value(self, observation, c, h):
        sess = tf.get_default_session()
        # obs_code = self.obs_encoder(observation)
        return sess.run({"value": self.values, "feature": self.features},
                        {self.x: observation, self.state_in[0]: c, self.state_in[1]: h})


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
            obs_dim = list(observation_space.shape[:1])
            print("observation dim :", obs_dim)
            # [batch, seq_length, obs_dim]
            model = Sequential()
            model.add(Dense(n_hidden, input_shape=[None, observation_space.shape[0]], activation="relu"))
            for i in range(2):
                model.add(Dense(n_hidden, activation="relu"))
            return model, obs_dim
        else:
            print("Not implemented yet!")
    # Discrete
    else:
        obs_dim = observation_space.n
        print("observation dim :", obs_dim)

        x_in = Input(shape=[None], dtype="int32")
        # one hot action vector
        x = Lambda(K.one_hot, arguments={"num_classes": obs_dim}, output_shape=[None, obs_dim])(x_in)

        model = Model(inputs=x_in, outputs=x)
        model = Sequential(model.layers)
        model.add(Dense(n_hidden, activation="relu", input_shape=[None, obs_dim]))
        for l in range(2):
            model.add(Dense(n_hidden, activation="relu"))

        return model, []