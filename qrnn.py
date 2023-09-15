
import sys
print("User Current Version:-", sys.version)

import tensorflow as tf
import pennylane as qml
from numpy import pi

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Encoding
def encode_input(input_vec, q):
    if len(q) < input_vec.shape[0]:
        raise Exception('Not enough wires to encode the input vector')
    for i, x in enumerate(input_vec):
        qml.Displacement(x, 0, wires = q[i])


def encode_input_inv(input_vec, q):
    if len(q) < input_vec.shape[0]:
        raise Exception('Not enough wires to encode the input vector')
    for i, x in enumerate(input_vec):
         qml.Displacement(x, pi, wires = q[i])


# Interferometer
def get_interferometer_params_n(wires_n):
    N = wires_n
    return N * (N - 1) + max(1, N - 1)


def split_interferometer_params(params, wires_n):
    N = wires_n
    theta = params[:N*(N-1)//2]
    phi = params[N*(N-1)//2:N*(N-1)]
    rphi = params[-N+1:]
    return theta, phi, rphi


def interferometer(params, q):
    N = len(q)
    theta, phi, rphi = split_interferometer_params(params, N)

    if N == 1:
        # the interferometer is a single rotation
        qml.Rotation(rphi[0], wires = q[0])
        return

    n = 0  # keep track of free parameters

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                qml.Beamsplitter(theta[n], phi[n], wires = [q1, q2])
                n += 1

    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        qml.Rotation(rphi[i], wires = q[i])


def interferometer_inv(params, q):
    N = len(q)
    theta, phi, rphi = split_interferometer_params(params, N)

    if N == 1:
        # the interferometer is a single rotation
        qml.Rotation(-rphi[0], wires = q[0])
        return

    # apply the final local phase shifts to all modes except the last one
    for i in range(max(1, N - 1)):
        qml.Rotation(-rphi[i], wires = q[i])

    pairs = list()

    # Apply the rectangular beamsplitter array
    # The array depth is N
    for l in range(N):
        for k, (q1, q2) in enumerate(zip(q[:-1], q[1:])):
            # skip even or odd pairs depending on layer
            if (l + k) % 2 != 1:
                pairs.append((q1, q2))

    for i in range(len(theta) - 1, -1, -1):
        qml.Beamsplitter(-theta[i], phi[i], wires = pairs[i])


# Linear layer
def get_linear_layer_params_n(wires_n):
    N = wires_n
    M = N * (N - 1) + max(1, N - 1)
    return 2*M+3*N


def split_linear_layer(params, wires_n):
    N = wires_n
    M = int(N * (N - 1)) + max(1, N - 1)
    int1 = params[:M]
    s = params[M:M+N]
    int2 = params[M+N:2*M+N]
    dr = params[2*M+N:2*M+2*N]
    dp = params[2*M+2*N:2*M+3*N]
    return int1, s, int2, dr, dp


def linear_layer(params, q):
    N = len(q)
    int1, s, int2, dr, dp = split_linear_layer(params, N)

    # begin layer
    interferometer(int1, q)

    for i in range(N):
        qml.Squeezing(s[i], 0, wires = q[i])

    interferometer(int2, q)

    for i in range(N):
        qml.Displacement(dr[i], dp[i], wires = q[i])


def linear_layer_inv(params, q):
    N = len(q)
    int1, s, int2, dr, dp = split_linear_layer(params, N)

    for i in range(N):
        qml.Displacement(dr[i], dp[i] + pi, wires = q[i])

    interferometer_inv(int2, q)

    for i in range(N):
        qml.Squeezing(s[i], pi, wires = q[i])

    interferometer_inv(int1, q)


# Activation layer
def get_activation_layer_params_n(wires_n):
    return wires_n


def activation_layer(params, q):
    N = len(q)
    for i in range(N):
        qml.Kerr(params[i], wires=q[i])


# Measurements
def get_x_quad_expectations(modes_it):
    return  [qml.expval(qml.X(position)) for position in modes_it]


def trace(state):
    return tf.math.reduce_sum(tf.math.abs(state) ** 2)

import numpy as np
def get_quantum_RNN_params_n(working_wires_n, hidden_wires_n):
    params_n = 0
    params_n += get_linear_layer_params_n(working_wires_n)
    params_n += get_linear_layer_params_n(hidden_wires_n)
    # params_n += hidden_wires_n # ControlledPhase
    params_n += hidden_wires_n # activation_like_layer
    return params_n


def split_quantum_RNN_params(params, working_wires_n, hidden_wires_n):
    ending = get_linear_layer_params_n(working_wires_n)
    working_linear_params = params[: ending]

    beginning = ending
    ending += get_linear_layer_params_n(hidden_wires_n)
    hidden_linear_params = params[beginning: ending]

    # beginning = ending
    # ending += hidden_wires_n
    # control_params = params[beginning: ending]
    # control_params = None
    control_params = np.ones(hidden_wires_n)

    beginning = ending
    ending += hidden_wires_n
    activation_params = params[beginning: ending]

    return working_linear_params, hidden_linear_params, control_params, activation_params


def quantum_RNN_cell(x_current, params, hidden_wires, working_wires):
    working_linear_params, hidden_linear_params, control_params, activation_params = split_quantum_RNN_params(params, len(working_wires), len(hidden_wires))

    # prepare new input state
    encode_input(x_current, working_wires)
    linear_layer(working_linear_params, working_wires)
    linear_layer(hidden_linear_params, hidden_wires)

    # this is how the input and hidden states are connected
    for i in range(min(len(hidden_wires), x_current.shape[0])):
        qml.ControlledPhase(control_params[i], wires = [working_wires[i], hidden_wires[i]])

    linear_layer_inv(working_linear_params, working_wires)
    encode_input_inv(x_current, working_wires)

    activation_layer(activation_params, hidden_wires)


def quantum_RNN(x, params, hidden_wires, working_wires):
    working_wires = working_wires[: x.shape[1]]

    for x_current in x:
        quantum_RNN_cell(x_current, params, hidden_wires, working_wires)

def quantum_rnn_circuit(input, params, hidden_modes, output_size, return_state):
    rnn_params, linear_params, activiation_params = params

    quantum_RNN(input, rnn_params, np.arange(hidden_modes), np.arange(input.shape[1]) + hidden_modes)
    linear_layer(linear_params, np.arange(hidden_modes))
    activation_layer(activiation_params, np.arange(hidden_modes))

    if return_state:
        # this cannot be optimized!
        return qml.state()
    else:
        return get_x_quad_expectations(np.arange(output_size))


class QuantumRNN(tf.keras.layers.Layer):
    def __init__(self, hidden_modes, output_size, dev):
        super(QuantumRNN, self).__init__()
        self.hidden_modes = hidden_modes
        self.output_size = output_size
        self.dev = dev

        self.qnode = qml.QNode(quantum_rnn_circuit, self.dev, interface='tf')


    def build(self, input_shape):
        x_step_dim = input_shape[2]
        self.rnn_params = tf.random.normal(shape=[get_quantum_RNN_params_n(x_step_dim, self.hidden_modes)], stddev=0.001)
        self.rnn_params = tf.Variable(self.rnn_params)
        self.linear_params = tf.random.normal(shape=[get_linear_layer_params_n(self.hidden_modes)], stddev=0.001)
        self.linear_params = tf.Variable(self.linear_params)
        self.activiation_params = tf.random.normal(shape=[get_activation_layer_params_n(self.hidden_modes)], stddev=0.001)
        self.activiation_params = tf.Variable(self.activiation_params)


    def call(self, batch, return_state = False):
        # TODO do it somehow in parallel
        outs = list()
        for x in batch:
            out = self.qnode(x, [self.rnn_params, self.linear_params, self.activiation_params], self.hidden_modes, self.output_size, return_state)
            outs.append(out)

        if return_state:
            for i in range(len(outs)):
                outs[i] = trace(outs[i])
        else:
            for i in range(len(outs)):
                outs[i] = tf.concat(outs[i], axis = 0)
        return tf.stack(outs)

def quantum_dense_net_circuit(input, params, modes, output_size, layers, last_activation, return_state):
    linear_params, activiation_params, last_activation_params = params
    q = np.arange(modes)
    encode_input(input, q)
    for i in range(layers - 1):
        linear_layer(linear_params[i], q)
        activation_layer(activiation_params[i], q)
    linear_layer(linear_params[layers - 1], q)
    if last_activation:
        activation_layer(last_activation_params, q[: output_size])
    if return_state:
        # this cannot be optimized!
        return qml.state()
    else:
        return get_x_quad_expectations(q[: output_size])


class QuantumDenseNet(tf.keras.layers.Layer):
    def __init__(self, modes, output_size, layers, dev, last_activation = True):
        super(QuantumDenseNet, self).__init__()
        self.modes = modes
        self.output_size = output_size
        self.layers = layers
        self.dev = dev
        self.last_activation = last_activation

        self.qnode = qml.QNode(quantum_dense_net_circuit, self.dev, interface='tf')


    def build(self, input_shape):
        self.linear_params = tf.random.normal(shape=[self.layers, get_linear_layer_params_n(self.modes)], stddev=0.001)
        self.linear_params = tf.Variable(self.linear_params)
        self.activiation_params = tf.random.normal(shape=[self.layers - 1, get_activation_layer_params_n(self.modes)], stddev=0.001)
        self.activiation_params = tf.Variable(self.activiation_params)
        self.last_activation_params = None
        if self.last_activation:
            self.last_activation_params = tf.random.normal(shape=[get_activation_layer_params_n(self.modes)], stddev=0.001)
            self.last_activation_params = tf.Variable(self.last_activation_params)


    def call(self, batch, return_state = False):
        # TODO do it somehow in parallel
        outs = list()
        for x in batch:
            out = self.qnode(x, [self.linear_params, self.activiation_params, self.last_activation_params], self.modes, self.output_size,
                             self.layers, self.last_activation, return_state)
            outs.append(out)

        if return_state:
            for i in range(len(outs)):
                outs[i] = trace(outs[i])
        else:
            for i in range(len(outs)):
                outs[i] = tf.concat(outs[i], axis = 0)
        return tf.stack(outs)

import statistics
import matplotlib.pyplot as plt


# metric must be mean per batch
def learn(model, train_ds, test_ds, loss_fn, opt, epochs, metric_fn = None, record_steps = False, record_trace = False, skip = 1):
    train_loss_record = list()
    test_loss_record = list()
    metric_record = list()

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # do opt
        for step, (x_batch, y_batch) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                out = model(x_batch)
                loss = loss_fn(y_batch, out)

            if record_steps and step % skip == 0:
                if record_trace:
                    state = model(x_batch, return_state = True)
                    print(
                        'Training loss and mean trace (for one batch) at step {}: {:.4f}, {:.4f}'.format(
                            step, float(loss), trace(state))
                    )
                else:
                    print(
                    'Training loss (for one batch) at step {}: {:.4f},'.format(
                        step, float(loss))
                    )

                print('Seen so far: {} samples'.format((step + 1) * y_batch.shape[0]))

            gradients = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables))

        losses_tr = list()
        traces_tr = list()
        metric_te = list()
        for step, (x_batch, y_batch) in enumerate(train_ds):
            out = model(x_batch)
            loss = loss_fn(y_batch, out)
            losses_tr.append(loss.numpy())
            if record_trace:
                state = model(x_batch, return_state = True)
                traces_tr += trace.numpy().tolist()

        losses_te = list()
        traces_te = list()
        for step, (x_batch, y_batch) in enumerate(test_ds):
            out = model(x_batch)
            loss = loss_fn(y_batch, out)
            losses_te.append(loss.numpy())
            if record_trace:
                state = model(x_batch, return_state = True)
                traces_te += trace.numpy().tolist()
            if metric_fn:
                metric_te.append(metric_fn(out, y_batch))

        losses_tr = statistics.mean(losses_tr)
        losses_te = statistics.mean(losses_te)
        print('Train and test losses: {:.4f}, {:.4f}'.format(losses_tr, losses_te))
        if record_trace:
            print('Train and test mean traces: {:.4f}, {:.4f}'.format(statistics.mean(traces_tr), statistics.mean(traces_te)))
            print('Train and test min traces: {:.4f}, {:.4f}'.format(min(traces_tr), min(traces_te)))

        train_loss_record.append(losses_tr)
        test_loss_record.append(losses_te)
        if metric_fn:
            metric_te = statistics.mean(metric_te)
            print('Metric: {:.4f}'.format(metric_te))
            metric_record.append(metric_te)

    return tf.stack(train_loss_record), tf.stack(test_loss_record), tf.stack(metric_record)


def plot(te_r, metric):
    if metric.shape[0]:
        fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Metrics')
        axes[0].set_ylabel("Loss", fontsize=14)
        axes[0].plot(te_r)

        # axes[1].set_ylim([0, 1.05])
        axes[1].set_ylabel("Metric", fontsize=14)
        axes[1].set_xlabel("Epoch", fontsize=14)
        axes[1].plot(metric)
    else:
        fig, axes = plt.subplots(1, sharex=True, figsize=(12, 4))
        fig.suptitle('Training Metric')
        axes.set_ylabel("Loss", fontsize=14)
        axes.plot(te_r)

    plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

class_n = max(y) + 1
one_hot = np.zeros((y.shape[0], class_n))
one_hot[np.arange(y.size), y] = 1
y = one_hot
# Because there is no alternative of sigmoid in quantum nets
# we will make targets to be a range from -1 to 1 for symmetry
y = y * 2 - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mean = np.mean(X_train, axis = 0)
X_train -= mean
X_test -= mean
# Due to the limited size of infinite dimensional hilbert space in simulation we will put all data in a range from -1 to 1
anti_var = np.max(abs(X_train), axis = 0)
X_train /=  anti_var
X_test /=  anti_var

tf.random.set_seed(42)

bs = 4
epochs = 10

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(bs)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(bs)

loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.legacy.Adam(learning_rate = 0.05)

def acc_fn(out, y):
    pred = tf.math.argmax(out, axis = 1)
    one_hot = np.zeros((pred.shape[0], class_n))
    one_hot[np.arange(pred.shape[0]), pred] = 1
    vector = one_hot * 2 - 1
    f = tf.cast(vector == y, tf.float32)
    return float(tf.reduce_mean(f))

#Classical Model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(4,)),
    tf.keras.layers.Dense(3, activation=tf.nn.tanh),
])

tr_r, te_r, acc = learn(model, train_dataset, test_dataset, loss_fn, opt, metric_fn = acc_fn, epochs = epochs)

plot(te_r, acc)

#Quantum Model

qml.enable_tape()
cutoff_dim = 9
dev = qml.device('strawberryfields.tf', cutoff_dim = cutoff_dim, wires = 4)
qml.enable_tape()

# it seems that it is not possible to calc qml.state() for fock space right now ...
qml.enable_tape()

model = tf.keras.Sequential([
    QuantumDenseNet(4, class_n, layers = 2, dev = dev),
    tf.keras.layers.Activation('tanh')
])
tr_r, te_r, acc = learn(model, train_dataset, test_dataset, loss_fn, opt, metric_fn = acc_fn, epochs = epochs, record_trace = False)

plot(te_r, acc)