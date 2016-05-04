import argparse
import tensorflow as tf
import csv
import os.path

# Parameters
learning_rate = 0.0001

# Network Parameters
n_input = 40 * 40  # data input (img shape: 40x40)
n_classes = 2  # total classes (we're using a binary classifier here)
dropout = 0.8  # dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create AlexNet model
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'), b), name=name)


def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def alex_net(_X, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 40, 40, 1])

    # First convolutional layer
    conv1 = conv2d('conv1', _X, wc1, bc1)
    pool1 = max_pool('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Second convolutional layer
    conv2 = conv2d('conv2', norm1, wc2, bc2)
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Third convolutional layer
    conv3 = conv2d('conv3', norm2, wc3, bc3)
    pool3 = max_pool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Reshape conv3 output to fit dense layer input
    dense1 = tf.reshape(norm3, [-1, wd1.get_shape().as_list()[0]])

    # Fully connected layers
    dense1 = tf.nn.relu(tf.matmul(dense1, wd1) + bd1, name='fc1')  # Relu activation
    dense2 = tf.nn.relu(tf.matmul(dense1, wd2) + bd2, name='fc2')  # Relu activation

    # Output, class prediction
    out = tf.matmul(dense2, wout) + bout
    return out

# Weights
wc1 = tf.Variable(tf.random_normal([3, 3, 1, 64]), name="wc1")
wc2 = tf.Variable(tf.random_normal([3, 3, 64, 128]), name="wc2")
wc3 = tf.Variable(tf.random_normal([3, 3, 128, 256]), name="wc3")
wd1 = tf.Variable(tf.random_normal([5 * 5 * 256, 1024]), name="wd1")
wd2 = tf.Variable(tf.random_normal([1024, 1024]), name="wd2")
wout = tf.Variable(tf.random_normal([1024, n_classes]), name="wout")

# Biases
bc1 = tf.Variable(tf.random_normal([64]), name="bc1")
bc2 = tf.Variable(tf.random_normal([128]), name="bc2")
bc3 = tf.Variable(tf.random_normal([256]), name="bc3")
bd1 = tf.Variable(tf.random_normal([1024]), name="bd1")
bd2 = tf.Variable(tf.random_normal([1024]), name="bd2")
bout = tf.Variable(tf.random_normal([n_classes]), name="bout")

# Construct model
pred = alex_net(x, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, tf.argmax(y, 1)))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Variables
var_lookup = {
    "wc1": wc1,
    "wc2": wc2,
    "wc3": wc3,
    "wd1": wd1,
    "wd2": wd2,
    "wout": wout,
    "bc1": bc1,
    "bc2": bc2,
    "bc3": bc3,
    "bd1": bd1,
    "bd2": bd2,
    "bout": bout
}

lesion_lookup = {
    0: (wc1, bc1),
    1: (wc2, bc2),
    2: (wc3, bc3),
    3: (wd1, bd1),
    4: (wd2, bd2),
    5: (wout, bout)
}

# Checkpoints and transplants
checkpoint_saver = tf.train.Saver()
transplant_saver = tf.train.Saver(var_lookup.values())


# Randomizes a TF variable
def lesion(var):
    return tf.assign(var, tf.random_normal(tf.shape(var)))

# Initializing the variables
init = tf.initialize_all_variables()


class AlexNet(object):
    def __init__(self):
        self._sess = None

    def __enter__(self):
        self._sess = tf.Session()
        self._sess.run(init)
        return self

    def record_all_biases_and_weights(self):
        assert(self._sess is not None)
        return {k: self._sess.run(v) for k, v in var_lookup.iteritems()}

    def save_full_checkpoint(self, path):
        assert(self._sess is not None)
        checkpoint_saver.save(self._sess, path)

    def load_full_checkpoint(self, path):
        assert(self._sess is not None)
        checkpoint_saver.restore(self._sess, path)

    def save_transplant(self, path):
        assert(self._sess is not None)
        transplant_saver.save(self._sess, path)

    def load_transplant(self, path):
        assert(self._sess is not None)
        transplant_saver.restore(self._sess, path)

    def train(self, training_data):
        assert(self._sess is not None)
        assert(training_data is not None)
        self._sess.run(optimizer, feed_dict={x: training_data[0], y: training_data[1], keep_prob: dropout})

    def lesion_layer(self, layer_index):
        assert(self._sess is not None)
        self._sess.run(lesion(lesion_lookup[layer_index][0]))
        self._sess.run(lesion(lesion_lookup[layer_index][1]))

    # layers = bit string of length 6
    def lesion_layers(self, layers):
        assert(self._sess is not None)
        for layer in layers:
            self.lesion_layer(layer)

    def measure_perf(self, testing_data):
        assert(self._sess is not None)

        test_accuracy = self._sess.run(accuracy, feed_dict={x: testing_data[0], y: testing_data[1], keep_prob: 1.})
        test_loss = self._sess.run(cost, feed_dict={x: testing_data[0], y: testing_data[1], keep_prob: 1.})

        return {
            'accuracy': test_accuracy,
            'loss': test_loss
        }

    def __exit__(self, type, value, traceback):
        self._sess.close()
