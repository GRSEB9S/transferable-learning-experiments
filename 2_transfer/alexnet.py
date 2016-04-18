import input_data
import tensorflow as tf
import csv
import os.path
import datetime


# Globals
RESTORING_OLD_CHECKPOINT = False
TRANSFERRING = True
GENERATING_FOR_TRANSFER = False

# IO Things
POS_CLASS = 'dog'
positive_dir = "./data/" + POS_CLASS
negative_dir = "./data/not" + POS_CLASS

# Parameters
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# Network Parameters
n_input = 40 * 40  # data input (img shape: 28*28)
n_classes = 2  # total classes (0-9 digits)
dropout = 0.8  # Dropout, probability to keep units

# Training data # TODO Partition test / training
data = input_data.read_data_sets(positive_dir, negative_dir)

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

    # Third convolutional layer
    conv1 = conv2d('conv1', _X, wc1, bc1)
    pool1 = max_pool('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, _dropout)

    # Third convolutional layer
    conv2 = conv2d('conv2', norm1, wc2, bc2)
    pool2 = max_pool('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    # Third convolutional layer
    conv3 = conv2d('conv3', norm2, wc3, bc3)
    pool3 = max_pool('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)

    # Fully connected layers
    # Reshape conv3 output to fit dense layer input
    dense1 = tf.reshape(norm3, [-1, wd1.get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, wd1) + bd1,
                        name='fc1')  # Relu activation
    dense2 = tf.nn.relu(tf.matmul(dense1, wd2) + bd2,
                        name='fc2')  # Relu activation

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
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Checkpoints and transplants
checkpoint_saver = tf.train.Saver()
transfer_saver = tf.train.Saver({
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
})

# Initialize CSV output file
fname = "performance " + str(datetime.datetime.utcnow()) + ".csv"
fpath = os.path.join(os.path.abspath('results'), fname)

with open(fpath, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["iteration", "accuracy", "loss"])

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    if RESTORING_OLD_CHECKPOINT:
        checkpoint_saver.restore(sess, "/tmp/model.ckpt")

    if TRANSFERRING:
        transfer_saver.restore(sess, "/tmp/transfer.ckpt")

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = data.train.next_batch(batch_size)

        # Fit training using batch data
        sess.run(optimizer, feed_dict={
                 x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:

            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={
                           x: batch_xs, y: batch_ys, keep_prob: 1.})

            # Calculate batch loss
            loss = sess.run(cost, feed_dict={
                            x: batch_xs, y: batch_ys, keep_prob: 1.})

            # Output benchmarks to csv
            with open(fpath, 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    [step * batch_size, "{:.5f}".format(acc), "{:.6f}".format(loss)])

            # Print to console
            print "Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)

            # Checkpointing
            checkpoint_saver.save(sess, "/tmp/model.ckpt")

            if GENERATING_FOR_TRANSFER:
                transfer_saver.save(sess, "/tmp/transfer.ckpt")

        step += 1

    print "Optimization Finished!"

    # Calculate accuracy for 256 data test images
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: data.test.images[:256], y: data.test.labels[:256], keep_prob: 1.})
