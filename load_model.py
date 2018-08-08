import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import read_data
import numpy as np
import time

filepath = '/DATA/data/hyguan/liuyuan_spine/data_all/patient_image_4'
train_filename = ''
# mnist = load_all_sets(filepath)

# Hyperparameter
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2
# dropout_rate = 0.4

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 2
batch_size = 32

total_epochs = 100


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            shape = x.get_shape().as_list()
            in_channel = shape[3]
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)


        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))


        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x


x = tf.placeholder(tf.float32, shape=[None, 50176])
batch_images = tf.reshape(x, [-1, 224, 224, 1])

label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

# """
# l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
# train = optimizer.minimize(cost + l2_loss * weight_decay)
# In paper, use MomentumOptimizer
# init_learning_rate = 0.1
# but, I'll use AdamOptimizer
# """

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)

# """
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
# """
log_loc = tf.argmax(logits, 1)
lab_loc = tf.argmax(label, 1)
correct_prediction = tf.equal(log_loc, lab_loc)
# error label
err = np.array([0, 0, 0, 0])
for i in range(300):
    if correct_prediction[i] == False:
        err[lab_loc[i]] += 1 
err_label = tf.constant(err)
# """
# correct_prediction_two = tf.equal(tf.argmax(logits[:, 4:7], 1), tf.argmax(label[:, 4:7], 1))
# correct_prediction_thr = tf.equal(tf.argmax(logits[:, 7:11], 1), tf.argmax(label[:, 7:11], 1))
# correct_prediction_fou = tf.equal(tf.argmax(logits[:, 11:14], 1), tf.argmax(label[:, 11:14], 1))
# """
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

# """
# # delete?
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# """
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/DATA/data/qyzheng/Tensorflow/model')
    saver.restore(sess, ckpt.model_checkpoint_path)

    train_size, test_size = input_data.get_size(filepath)
    print(train_size, ' ', test_size)
    total_test_batch = int(test_size / batch_size)

    accuracy_rates = 0
    # err_labels = np.array([0, 0, 0, 0])
    writer = tf.python_io.TFRecordWriter(filepath + '/next.tfrecords')
    for step in range(total_test_batch):
        test_images, test_labels, test_labels_mul = read_data.next_batch(train_filename, batch_size)
        test_feed_dict = {
            x: test_images,
            label: test_labels,
            learning_rate: init_learning_rate,
            training_flag : False
        }

        if step % 100 == 0:
            print('step', step)

        predict, accuracy_rate = sess.run([log_loc, accuracy], feed_dict=test_feed_dict)
        accuracy_rates += accuracy_rate

        for i in range(len(predict)):
            if predict[i] == 1:
                image_bytes = test_images.tobytes()
                label_bytes = test_labels.tobytes()
                labels_mul_bytes = test_labels_mul.tobytes()

                example = tf.train.Example(features=tf.train.Features(feature={
                    "label_mul": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                }))

                writer.write(example.SerializeToString())
    writer.close()
        # [lab, correct, accuracy_rate] = sess.run([lab_loc, correct_prediction, accuracy], feed_dict=test_feed_dict)
        # accuracy_rates += accuracy_rate
        # for i in range(correct.shape[0]):
        #     if correct[i] == False:
        #         err_labels[lab[i]] += 1

    accuracy_rates /= total_test_batch
    print('Test Accuracy =', accuracy_rates)
    # print('Test error label = ', err_labels)