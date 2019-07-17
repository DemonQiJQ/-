# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'mnist/data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.


FLAGS = None


def data_type():
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32


def maybe_download(filename):#下载数据集（指定目录下不存在数据集）
  if not tf.gfile.Exists(WORK_DIRECTORY):     #目录不存在则创建目录
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):           #文件不存在
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath


def extract_data(filename, num_images):       #提取数据（将图像转为4D张量）
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH         #将图像像素值从[0,255]转为[-0.5,0.5]
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)     #数据reshape为(num_images,)
    return data


def extract_labels(filename, num_images):     #提取图像标签
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):            #产生虚拟数据
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):      #计算错误率（百分制）
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /       #预测正确总数除以样本总数
      predictions.shape[0])


def main(_):
  if FLAGS.self_test:         #执行测试
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')      #训练数据
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')    #训练标签
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')        #测试数据
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')      #测试标签

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)                   #提取训练数据
    train_labels = extract_labels(train_labels_filename, 60000)             #提取训练标签
    test_data = extract_data(test_data_filename, 10000)                     #提取测试数据
    test_labels = extract_labels(test_labels_filename, 10000)               #提取测试标签

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, ...]                     #从训练数据中划分验证集
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS                                             
  train_size = train_labels.shape[0]                                        #训练数据大小


  train_data_node = tf.placeholder(                                         #训练数据节点
      data_type(),
      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))         #训练标签节点
  eval_data = tf.placeholder(                                               #验证数据节点
      data_type(),
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))



  conv1_weights = tf.Variable(                                              #卷积层1权重w     5x5x1x32
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],                         
                          stddev=0.1,
                          seed=SEED, dtype=data_type()))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))             #卷积层1偏置b
  conv2_weights = tf.Variable(tf.truncated_normal(                          #卷积层2权重w     5x5x32x64
      [5, 5, 32, 64], stddev=0.1,                                           
      seed=SEED, dtype=data_type()))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type())) #卷积层2偏置b
  fc1_weights = tf.Variable(                                                  #全连接层1权重w
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.1,
                          seed=SEED,
                          dtype=data_type()))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))  #全连接层1偏置b
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],            #全连接层2权重w
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=data_type()))
  fc2_biases = tf.Variable(tf.constant(                                       #全连接层2偏置b
      0.1, shape=[NUM_LABELS], dtype=data_type()))

  # We will replicate the model structure for the training subgraph, as well
  # as the evaluation subgraphs, while sharing the trainable parameters.
  def model(data, train=False):
    """The Model definition."""
    #卷积-relu激活-池化-卷积-relu激活-池化-
    conv = tf.nn.conv2d(data,                   #定义卷积层
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))           #激活函数
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,                                     #池化层
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,                                       #卷积层2
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))           #激活
    pool = tf.nn.max_pool(relu,                                     #池化层
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    
    pool_shape = pool.get_shape().as_list()                       
    reshape = tf.reshape(                                           #将得到的张量拉伸为1D，方便后续全连接层计算
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])     
    
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)   #全连接：relu(wx+b)
    

    if train:                                                         #训练时增加dropout，避免过拟合
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases                #全连接层2：wx+b

  

  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(     #计算损失
      labels=train_labels_node, logits=logits))

  
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +      #添加正则化项，避免过拟和
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  
  loss += 5e-4 * regularizers

  

  batch = tf.Variable(0, dtype=data_type())                     #第几个batch                      
  
  learning_rate = tf.train.exponential_decay(
      0.01,                #基础学习率
      batch * BATCH_SIZE,  #当前batch序号
      train_size,          #训练集合大小
      0.95,                #学习率下降速率
      staircase=True)
  
  optimizer = tf.train.MomentumOptimizer(learning_rate,             #优化器
                                         0.9).minimize(loss,
                                                       global_step=batch)

  
  train_prediction = tf.nn.softmax(logits)                          #softmax计算预测类别

  
  eval_prediction = tf.nn.softmax(model(eval_data))                 



  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    
    tf.global_variables_initializer().run()         #训练参数初始化
    print('Initialized!')
    
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      

      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)            #计算训练batch偏移量
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]          #生成batch数据
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]           #生成batch标签
      
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      
      sess.run(optimizer, feed_dict=feed_dict)                            #执行优化器更新参数
      
      if step % EVAL_FREQUENCY == 0:
        
        l, lr, predictions = sess.run([loss, learning_rate, train_prediction],      #计算训练损失，学习率及得到的预测
                                      feed_dict=feed_dict)
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true')
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
