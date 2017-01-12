---
layout: post
title: Tensorflow：Deep MNIST for Experts
description: 
tags: [Tensorflow]
comments: true
share: true
---

---

> TensorFlow is a powerful library for doing large-scale numerical computation. One of the tasks at which it excels is implementing and training deep neural networks. In this tutorial we will learn the basic building blocks of a TensorFlow model while constructing a deep convolutional MNIST classifier.

### Load Data

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

---

take a look

|                  mnist                   |                  train                   |
| :--------------------------------------: | :--------------------------------------: |
| ![](http://ww1.sinaimg.cn/large/006y8mN6jw1f9o7bek3caj30m403odg5.jpg) | ![](http://ww1.sinaimg.cn/large/006y8mN6jw1f9o7atqd3qj30v003o74x.jpg) |

|                  images                  |                  labels                  |
| :--------------------------------------: | :--------------------------------------: |
| ![](http://ww3.sinaimg.cn/large/006y8mN6jw1f9o7bzk9phj30ny09cq4e.jpg) | ![](http://ww2.sinaimg.cn/large/006y8mN6jw1f9o7clkmlrj30jm09c3zt.jpg) |
| ![](http://ww2.sinaimg.cn/large/006y8mN6jw1f9o9c4zx83j30n20hcgn3.jpg) | ![](http://ww4.sinaimg.cn/large/006y8mN6jw1f9o9axo9cgj30pm05sdgn.jpg) |

> Here mnist is a lightweight class which stores the **training**, **validation**, and **testing** sets as NumPy arrays.

---

```python
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
```



### Build a Multilayer Convolutional Network

##### Weight Initialization

```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) 
  return tf.Variable(initial)

# tf.truncated_normal
# Outputs random values from a truncated normal distribution.
# The generated values follow a normal distribution with
# specified mean and standard deviation, except that values
# whose magnitude is more than 2 standard deviations from 
# the mean are dropped and re-picked.

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```

##### Convolution and Pooling

```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.conv2d
# Computes a 2-D convolution given 4-D `input` and `filter` tensors.
# Given an input tensor of shape `[batch, in_height, in_width, in_channels]` and a filter / kernel tensor of shape

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
# tf.nn.max_pool 
# Performs the max pooling on the input.
```

##### First Convolutional Layer

> We can now implement our first layer. It will consist of convolution, followed by max pooling. The convolutional will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of `[5, 5, 1, 32]`. The first two dimensions are the patch size, the next is the number of input channels, and the last is the number of output channels. We will also have a bias vector with a component for each output channel.

```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```

> To apply the layer, we first reshape x to a **4d tensor**, with the second and third dimensions corresponding to image width and height, and the final dimension corresponding to the number of color channels.

```python
x_image = tf.reshape(x, [-1,28,28,1])
# If one component of `shape` is the special value -1, the size of that dimension is computed so that the total size remains constant.  In particular, a `shape` of `[-1]` flattens into 1-D.  At most one component of `shape` can be -1.
```

```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# Computes rectified linear: `max(features, 0)`.
h_pool1 = max_pool_2x2(h_conv1)
```

##### Second Convolutional Layer

> In order to build a deep network, we stack several layers of this type. The second layer will have 64 features for each 5x5 patch.

```python
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```

##### Densely Connected Layer

```python
# Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image. 
# 7 = 28 / 2 / 2
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```

##### Dropout

```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```

##### Readout Layer

```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```

##### Train and Evaluate the Model

```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(1000):
  batch = mnist.train.next_batch(50)
  if i%1 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

```
...
step 992, training accuracy 1
step 993, training accuracy 0.96
step 994, training accuracy 0.98
step 995, training accuracy 0.96
step 996, training accuracy 0.96
step 997, training accuracy 0.88
step 998, training accuracy 0.92
step 999, training accuracy 0.94
test accuracy 0.9562
```

