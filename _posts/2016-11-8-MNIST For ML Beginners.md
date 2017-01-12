---
layout: post
title: Tensorflow：MNIST For ML Beginners
tags: [Tensorflow]
comments: true
share: true
---

---

### The MNIST Data

Overview

MNIST is a simple computer vision dataset. It consists of images of handwritten digits like these:

![](https://www.tensorflow.org/versions/r0.11/images/MNIST.png)

It also includes labels for each image, telling us which digit it is. For example, the labels for the above images are 5, 0, 4, and 1.

Data load

```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

Data Process

`mnist.train.images`

Each image is 28 pixels by 28 pixels. We can interpret this as a big array of numbers,We can **flatten** this array into a vector of 28x28 = 784 numbers

The result is that mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784]. 

`mnist.train.labels`

Each image in MNIST has a corresponding label, a number between 0 and 9 representing the digit drawn in the image.

For the purposes of this tutorial, we're going to want our labels as "one-hot vectors". A **one-hot vector** is a vector which is 0 in most dimensions, and 1 in a single dimension. Consequently, mnist.train.labels is a [55000, 10] array of floats.For example, 3 would be $[0,0,0,1,0,0,0,0,0,0]$

---

### Softmax Regressions

A softmax regression has two steps: first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities.
$$
\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$

> Here softmax is serving as an "activation" or "link" function, shaping the output of our linear function into the form we want

![](https://www.tensorflow.org/versions/r0.11/images/softmax-regression-scalargraph.png)

write into equation

![](https://www.tensorflow.org/versions/r0.11/images/softmax-regression-vectorequation.png)
$$
y = \text{softmax}(Wx + b)
$$

---

### Implementing the Regression

```python
>>> import tensorflow as tf
>>> x = tf.placeholder(tf.float32, [None, 784])
# Here None means that a dimension can be of any length.
>>> W = tf.Variable(tf.zeros([784, 10]))
>>> b = tf.Variable(tf.zeros([10]))
>>> y = tf.nn.softmax(tf.matmul(x, W) + b)
```

---

### Training

> One very common, very nice function to determine the loss of a model is called "cross-entropy." Cross-entropy arises from thinking about information compressing codes in information theory but it winds up being an important idea in lots of areas, from gambling to machine learning. It's defined as:
> $$
> H_{y'}(y) = -\sum_i y'_i \log(y_i)
> $$
> Where y is our predicted probability distribution, and $y^{'}$ is the true distribution (the one-hot vector with the digit labels). In some rough sense, the cross-entropy is measuring how inefficient our predictions are for describing the truth. `[feels like inner product]`

```python
>>> y_ = tf.placeholder(tf.float32, [None, 10])
>>> cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
>>> train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
>>> init = tf.initialize_all_variables()
>>> sess = tf.Session()
>>> sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

---

### Evaluating Our Model

```python
>>> correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
>>> accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
>>> print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```

----

Reference :  [MNIST For ML Beginners](https://www.tensorflow.org/versions/r0.11/tutorials/mnist/beginners/index.html)