---
layout: post
title: Tensorflow：tensorflow introduction
description: "update at 2016-10-09"
tags: [Tensorflow]
comments: true
share: true
---

### Overview

TensorFlow is a programming system in which you represent computations as **graphs**. Nodes in the graph are called **ops** (short for operations). An op takes zero or more **Tensors**, performs some computation, and produces zero or more Tensors. A Tensor is a typed multi-dimensional array. For example, you can represent a mini-batch of images as a 4-D array of floating point numbers with dimensions [batch, height, width, channels].

A TensorFlow graph is a description of computations. To compute anything, a graph must be launched in a Session. A **Session** places the graph ops onto Devices, such as CPUs or GPUs, and provides methods to execute them. These methods return tensors produced by ops as numpy ndarray objects in **Python**, and as tensorflow::Tensor instances in C and C++.

> TensorFlow has extensive built-in support for deep learning, but is far more general than that -- **any computation that you can express as a computational flow graph, you can compute with TensorFlow** (see some examples). Any **gradient-based** machine learning algorithm will benefit from TensorFlow’s auto-differentiation and suite of first-rate optimizers. And it’s easy to express your new ideas in TensorFlow via the flexible Python interface.

> TensorFlow is graph symbolic framework whereby the optimization is done within the graph. 

##### Theano and Thensoflow

> **Theano** and **TensorFlow** are very similar systems. TensorFlow has better support for distributed systems though, and has development funded by Google, while Theano is an academic project.

---

### Installation and usage

```
Create a conda environment called tensorflow:
# Python 2.7
$ conda create -n tensorflow python=2.7
# Python 3.4
$ conda create -n tensorflow python=3.4
# Python 3.5
$ conda create -n tensorflow python=3.5


Only the CPU version of TensorFlow is available at the moment and can be installed in the conda environment for Python 2 or Python 3.
$ source activate tensorflow
(tensorflow)$  # Your prompt should change

# Linux/Mac OS X, Python 2.7/3.4/3.5, CPU only:
(tensorflow)$ conda install -c conda-forge tensorflow
```

```
$ source activate tensorflow
(tensorflow)$  # Your prompt should change.
# Run Python programs that use TensorFlow.
...
# When you are done using TensorFlow, deactivate the environment.
(tensorflow)$ source deactivate
```

---

`An easy example`

```python
import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    result = sess.run([product])
    print(result)
```

The default graph now has three nodes: two constant() ops and one matmul() op. To actually multiply the matrices, and get the result of the multiplication, you must launch the graph in a session.

---

### Tensorflow and Numpy 

```python
import tensorflow as tf
import numpy as np
```

| Numpy                                   | TensorFlow                              |
| --------------------------------------- | --------------------------------------- |
| a = np.zeros((2,2)); b = np.ones((2,2)) | a = tf.zeros((2,2)), b = tf.ones((2,2)) |
| np.sum(b, axis=1)                       | tf.reduce_sum(a,reduction_indices=[1])  |
| a.shape                                 | a.get_shape()                           |
| np.reshape(a, (1,4))                    | tf.reshape(a, (1,4))                    |
| b*5+1                                   | b*5+1                                   |
| np.dot(a,b)                             | tf.matmul(a, b)                         |
| a[0,0], a[:,0], a[0,:]                  | a[0,0], a[:,0], a[0,:]                  |

`use eval in tf`

```python
ta = tf.zeros((2,2))
print(ta.eval())	
```

---

###  TensorFlow Session Object

> A Session object **encapsulates** the environment in which Tensor objects are evaluated

```python
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
with tf.Session() as sess:
    print(sess.run(c))
    print(c.eval())
```

---

###  TensorFlow Variables

> When you train a model you use variables to hold and update parameters. Variables are in-memory buffers containing tensors

```python
W1 = tf.ones((2,2))
W2 = tf.Variable(tf.zeros((2,2)), name="weights")
with tf.Session() as sess:
    print(sess.run(W1))
    sess.run(tf.initialize_all_variables())
    print(sess.run(W2))
```

TensorFlow variables must be initialized before they have values!

#### Updating Variable State

```python
state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	print(sess.run(state))
	for _ in range(3):
    	sess.run(update)
	    print(sess.run(state))
```

```python
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.mul(input1, intermed)
```

---

### Inputting Data

`Simple solution: Import from Numpy by using tf.convert_to_tensor`

```python
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
	print(sess.run(ta))
```

---

###  Placeholders and Feed Dictionaries

> A **feed_dict** is a python dictionary mapping from tf. placeholder vars (or their names) to data (numpy arrays, lists, etc)

```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))
```

---

### Variable Scope

> Variable scope is a simple type of namespacing that adds **prefixes** to variable names within scope

|     function      |                  usage                   |
| :---------------: | :--------------------------------------: |
| tf.variable_scope | provides simple name-spacing to avoid clashes. |
|  tf.get_variable  | creates/accesses variables from within a variable scope. |

```python
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
    w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
with tf.variable_scope("foo", reuse=True)
    v1 = tf.get_variable("v")  # The same as v above.
```

---

###  Linear Regression in TensorFlow [Example]

```python
import numpy as np
import seaborn
# Define input data
X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)
# Plot input data
plt.scatter(X_data, y_data)
```

```python
#Define data size and batch size
n_samples = 1000
batch_size = 100
# Tensorflow is finicky about shapes, so resize
X_data = np.reshape(X_data, (n_samples,1))
y_data = np.reshape(y_data, (n_samples,1))
# Define placeholders for input
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))
```

```python
#Define variables to be learned
with tf.variable_scope("linear-regression"):
    W = tf.get_variable("weights", (1, 1),initializer=tf.random_normal_initializer())
	b = tf.get_variable("bias", (1,),initializer=tf.constant_initializer(0.0))
	y_pred = tf.matmul(X, W) + b
	loss = tf.reduce_sum((y - y_pred)**2/n_samples)
```

```python
#Sample code to run one step of gradient descent
opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run([opt_operation], feed_dict={X: X_data, y: y_data})
```

```python
#Sample code to run full gradient descent:
# Define optimizer operation
opt_operation = tf.train.AdamOptimizer().minimize(loss)
with tf.Session() as sess:
    # Initialize Variables in graph
    sess.run(tf.initialize_all_variables())
    # Gradient descent loop for 500 steps
    for _ in range(500):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        # Do gradient descent step
        _, loss_val = sess.run([opt_operation, loss], feed_dict={X: X_batch, y: y_batch}
```

![](http://ww4.sinaimg.cn/large/801b780ajw1f9kq3kgdn4j21kw0tddme.jpg)

---

### Interactive Usage

For ease of use in **interactive** Python environments, such as IPython you can instead use the InteractiveSession class, and the Tensor.eval() and Operation.run() methods. This avoids having to keep a variable holding the session.

```python
# Enter an interactive TensorFlow Session.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.sub(x, a)
print(sub.eval())
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()
```



### Reference:  

1. [cs224d - stanford](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf)
2. [tensorflow api-docs](https://www.tensorflow.org/versions/r0.11/get_started/basic_usage.html)
3. [google research blog](https://research.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html)
4. [mnist_softmax.py Code](https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist_softmax.py)
5. [DL4J vs. Torch vs. Theano vs. Caffe vs. TensorFlow](https://deeplearning4j.org/compare-dl4j-torch7-pylearn.html)

