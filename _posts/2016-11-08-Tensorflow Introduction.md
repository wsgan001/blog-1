---
layout: post
title: Tensorflow：tensorflow introduction
tags: [Tensorflow]
---

### Background

Theano and Thensoflow

> **Theano** and **TensorFlow** are very similar systems. TensorFlow has better support for distributed systems though, and has development funded by Google, while Theano is an academic project.

what does tensorfolw do?

> TensorFlow provides primitives for defining functions on tensors and automatically computing their derivatives

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

> When you train a model you use variables to hold andupdate parameters. Variables are in-memory bufferscontaining tensors

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

### Reference:  

1. [cs224d - stanford](https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf)

2. [tensorflow api-docs](https://www.tensorflow.org/versions/r0.11/api_docs/python/train.html#optimizers)

   ​