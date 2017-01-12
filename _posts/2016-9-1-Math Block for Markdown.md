---
layout: post
title: Note：Math Block for Markdown 
tags: [Notes]
comments: true
share: true
---

Code:[Inline Math Formula]

```markdown
This formula $$f(x)=x_{1}^{2}+x_{2}^{2}$$ is an inline formula
```
preview:

This formula $$f(x)=x_{1}^{2}+x_{2}^{2}$$ is an inline formula

----
Code:[Fraction]
```markdown
$$
f(x) = x^2+\dfrac{1}{x}
$$
```
preview:

$$
f(x) = x^2+\dfrac{1}{x}
$$

----

Code:[Set]

```markdown
$$
D = \{(x_1,y_1),(x_2,y_2),…,(x_n,y_n)\}
$$
```
preview:

$$
D = \{(x_1,y_1),(x_2,y_2),…,(x_n,y_n)\}
$$
----

Code:[Sum]

```markdown
$$
　(w^{*},b^{*})= arg  \ min_{\left( w,b\right)} \sum^{n}_{i=1}\left(y_{i}-wx_{i} - b\right)^{2}
$$
```
preview:

$$
(w^{*},b^{*})= arg  \ min_{\left( w,b\right)} \sum^{n}_{i=1}\left(y_{i}-wx_{i} - b\right)^{2}
$$

----
Code:[Matrix]

```markdown
$$
X = \begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1d}&1\\
x_{21}&x_{22}&\cdots&x_{2d}&1\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
x_{n1}&x_{n2}&\cdots&x_{nd}&1\\
\end{bmatrix}
$$
```
preview:

$$
X = 
\begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1d}&1 \\
x_{21}&x_{22}&\cdots&x_{2d}&1\\
\vdots&\vdots&\ddots&\vdots&\vdots\\
x_{n1}&x_{n2}&\cdots&x_{nd}&1\\
\end{bmatrix}
$$

----
Code:[Piecewise functions]

```markdown
$$
y = 
\begin{cases}
0, &z<0\cr
0.5, &z=0\cr
1, &z>0\cr
\end{cases}
$$
```
preview:

$$
y=
\begin{cases}
0, &z<0\cr
0.5, &z=0\cr
1, &z>0\cr
\end{cases}
$$

----
Code:[Hat]

```markdown
$$
Var[\hat{f}(x)] = E[\hat{f}(x)-E[\hat{f}(x)]^2]
$$
```
preview:

$$
Var[\hat{f}(x)] = E[\hat{f}(x)-E[\hat{f}(x)]^2]
$$

----
Code:[Set Belong]

```markdown
$$
\rho = \dfrac{\sum_{x \in \hat{D}}w_x}{\sum_{x \in D}w_x}
$$
```
preview:

$$
\rho = \dfrac{\sum_{x \in \hat{D}}w_x}{\sum_{x \in D}w_x}
$$

----
Code:[Arrow]

```markdown
$$
\lim _{m\rightarrow \infty }\left( 1-\dfrac {1}{m}\right) ^{m}\rightarrow \dfrac{1}{e}
$$
```
preview:
$$
\lim _{m\rightarrow \infty }\left( 1-\dfrac {1}{m}\right) ^{m}\rightarrow \dfrac{1}{e}
$$

----
Code:[Integrate]

```markdown
$$
\int_{0}^{1}x^{2}dx
$$
```
preview:
$$
\int_{0}^{1}x^{2}dx
$$



----

Code:[Space]

```markdown
$$
f(x_i) = wx_i+b  \ \ \ \  s.t.  \  f(x_i) \approx y_i
$$
```
preview:
$$
f(x_i) = wx_i+b  \ \ \ \  s.t.  \  f(x_i) \approx y_i
$$



