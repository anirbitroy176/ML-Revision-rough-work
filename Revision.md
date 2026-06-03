# Linear Regression from First Principles

## 1. What Problem Does Linear Regression Solve?

Suppose we are given a dataset consisting of observations:

$$
(x_1,y_1), (x_2,y_2), \ldots, (x_n,y_n)
$$



The goal is to find a straight line that best represents the relationship between the input variable (x) and the target variable (y).

Since the data points do not usually lie perfectly on a straight line, some prediction error is unavoidable. Linear Regression aims to find the line that minimizes these prediction errors.

The prediction errors are called **residuals**.

---

## 2. Equation of a Line

A straight line can be represented as:

$$
\hat y = mx + b
$$

where:

* $m$ = slope of the line
* $b$ = intercept
* $x$ = input feature
* $\hat y$ = predicted value

For a given input (x), the model predicts:

$$
\hat y = mx + b
$$

The actual observed value is $y$. 

The difference between the actual and predicted value is called the **prediction error** or **residual**:

$$
e = y - \hat y
$$

---

## 3. Residuals

For the $i^{th}$ data point:

$$
e_i = y_i - \hat y_i
$$

Substituting the prediction equation:

$$
e_i = y_i - (mx_i + b)
$$

A residual measures how far the model's prediction is from the actual value.

* Small residual → good prediction
* Large residual → poor prediction

A good regression line is one where the residuals are collectively small across all observations.

However, because a dataset contains many residuals, we need a single number that summarizes the overall error of the model.

This leads to the **Mean Squared Error (MSE)**.

---

## 4. Mean Squared Error (MSE)

Since there are many residuals, MSE aggregates all of them into a single measure of model error.

The Mean Squared Error is defined as:

$$
J(m,b)=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
$$

Substituting:

$$
\hat y_i = mx_i+b
$$

gives:

$$
J(m,b)=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-(mx_i+b)\right)^2
$$

where:

* $n$ = number of data points
* $J(m,b)$  = cost function (loss function)

The objective of Linear Regression is to find the values of $m$) and $b$ that minimize this cost function.

---
MSE is used because :
- The squared error is continuous and differentiable with respect to parameters m and b,
   allowing the use of calculus based methods like partial derivatives, to find the optimal solution.
-  In Linear regression, MSE loss is convex with respect to the parameters. This means
   local minimum is also the global minimum, which means optimization algorithms like
   gradient descent can move downwards without getting stuck

## Why Do We Square the Residuals?

If we simply added residuals together,

$$
\sum e_i
$$

positive and negative errors could cancel each other out.

Example:

$$
+10 + (-10) = 0
$$

even though both predictions are poor.

Squaring solves this problem:

$$
(y_i-\hat y_i)^2
$$

because:

1. All errors become positive.
2. Larger errors receive a heavier penalty.
3. The resulting function becomes smooth and mathematically convenient for optimization.

Thus, Linear Regression minimizes the **sum of squared residuals** rather than the raw residuals themselves.

---
## Closed-form solution idea and intuition

$$
J(m,b)=\frac{1}{n}\sum_{i=1}^{n}\left(y_i-(mx_i+b)\right)^2
$$

$$
\begin{aligned}
\frac{\partial J}{\partial m}
&=
\frac{\partial}{\partial m}
\left[
\frac{1}{n}\sum_{i=1}^{n}
\left(y_i - (mx_i + b)\right)^2
\right]  
&=
\frac{1}{n}\sum_{i=1}^{n}
\frac{\partial}{\partial m}
\left(y_i - (mx_i + b)\right)^2  
&=
\frac{1}{n}\sum_{i=1}^{n}
2\left(y_i - (mx_i + b)\right)
\frac{\partial}{\partial m}
\left(y_i - (mx_i + b)\right)  
&=
\frac{1}{n}\sum_{i=1}^{n}
2\left(y_i - (mx_i + b)\right)(-x_i)  
&=
-\frac{2}{n}\sum_{i=1}^{n}
x_i\left(y_i - (mx_i + b)\right)
\end{aligned}
$$


$$
\begin{aligned}
\frac{\partial J}{\partial b}
&=
\frac{\partial}{\partial b}
\left[
\frac{1}{n}\sum_{i=1}^{n}
\left(y_i - (mx_i + b)\right)^2
\right] \\[6pt]
&=
\frac{1}{n}\sum_{i=1}^{n}
\frac{\partial}{\partial b}
\left(y_i - (mx_i + b)\right)^2 
&=
\frac{1}{n}\sum_{i=1}^{n}
2\left(y_i - (mx_i + b)\right)
\frac{\partial}{\partial b}
\left(y_i - (mx_i + b)\right)  
&=
\frac{1}{n}\sum_{i=1}^{n}
2\left(y_i - (mx_i + b)\right)(-1) 
&=
-\frac{2}{n}\sum_{i=1}^{n}
\left(y_i - (mx_i + b)\right)
\end{aligned}
$$


$$
\begin{aligned}
-\frac{2}{n}\sum_{i=1}^{n}\left(y_i - (mx_i + b)\right) &= 0 \\
\sum_{i=1}^{n}\left(y_i - mx_i - b\right) &= 0 \\
\sum_{i=1}^{n}y_i - m\sum_{i=1}^{n}x_i - \sum_{i=1}^{n}b &= 0 \\
\sum_{i=1}^{n}y_i - m\sum_{i=1}^{n}x_i - nb &= 0 \\
nb &= \sum_{i=1}^{n}y_i - m\sum_{i=1}^{n}x_i \\
b &= \frac{1}{n}\sum_{i=1}^{n}y_i - \frac{m}{n}\sum_{i=1}^{n}x_i
\end{aligned}
$$

$$
b = \hat{y}-m\hat{x}
$$

$$
\begin{aligned}
-\frac{2}{n}\sum_{i=1}^{n} x_i\left(y_i - (mx_i + b)\right) &= 0 
\sum_{i=1}^{n} x_i\left(y_i - (mx_i + b)\right) &= 0  
\sum_{i=1}^{n} x_i y_i - \sum_{i=1}^{n} x_i(mx_i + b) &= 0  
\sum_{i=1}^{n} x_i y_i - m\sum_{i=1}^{n}x_i^2 - b\sum_{i=1}^{n}x_i &= 0 
m\sum_{i=1}^{n}x_i^2 &= \sum_{i=1}^{n} x_i y_i - b\sum_{i=1}^{n}x_i  
m &= \frac{\sum_{i=1}^{n} x_i y_i - b\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}x_i^2}
\end{aligned}
$$

$$
\begin{aligned}
m &= \frac{\sum_{i=1}^{n} x_i y_i - (\hat{y} - m\hat{x})\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}x_i^2} \\[4pt]
m &= \frac{\sum_{i=1}^{n} x_i y_i - \hat{y}\sum_{i=1}^{n}x_i + m\hat{x}\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}x_i^2} \\[4pt]
m\sum_{i=1}^{n}x_i^2 &= \sum_{i=1}^{n} x_i y_i - \hat{y}\sum_{i=1}^{n}x_i + m\hat{x}\sum_{i=1}^{n}x_i \\[4pt]
m\sum_{i=1}^{n}x_i^2 - m\hat{x}\sum_{i=1}^{n}x_i &= \sum_{i=1}^{n} x_i y_i - \hat{y}\sum_{i=1}^{n}x_i \\[4pt]
m\left(\sum_{i=1}^{n}x_i^2 - \hat{x}\sum_{i=1}^{n}x_i\right) &= \sum_{i=1}^{n} x_i y_i - \hat{y}\sum_{i=1}^{n}x_i \\[4pt]
m &= \frac{\sum_{i=1}^{n} x_i y_i - \hat{y}\sum_{i=1}^{n}x_i}{\sum_{i=1}^{n}x_i^2 - \hat{x}\sum_{i=1}^{n}x_i}
\end{aligned}
$$
