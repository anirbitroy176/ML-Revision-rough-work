
# GD Revision
## What problem does Gradient Descent solve?
Instead of calculating the best parameters in one step, it starts for initial values m and b, then updates
them repeatedly to reduce the loss

So the main difference is:
- Simple LR solves in one step
- Gradient descent learns the parameter step by step through repeated updates

This makes Gradient Descent important because many machine learning models do not have simple closed-form solutions, 
or direct computation may become expensive for large datasets.

## Why do we need Gradient Descent if closed-form solution exists?
In simple linear regression, closed-form solution can directly calculate the best parameters. But Gradient Descent is still needed because </br>
many real-world machine learning models do not have a simple direct formula. </br>
Also, for very large datasets or many features, closed-form computation can become expensive. </br>
Gradient Descent is more general because it learns the parameters step by step by reducing the cost function.

## What is a cost function?
In simple linear regression, $y_i$ is the actual value and $\hat{y}_i$
	​

 is the predicted value. A cost function measures how wrong the model is by calculating the average of the squared residuals.

$$
\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Now,

$$
\hat{y}=mx_i+b
$$

So,

$$
J(m,b)=\frac{1}{n}\sum_{i=1}^{n}(y-(mx_i+b))^2
$$

We adjust m and b to minimize this cost and find the best-fit line.

## What does MSE Measure?

MSE, or Mean Squared Error, measures the average of the squared errors between actual values and predicted values. The lower the MSE,</br>
the better the model’s predictions are.

$$
MSE= \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
$$

Squaring is used because:

- Positive and negative errors should not cancel each other out.
- Larger errors are penalized more heavily.
- Squared error is differentiable, which makes it suitable for gradient descent.

## What does the gradient mean?
Gradient means a mathematical vector that points in the direction of steepest increase of a function.</br>
For a function with multiple parameters, the gradient contains partial derivatives with respect to each parameters.</br>
For the linear regression loss function:</br>

$$
J(m,b)
$$

the gradient is:


$$
\nabla J(m,b) =
\left[
\frac{\partial J}{\partial m},
\frac{\partial J}{\partial b}
\right]
$$

This slope vector tells us how the loss changes 
- when slope (m) changes
- when intercept (b) changes
- the direction in which the loss changes most rapidly

Since Gradient Descent aims to minimize the loss, it moves in the opposite direction of the gradient.

Therefore:

- Gradient → direction of steepest increase
- Negative Gradient → direction of steepest decrease

This idea forms the foundation of the Gradient Descent optimization algorithm.

## Why do we move opposite to the gradient?
We move opposite to the gradient because gradient points in the direction where the loss increases the </br>
fastest. Since our goal is to minimise the loss, we move in the opposite direction




