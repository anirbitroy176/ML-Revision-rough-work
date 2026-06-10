
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
