## Gradient Descent

To minimize the cost function J,we need to estimate the parameters in the hypothesis function.

The way is to take the derivative (the tangential line to a function) of our cost function.(The fast way to reach the bottom).

We make steps down the cost function in the direction with the steepest descent.

## Formula

$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$

:= This Assignment.

The size of each step is determined by the parameter α, which is called the learning rate.

The direction in which the step is taken is determined by the partial derivative of $J(\theta_0,\theta_1)$

Depending on where one starts on the graph, one could end up at different points.(This is one feature of this algorithm).

## Be careful

At each iteration j, one should simultaneously update the parameters.

![simultaneously_update](https://github.com/chanchann/MIT-machine-learning-notes/blob/master/img/Simultaneous_update.png?raw=true)
