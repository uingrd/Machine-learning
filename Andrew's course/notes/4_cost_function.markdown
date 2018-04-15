## Cost function

![cost_function](https://github.com/chanchann/MIT-machine-learning-notes/blob/master/img/cost_function_1.png?raw=true)

<p id = "build"></p>
---

This function is otherwise called the "Squared error function", or "Mean squared error".

It works well for problems for most regression programs.

To figure out how to fit the best possible straight line to our data.

We can measure the accuracy of our hypothesis function by using a cost function.



## Goal:

How to go about choosing these two parameter values, $\theta_0 and \theta_1$?

## The understanding of $\frac{1}{2}$:

The mean is halved $\left(\frac{1}{2}\right)$ as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the $\frac{1}{2}$ term.
