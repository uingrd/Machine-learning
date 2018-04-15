## Linear regression

This model is called linear regression,another name for this model is univariate linear regression.



<p id = "build"></p>
---

## Training set

Learn from this data how to predict.


## Symbols

m:  the number of training examples.

x:  to denote the input variables often also called the features.

X:  to denote the space of input values

y:  to denote my output variables or the target variable which I'm going to predict.

Y:  to denote the space of output values

(x, y): to denote a single training example.

$$(x^{(i)} , y^{(i)} )$$ i=1,...,m:  just refers to the ith row of this table.superscript “(i)” in the notation is simply an index into the training set

h:  stands for hypothesis which is a function that maps from x's to y's.

## Thinking:

How do we represent this hypothesis h?

Our goal is, given a training set, to learn a function h : X → Y so that h(x) is a “good” predictor for the corresponding value of y

## The process:

                     Traning set
                         |
                         |
                  Learning Algorithm
                         |
                         |
                     x-->h-->predicted y
