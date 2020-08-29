# Bayesian methods in machine learning: a brief introduction

### Short description

At the heart of any machine learning (ML) problem is the identification of models that explain the data well, where learning about the model parameters, treated as random variables, is integral. Bayes' theorem, and in general Bayesian learning, offers a principled framework to update one's beliefs about an unknown quantity; Bayesian methods therefore play an important role in many aspects of ML. This introductory talk aims to highlight some of the most prominent areas in Bayesian ML from the perspective of statisticians and analysts, drawing parallels between these areas and common problems that Bayesian statisticians work on.

### Abstract

This talk focuses on two machine learning (ML) frameworks: Bayesian modeling and Bayesian decision-making. Each of these two topics is discussed sequentially, building from toy examples to cutting-edge ML technologies; examples done in PyMC will be shown. Listeners with familiarity with Bayes' theorem and Bayesian inference will be able to follow the materials covered. No background on advanced statistics or ML is assumed.

First, we discuss Bayesian modeling, which also serves as a brief introduction for listeners who are relatively new to Bayesian statistics in general. We start with classical examples of coin flipping and Gaussian process regression, and subsequently move on to more involved techniques (e.g., mixture models, Bayesian optimization, Bayesian neural networks). Throughout this discussion, PyMC's ability to allow for flexible declaration of priors and hierarchical structures of one's models, which is often not available with other Python tools, is highlighted.

Bayesian decision theory offers a different flavor yet complements various modeling techniques described above well. An interesting starting example could be a Bayesian-optimal strategy for The Price is Right, which highlights the spirit of Bayesian decision-making under uncertainty: utility/loss functions. We then cover common (such as model selection and Bayesian optimization acquisition functions) as well as more exotic ML applications (active learning, active search).

References:
Gaussian Processes for Machine Learning: http://www.gaussianprocess.org/gpml/
Bayesian optimization using PyMC: https://pygpgo.readthedocs.io/en/latest/
The Price is Right from the perspective of Bayesian decision theory by Allen Downey: http://allendowney.blogspot.com/2013/04/the-price-is-right-problem.html
The problem of Active Search from the perspective of Bayesian decision theory: https://arxiv.org/ftp/arxiv/papers/1206/1206.6406.pdf
