# Title

But what is a Gaussian process? Regression while knowing how certain you are

# Abstract

Given a test data point similar to the training points, we should expect the prediction of a machine learning model to be accurate.
However, we don't have the same guarantee for the prediction on the test point very far away from the training data, but many models offer no quantification of this uncertainty in our predictions.
These models, including the increasingly popular neural networks, produce a single-valued number as the prediction of a test point of interest, making it difficult to quantify how much the user should have trust in this prediction.

Gaussian processes (GPs) address this concern; a GP outputs as its prediction of a given a test point, instead of a single number, a probability distribution representing the range that the value we're predicting is likely to fall into.
By looking at the mean of this distribution, we obtain the most likely predicted value; by inspecting the variance of the distribution, we can quantify how uncertain we are about this prediction.
This ability to produce well-calibrated uncertainty quantification gives GPs an edge in high-stakes machine learning use cases such as oil drilling, drug discovery, and product recommendation.

While GPs are widely used in academic research in Bayesian inference and active learning tasks, many ML practitioners still shy away from it, believing that they need a highly technical background to understand and use GPs.
This talk aims to dispel that message and offers a friendly introduction to GPs, including its fundamentals, how to implement it in Python, and common practices.
Data scientists and ML practitioners who are interested in uncertainty quantification and probabilistic ML will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

# Description

Uncertainty quantification finds relevance in many settings in machine learning and science.
Suppose after training a model on some training set, we'd like to make predictions on two data points x1, whose features take on the same values as those in the training set, and x2, which is very different from the training points.
Due to these similarity comparisons, we can feel fairly confident that our prediction on x1 will be quite accurate–after all, we're performing interpolation.
However, it's not the same story for x2, where we're performing extrapolation, but we have no way to quantify this lack of confidence with many ML models (e.g., linear regression, decision trees, neural networks) that only produce a single number for every prediction.
Gaussian processes (GPs) address this problem by modeling the unknown target function we're learning from with a probability distribution, using Bayesian principles.

This talk first presents the motivation and fundamentals behind GPs in an accessible manner.
We discuss multivariate Gaussian distributions for finite variables as the base case and show that a GP is a generalization of this concept to infinite dimensions, modeling a function.
With the fundamentals covered, we then move on to implementing GPs in practice using the state-of-the-art Python library, GPyTorch.
Finally, we briefly iterate real world use cases that allow GPs to apply to a wide range of settings, such as scaling GPs to very large data sets, combining GPs with neural networks for more flexibility, learning from and modeling derivatives of the target function, and using GPs for adaptive sampling applications such as Bayesian optimization.

Overall, this talk explains GPs in a friendly way and gets you up and running with the best tools in Python.

# Intended audience

Data scientists and ML practitioners who are interested in uncertainty quantification and probabilistic machine learning will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

By the end of the talk, the audience will:
1. Understand the motivation behind Gaussian processes (GPs) as an infinite-dimensional multivariate distribution to model functions.
2. Know the main components of a GP, including a mean function modeling expected trend and a covariance function modeling variability.
3. Gain practical insights into how to implement GPs in Python.
4. See the various scenarios in which GPs are found in the real world.

# Detailed outline

**Motivation (3 minutes)**
- Uncertainty quantification is an important task in machine learning but isn't supported by many models.
- Gaussian processes (GPs) offer a principled way to quantify uncertainty using Bayesian probability.
- A GP models the unknown target function to be learned with a probability distribution.

**Introducing Gaussian processes (8 minutes)**
- A one-dimensional Gaussian distribution models the range of values one random variable could take with a bell curve.
A multivariate Gaussian distribution jointly models many Gaussian random variables and their covariances (or correlations).
- A GP generalizes this idea to infinitely many variables, giving us the ability to model functions.
That is, we model the value of the function f we're learning from at any input x as part of an infinite-variate Gaussian distribution and therefore can make inferences about f(x) for any x.
- We'll be using the example of modeling prices of houses at different locations to guide this conversation.

**Implementing Gaussian processes in Python (8 minutes)**
- A GP can be implemented in Python using the state-of-the-art library GPyTorch.
- GPyTorch makes implementing GPs straightforward and painless, allowing us to flexibly customize the components of a GP in a modular manner and model a wide range of behaviors.
These include what the general trend of the underlying function is, how smoothly the function varies across different input dimensions, and how much uncertainty is reduced after learning from the training data.

**Gaussian processes in the real world (8 minutes)**
- GPs' ability to quantify uncertainty in a principled way make them the go-to predictive model in high-stakes decision-making-under-uncertainty problems such as drug discovery, oil drilling, and product recommendation.
We take a look at these use cases to understand how GPs are being used in many settings.
- While lack uncertainty quantification, neural networks excel at learning from complex, structured data and allowing for a high level of modeling flexibility.
We see how we can achieve the best of both worlds by combining neural networks and GPs.
- ML practitioners might find it hard to scale GPs to very large data sets.
We discuss practical strategies available to learn a GP from big data.
- Thanks to its principled mathematical formulations, a GP can learn from and/or make predictions about the derivative of the target function.
We explore how this ability allows us to tackle unique learning problems in the sciences.

**Q&A (3 minutes)**

# Other notes

GPs are a topic I'm quite familiar with and very excited to talk about.
It's one of my graduate research topics, and the topic of the first part of [a book I wrote](https://www.manning.com/books/bayesian-optimization-in-action).
The goal is to add to the list of available resources and make this useful machine learning model more accessible.

This is my third time submitting to PyData Global.
Some talks I have given at other venues are [this talk on Bayesian optimization](https://global2022.pydata.org/cfp/talk/UTM78E/) (a topic related to GPs) at last year's PyData Global, [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) at PyData Global 2021, and [this talk on Bayesian machine learning](https://discourse.pymc.io/t/bayesian-machine-learning-a-pymc-centric-introduction-by-quan-nguyen/5985) at PyMCon 2020.
