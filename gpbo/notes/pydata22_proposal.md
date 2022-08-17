# Title

Bayesian Optimization: Fundamentals, Implementation, and Practice

# Abstract

How should we make smart decisions when optimizing a black-box function?
Expensive black-box optimization refers to situations where we need to maximize/minimize some input–output process, but we cannot look inside and see how the output is determined by the input.
Further, evaluating an output is expensive in terms of money, time, or other safety-critical conditions, limiting the number of times you can evaluate the function.
Black-box optimization can be found in many tasks such as hyperparameter tuning in machine learning, product recommendation, process optimization in physics, or scientific and drug discovery.

Bayesian optimization (BayesOpt) sets out to solve this black-box optimization problem by combining probabilistic machine learning (ML) and decision theory.
This technique gives us a way to intelligently design queries to the function while balancing between exploration (looking at regions without observed data) and exploitation (zeroing in on good-performance regions).
While BayesOpt has proven effective at many real-world black-box optimization tasks, many ML practitioners still shy away from it, believing that they need a highly technical background to understand and use BayesOpt.

This talk aims to dispel that message and offers a friendly introduction to BayesOpt, including its fundamentals, how to get it running in Python, and common practices to using BayesOpt.
Data and ML practitioners who are interested in hyperparameter tuning, A/B testing, or more generally experimentation and decision making will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

# Description

Optimization of expensive black-box functions is ubiquitous in machine learning and science.
It refers to the problem where we aim to optimize a function (any input–output process) $f(x)$, but we don't know the formula for $f$ and can only observe $y = f(x)$ at the location $x$ we specify.
Evaluating $y = f(x)$ may also cost a lot of time and money, constraining the number of times we can evaluate $f(x)$.
This problem of expensive black-box optimization is found in many fields such as hyperparameter tuning in machine learning, product recommendation, process optimization in physics, or scientific and drug discovery.
How can we intelligently select the locations $x$ to evaluate the function $f$ at, so that we can identify the point that maximizes the function as quickly as possible?
BayesOpt sets out to solve this question using machine learning and Bayesian probability.

This talk first covers the motivation and fundamentals behind BayesOpt in an accessible manner.
We discuss Gaussian processes (GPs), the machine learning model commonly used in BayesOpt, and decision-making policies that help us select function evaluations for the goal of optimization.
With the fundamentals covered, we then move on to how to implement BayesOpt in practice by using the state-of-the-art Python libraries, including PyTorch for array manipulation, GPyTorch for GP modeling, and BoTorch for implementing BayesOpt policies.
Finally, we cover special cases in BayesOpt that are common in the real world, such as when function evaluations may be made in batches, when evaluations have variable costs depending on the input, or when evaluations can only be made in pair-wise comparisons.

Overall, this talk explains BayesOpt in a friendly way and gets you up and running with the best BayesOpt tools in Python.

# Intended audience

Data and ML practitioners who are interested in hyperparameter tuning, A/B testing, or more generally experimentation and decision making will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

By the end of the talk, the audience will:
1. Understand the motivation behind BayesOpt as an optimization technique.
2. Know the main components of a BayesOpt procedure, including a predictive model (a GP) and a decision-making policy.
3. Gain practical insights into how to implement BayesOpt in Python.
4. See the various scenarios in which special forms of BayesOpt are found in the real world.

# Detailed outline

**Motivation (3 minutes)**

**Introducing Bayesian optimization (8 minutes)**

**Implementing Bayesian optimization in Python (8 minutes)**

**Bayesian optimization in the real world (8 minutes)**

**Q&A (3 minutes)**

# Other notes

BayesOpt is a topic I'm quite familiar with and very excited to talk about.
It's one of my graduate research topics, and the topic of [a book I'm working on](https://www.manning.com/books/bayesian-optimization-in-action).
The goal is to add to the list of available resources and make this useful machine learning technique more accessible.

This is my second time submitting to PyData Global.
Some talks I have given at other venues are [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) at last year's PyData Global and [this talk on Bayesian machine learning](https://discourse.pymc.io/t/bayesian-machine-learning-a-pymc-centric-introduction-by-quan-nguyen/5985) at PyMCon 2020.
