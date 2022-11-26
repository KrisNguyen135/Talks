# Title

Bayesian Optimization: Fundamentals, Implementation, and Practice

# Abstract

How can we make smart decisions when optimizing a black-box function?
Expensive black-box optimization refers to situations where we need to maximize/minimize some input–output process, but we cannot look inside and see how the output is determined by the input.
Making the problem more challenging is the cost of evaluating the function in terms of money, time, or other safety-critical conditions, limiting the size of the data set we can collect.
Black-box optimization can be found in many tasks such as hyperparameter tuning in machine learning, product recommendation, process optimization in physics, or scientific and drug discovery.

Bayesian optimization (BayesOpt) sets out to solve this black-box optimization problem by combining probabilistic machine learning (ML) and decision theory.
This technique gives us a way to intelligently design queries to the function to be optimized while balancing between exploration (looking at regions without observed data) and exploitation (zeroing in on good-performance regions).
While BayesOpt has proven effective at many real-world black-box optimization tasks, many ML practitioners still shy away from it, believing that they need a highly technical background to understand and use BayesOpt.

This talk aims to dispel that message and offers a friendly introduction to BayesOpt, including its fundamentals, how to get it running in Python, and common practices.
Data scientists and ML practitioners who are interested in hyperparameter tuning, A/B testing, or more generally experimentation and decision making will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

# Description

Optimization of expensive black-box functions is ubiquitous in machine learning and science.
It refers to the problem where we aim to optimize a function (any input–output process) f(x), but we don't know the formula for f and can only observe y = f(x) at the locations x we specify.
Evaluating y = f(x) may also cost a lot of time and money, constraining the number of times we can evaluate f(x).
This problem of expensive black-box optimization is found in many fields such as hyperparameter tuning in machine learning, product recommendation, process optimization in physics, or scientific and drug discovery.
How can we intelligently select the locations x to evaluate the function f at, so that we can identify the point that maximizes the function as quickly as possible?
BayesOpt tackles this question using machine learning and Bayesian probability.

This talk first presents the motivation and fundamentals behind Bayesian optimization (BayesOpt) in an accessible manner.
We discuss Gaussian processes (GPs), the machine learning model used in BayesOpt, and decision-making policies that help us select function evaluations for the goal of optimization.
With the fundamentals covered, we then move on to implementing BayesOpt in practice using the state-of-the-art Python libraries, including GPyTorch for GP modeling and BoTorch for implementing BayesOpt policies.
Finally, we cover special cases in BayesOpt common in the real world, such as when function evaluations may be made in batches (batch optimization), when evaluations have variable costs depending on the input (cost-aware optimization), or when we need to balance between multiple objectives at the same time (multi-objective optimization).

Overall, this talk explains BayesOpt in a friendly way and gets you up and running with the best BayesOpt tools in Python.

# Intended audience

Data scientists and ML practitioners who are interested in hyperparameter tuning, A/B testing, or more generally experimentation and decision making will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

By the end of the talk, the audience will:
1. Understand the motivation behind Bayesian optimization (BayesOpt) as an optimization technique.
2. Know the main components of a BayesOpt procedure, including a predictive model (a Gaussian process) and a decision-making policy.
3. Gain practical insights into how to implement BayesOpt in Python.
4. See the various scenarios in which special forms of BayesOpt are found in the real world.

# Detailed outline

**Motivation (3 minutes)**
- Expensive, black-box optimization problems are present in many applications such as hyperparameter tuning, product recommendation, and drug discovery.
- Naïve strategies such as random search and grid search may waste valuable resources inspecting low-performance region in the search space.
- Bayesian optimization (BayesOpt) provides a method of leveraging machine learning and Bayesian decision theory to automate the search for the global optimum.

**Introducing Bayesian optimization (8 minutes)**
- BayesOpt comprises of two main components: a predictive model, commonly a Gaussian process (GP), and a decision-making algorithm called a _policy_.
The BayesOpt policy uses the GP to inform its decisions, while the GP is continually updated by the data collected by the policy, forming a virtuous cycle of optimization.
- Predictions made by a GP come in the form of multivariate normal distributions, which allow us to not only predict but also quantify our _uncertainty_ about those predictions.
This uncertainty quantification is invaluable in this problem of decision-making under uncertainty, where the cost of taking an action is high.
- A BayesOpt policy guides us towards regions in the search space that can help us find the function optimizer more quickly.
There are different BayesOpt policies, each designed with a different motivation.
We will discuss a wide range of policies ranging from improvement-based policies, policies from the multi-armed bandit problem, to policies that leverage information theory.

**Implementing Bayesian optimization in Python (8 minutes)**
- BayesOpt can be implemented in Python using a cohesive ecosystem of PyTorch for tensor manipulation, GPyTorch for implementing GPs, and BoTorch for implementing BayesOpt policies.
- GPyTorch makes implementing GPs straightforward and painless.
With GPyTorch, we can flexibly customize the components of a GP, scale GPs to large data sets, and even combine a GP with a neural network.
- BoTorch offers modular implementation of popular BayesOpt policies.
We will see that once we have defined a BayesOpt optimization loop, swapping different policies in and out is easy to do.
- Overall, the three libraries allow us to implement BayesOpt in a streamlined manner and get BayesOpt up and running in no time.

**Bayesian optimization in the real world (8 minutes)**
- The sequential nature of standard BayesOpt (choose one data point, observe its value, and repeat) is not applicable to all real-world scenarios.
We explore special variants of BayesOpt that are common in the real world.
- Batch BayesOpt is the setting where multiple function evaluations can be made at the same time.
We will discuss strategies of extending single-query BayesOpt policies to the batch setting.
- Different function evaluations could constitute different querying costs.
We will explore ways of incorporating querying costs into decision-making and develop cost-sensitive policies.
- Many real-life scenarios require optimizing more than one objective at the same time, making up multi-objective optimization problems.
We will see how BayesOpt tackles this setting by trading off the multiple objective functions at the same time.

**Q&A (3 minutes)**

# Other notes

BayesOpt is a topic I'm quite familiar with and very excited to talk about.
It's one of my graduate research topics, and the topic of [a book I'm working on](https://www.manning.com/books/bayesian-optimization-in-action).
The goal is to add to the list of available resources and make this useful machine learning technique more accessible.

This is my second time submitting to PyData Global.
Some talks I have given at other venues are [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) at last year's PyData Global and [this talk on Bayesian machine learning](https://discourse.pymc.io/t/bayesian-machine-learning-a-pymc-centric-introduction-by-quan-nguyen/5985) at PyMCon 2020.
