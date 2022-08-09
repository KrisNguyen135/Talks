# Title

Bayesian Optimization: Fundamentals, Implementation, and Practice

# Abstract

How can you make smart decisions when optimizing a black-box function?
Expensive black-box optimization refers to situations where you need to maximize/minimize some input–output process, but you cannot look inside and see how the output is determined by the input.
Further, evaluating an output is expensive in terms of money, time, or other safety-critical conditions, limiting the number of times you can evaluate the function.
Black-box optimization can be found in many tasks such as hyperparameter tuning in machine learning, product recommendation, process optimization in physics, or scientific and drug discovery.

Bayesian optimization (BayesOpt) sets out to solve this black-box optimization problem by combining probabilitic machine learning (ML) and decision theory.
This technique gives us a way to intelligently design queries to the function while balancing between exploration (looking at regions without observed data) and exploitation (zeroing in on good-performance regions).
While BayesOpt has proved effective at many real-world black-box optimization tasks, many ML practitioners still shy away from it, believing that they need a highly technical background to understand and use BayesOpt.

This talk aims to dispel that message and offers a friendly introduction to BayesOpt, including its fundamentals, how to get it running in Python, and common practices to using BayesOpt.
Data and ML practitioners who are interested in hyperparameter tuning, A/B testing, or more generally experimentation and decision making will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.
