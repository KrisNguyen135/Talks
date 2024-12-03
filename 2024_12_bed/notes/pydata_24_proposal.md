# Title

Cost-effective data annotation with Bayesian experimental design

## Abstract

Unlike stylized machine learning examples in textbooks and lectures, data are often not readily available to be used to train models and gain insight in real-world applications; instead, practitioners are required to collect those data themselves.
However, data annotation can be expensive (in terms of time, money, or some safety-critical conditions), thus limiting the amount of data we can possibly obtain.
(Examples include eliciting an online shopper's preference with ads at the risk of being intrusive, or conducting an expensive survey to understand the market of a given product.)
Further, not all data are created equal: some are more informative than others.
For example, a data point that is similar to one already in our training set is unlikely to give us new information; conversely, a point that is different from the data we have thus far could yield novel insight.
These considerations motivate a way for us to identify the most informative data points to label and gain knowledge in a way that makes use of our labeling budget as effectively as possible.
Bayesian experimental design (BED) formalizes this framework, leveraging the tools from Bayesian statistics and machine learning to answer the question: which data point is the most valuable that should be labeled to improve our knowledge?

This talk serves as a friendly introduction to BED including its motivation as discussed above, how it works, and how to implement it in Python.
During our discussions, we will show that interestingly, binary search, a popular algorithm in computer science, is a special case of BED.
Data scientists and ML practitioners who are interested in decision-making under uncertainty and probabilistic ML will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, and common probability distributions (normal, uniform, etc.).

## Description

The following shows the tentative outline for the talk.

**Motivation (5 minutes)**
- Data are expensive to obtain and label in the real world; examples are numerous in engineering, market research, and ads.
- Not all data are created equal: some data points are more informative than others.
- Given the data that we already have, how can we identify which data to add to that training set to best improve our knowledge?

**Introducing Bayesian experimental design (5 minutes)**
- Bayesian experimental design (BED) starts out with a probabilistic model that quantifies what we know and what we don't know.
- We then leverage decision theory to calculate the potential benefit of each new data point that we can possibly collect.
- We finally pick out the point with the highest benefit, label it, and add it to our data set.
- The entire process repeats until some termination condition is reached.

**Bayesian experimental design as binary search (5 minutes)**
- Using the tools of BED to tackle the problem of searching for an element within a sorted array, we recover binary search as a special case under a uniform prior.
- Non-uniform priors give rise to different, more complicated search strategy that reflects the advantages of BED.

**A tour of Bayesian experimental design applications (5 minutes)**
- To illustrate the generalizability and flexibility of the BED framework, we show a wide range of real-world applications: modeling infection rates in epidemiology, learning the prevalence of a disease, and collecting data for a machine learning model.
- In each scenario, we observe the interesting behavior that is induced by Bayesian decision theory.

**Bayesian experimental design in Python (5 minutes)**
- The OptBayesExpt package offers a clean, minimal implementation of BED.
- The interface allows for flexible definition of the problem, the observation model, and the data-collection strategy.
- The documentation includes many instructive examples that users can build off of.

**Q&A (5 minutes)**

By the end of the talk, the audience will:
1. Understand the motivation behind BED as a framework for decision-making under uncertainty.
2. Know the main elements of BED, including a probabilistic model and a decision-making strategy.
3. Gain practical insights into how to implement BED in Python.
4. See the various scenarios in which BED is applied in the real world.

## Other notes

Bayesian experimental design (BED) is a topic I'm quite familiar with and very excited to talk about.
It's part of the research done during my PhD, and one of the topics covered in [a book I wrote](https://www.manning.com/books/bayesian-optimization-in-action).
The goal of the talk is to make this useful technique more well-known within non-academic communities.
I'm not associated with the OptBayesExpt package but still super excited to talk about it as one of the best tools for BED in Python.

This is my forth time submitting to PyData Global.
Some talks I have given are [last year's PyData Global talk on Gaussian processes](https://global2023.pydata.org/cfp/talk/UBLDAX/), [PyData Global 2022 on Bayesian optimization](https://global2022.pydata.org/cfp/talk/UTM78E/), and [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) the year before that.
