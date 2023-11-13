# Title

Making the Perfect Cup of Joe: Active Preference Learning and Optimization Under Uncertainty

# Brief summary

_If your proposal is accepted this will be made public and printed in the program. Should be one paragraph, maximum 400 characters._

This talk discusses the problem of modeling user preferences over a feature space (learning) and efficiently identifying the highest-valued regions (optimization). We examine Gaussian Processes as a preference model, which offers calibrated uncertainty quantification, and Bayesian optimization, the state-of-the-art optimization framework for expensive, blackbox functions.

# Brief bullet point outline

_Brief outline. Will be made public if your proposal is accepted. Edit using Markdown._

**Introduction to Gaussian Processes and Bayesian optimization**

- Optimizing expensive-to-query, blackbox functions is a common problem in science, engineering, and development of new products.
- In many cases, the cost of evaluating the function's values restricts the size of the training dataset, making machine learning models that require a large amount of data unsuitable.
- [Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/) can naturally model small datasets and offer calibrated quantification of uncertainty.
- [Bayesian optimization](http://krasserm.github.io/2018/03/21/bayesian-optimization/) is a framework of deciding at which location the function in question should be evaluated next, so that its global optimum may be found as efficiently as possible.

**Preference learning**

- Real-world A/B testing applications may be modeled as preference learning problems, where we query a user's valuation of a product with specific settings and aim to identify the setting that gives the maximal valuation.
- Research has shown that [humans are better at giving preference-based responses (pairwise comparisons) than rating products on a scale](http://hassler-j.iies.su.se/COURSES/NewPrefs/Papers/KahnemanTversky%20Ec%2079.pdf).
- These pairwise comparisons are in the form of, "the product at setting A is preferred to the product at setting B."
- We will see how a Gaussian Process can also be trained on this type of preference-based data, using a special learning algorithm.

**Preference optimization**

- With the specialized Gaussian Process in hand, we then appeal to Bayesian optimization as a sample-efficient optimization framework to identify the product setting that maximizes the user's preference in as few queries as possible.
- We will see how to translate common optimization strategies (e.g., _expected improvement_) to this problem and examine their performance on test benchmarks.

**Others**

- We will discuss two Python packages that implement this preference optimization framework---[GPro](https://github.com/chariff/GPro) and [BoTorch](https://github.com/pytorch/botorch)---which meet different implementation needs.
- To make the problem less abstract, we will center our discussion around the running toy example of optimizing the amount of sugar to put in a cup of coffee.

# Description

_Detailed outline. Will be made public if your proposal is accepted. Edit using Markdown._

**Abstract**

Optimizing a user's valuation is common in science, engineering, and development of new products, and is typically framed as a blackbox optimization problem.
[Bayesian optimization](http://krasserm.github.io/2018/03/21/bayesian-optimization/) (BO) is an attractive solution, as it offers a principled way of intelligently designing queries to the user while balancing between exploration and exploitation, with the goal of quickly identifying the user's highest valuation.
However, this framework assumes that observations come in the form of real-valued evaluations/ratings.
In many practical scenarios, we only have access to _pairwise comparisons_, expressing the user's preference between two given products.

_Preference learning_ (PL) refers to the task of learning the user's valuation via these pairwise comparisons.
This talk presents how BO may be extended to PL, including using a [Gaussian Process](https://distill.pub/2019/visual-exploration-gaussian-processes/) (GP) for modeling and designing optimization strategies.
We will see how to build an optimization loop of this kind and gain practical implementation insights, such as which strategy is effective in low- vs. high-dimensional spaces.

Machine learning (ML) practitioners who work on human-in-the-loop, A/B testing, product recommendation, and user preference optimization problems will benefit from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML such as training data, predictive models, multivariate normal distributions, etc.

**Description**

The following is the tentative plan for the talk.

---

## 0. Motivating the Problem (minute 1--3)

We set up the main problem that this talk addresses: optimizing a user's preference, which is treated as an expensive-to-query, blackbox function.
The workflow consists of multiple iterations where we repeatedly query and observe the function's value at a location, train a model on observed data, and use that model to inform our next query.
The goal is to identify the global optimum of the function as quickly as possible.

Real-world applications include machine learning hyperparameter tuning, optimizing a physical process, and finding the product that a user is most likely to be interested in.
A toy example we will be using throughout the talk is finding the best amount of sugar to use when making coffee.

## 1. Gaussian Processes and Bayesian Optimization (minute 4--10)

We will discuss the background for BO as a framework of optimizing a blackbox function.
This will be quite brief as it has been the topic of many talks in the past and not the focus of this talk.

### 1.1. Gaussian Processes (minute 4--6)

A GP defines a distribution over functions such that the function values at any given array of locations follow a multivariate Normal distribution, which is still the case even when a GP is conditioned on some observed data.
GPs are almost exclusively used as the predictive model in BO for its ability to quantify uncertainty about the function value in regions where we don't have a lot of observations.

We will include plots showing how our belief about a function represented by a GP changes as new datapoints are observed, emphasizing on the level of uncertainty in that belief.

### 1.2. Bayesian Optimization (minute 7--9)

What is the most informative location to query the function value at, so that we could identify the global optimum quickly?
A BO _policy_ answers this question by taking into account our belief and uncertainty about the function (represented by the GP).
We focus on the _expected improvement_ (EI) policy, arguably the most popular in practice.
This policy computes the expected value of the increase in the current best-seen point for each potential query location, which is straightforward to do given the multivariate normal belief about the function values.

This policy naturally balances exploration and exploitation in its scoring, favoring locations with either high function values or high uncertainty.
We will include plots to demonstrate this point.
Making _a batch of queries_ using this EI score is also possible.

### 1.3. Stopping for Questions (minute 10)

## 2. Preference Learning (minute 11--21)

We now discuss our PL framework, where observations come in the form of pairwise comparisons, which is the main focus of this talk.

### 2.1. Learning (minute 11--15)

The GP conditioned on pairwise computation observations cannot be derived in closed form, but may be approximated using [Laplace approximation](https://bookdown.org/rdpeng/advstatcomp/laplace-approximation.html).
We won't go into detail about how this algorithm works, but will instead look at plots showing the output of this algorithm for different datasets.

These plots will show that the learned GP successfully reflects the information contained in the pairwise comparisons, while still offering reasonable measures of uncertainty.

### 2.2. Optimization (minute 16--20)

Our task is to choose a _pair of locations_ whose output comparison from the user will help us identify the global optimum the fastest.
We will consider two extensions of EI to this setting: (1) choose _a batch of two queries_ using the EI score and (2) compare the single EI candidate against the current best observation.
We will see the difference in querying behavior of these two policies in different scenarios.

We will run these two policies on a suite of benchmark functions and examine their performance.
We will observe that policy (1) performs better on one-dimensional functions, while policy (2) is better on two- and three-dimensional ones.
This insight will help practitioners choose the appropriate policy in their own use cases.

### 2.3. Stopping for Questions (minute 21)

## 3. Practical Considerations (minute 22--26)

We will end the talk with some high-level discussions.

### 3.1. Implementation (minute 22--23)

We will first talk about two Python packages that implement this preference-based optimization loop: [GPro](https://github.com/chariff/GPro) and [BoTorch](https://github.com/pytorch/botorch).
The former is built on top of Scikit-learn and more lightweight, while the latter belongs to the PyTorch + GPyTorch + BoTorch ecosystem and comes with extensive support.

Overall, this discussion will help practitioners navigate the space of available software.

### 3.2. When **Not** to Do This (minute 24--26)

Not all preference optimization applications are appropriate for this workflow.
We will end this talk by outlining when we should not (or don't have to) employ this specialized optimization routine: such as when real-valued evaluations/ratings are more natural; when evaluation is cheap; or when there are only a small, discrete set of choices.

## 4. Q&A (minute 27--30)

"What questions do you have?"
