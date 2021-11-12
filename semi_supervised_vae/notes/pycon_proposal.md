# Title

Semisupervised Variational Autoencoders: the Best of Both Worlds

# Abstract

Variational autoencoders (VAEs) are a class of powerful deep generative models that could learn expressive, continuous latent spaces of complex objects (e.g., images, protein structures) in an unsupervised manner.
In many applications, labels of datapoints that are used to train a VAE are also available; this raises the natural question of whether these labels could be incorporated into the training of the VAE to improve the quality of the learned latent space.

By simply training on these labels an additional predictor whose domain is the latent space, we can combine the VAE's reconstruction loss with supervised signals into one objective function.
The resulting model produces a latent space that is both conducive to object generation and expressive with respect to available labels, leading to better performance in learning, optimization, and user interface.
Advantages to using the Python stack will become clear in code examples shown, where the decomposition of the loss function described above is implemented as seamlessly as possible.

# Description

VAEs are deep generative models that learn from data in an unsupervised manner and have proven to be able to learn from extremely complex objects such as images and protein structures.
In most use cases however, labels are also available and are only used in downstream supervised learning tasks, after a VAE has already been trained.
Is it possible to incorporate these labels into the training of the VAE, and what are the benefits in doing so?

In this talk, we will first show that the loss function used to train a VAE could be naturally extended to take into account labels---even if they are only present for a fraction of the entire dataset---making training a _semisupervised_ procedure.
(This setting is even more appropriate in big data use cases where unlabeled data are abundant but labeling is expensive such as document annotation, protein synthesis.)
With this new training objective in hand, we will then demonstrate that the supervised signals are reflected in the learned latent space of the VAE, which leads to improved performance in various downstream tasks that uses this space, including classification/regression, Bayesian optimization, and interactivity with a human user.

# Intended audience

Machine learning (ML), especially deep learning (DL), engineers and practitioners who are interested in generative models will benefit the most out of this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML/DL such as training data, loss functions, gradient descent, normal distributions, etc.

By the end of the talk, the listener will:
1. Understand the motivation behind autoencoders and variational autoencoders as deep generative models.
2. Know how to incorporate labels into the training of a VAE, making it semisupervised.
3. Gain practical insights on how to actually implement a semisupervised VAE.
4. See the various benefits of using such a model, including better predictive and optimization performance as well as interactivity.

# Detailed outline

**Motivation (3 minutes)**
- With an autoencoder (a deep generative model), dimensionality reduction is done bottlenecking the middle portion of a neural network whose output produces reconstructed objects.
- The space produced by the middle portion, which is called the latent space, may also be used for downstream tasks such as supervised learning, blackbox optimization, and visual analytics.

**Basic autoencoders (5 minutes)**
- The topics discussed above are realized via an example using the MNIST handwritten digit dataset.
- The learned latent space shows separation between the classes, suggesting that this approach is effective at capturing the variation between the datapoints.

**Variational autoencoders (5 minutes)**
- A drawback of the latent space learned with a vanilla autoencoder is that it could be biased to be disjoint and non-continuous, which is not a desideratum of a latent space.
- Variational autoencoders address this problem by imposing a normal probability distribution over the latent vectors and reconstructing objects from samples drawn from this distribution.
- This modification comes at a low cost thanks to the so-called reparameterization trick in machine learning.
- The resulting latent space is more condensed and closer to a normal distribution.

**Semisupervised variational autoencoders (7 minutes)**
- So far, the models we have been working with are trained in an unsupervised manner.
- In many applications where a VAE is used, there are also labels

**Benefits of the semisupervised approach (7 minutes)**

Q&A (3 minutes)

# Other notes

This talk is a cleaned-up version of a project I am taking part in for my graduate research, so it is a topic I'm quite familiar with and very excited to talk about!
The idea is fairly simple and natural, but it hasn't been widely adopted by the deep learning community, so the hope/goal is to help popularize this framework further.

Some parts of the talk will overlap with [Martin Krasser's excellent blog post](https://github.com/krasserm/bayesian-machine-learning/blob/dev/autoencoder-applications/variational_autoencoder_opt.ipynb), which covers the same idea and some of the benefits of the model.

This is my first time submitting to PyCon US!
Some talks I have given at other venues are [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) at this year's PyData Global and [this talk on Bayesian machine learning](https://discourse.pymc.io/t/bayesian-machine-learning-a-pymc-centric-introduction-by-quan-nguyen/5985) at last year's PyMCon.
