# Title

Semisupervised Variational Autoencoders: Learning the Best of Both Worlds

# Abstract

[Variational autoencoders (VAEs)]((https://www.jeremyjordan.me/variational-autoencoders/)) are a class of powerful generative models that could learn expressive, continuous latent spaces of complex objects (e.g., images) in an unsupervised manner.
In many applications, labels of datapoints that are used to train a VAE are also available; this raises the natural question of whether these labels could be incorporated into the training of the VAE to improve the quality of the learned latent space.
By simply training on these labels an additional predictor whose domain is the latent space, we can combine the VAE's reconstruction loss with supervised signals into one objective function.
The resulting model produces a latent space that is both conducive to object generation and expressive with respect to available labels, leading to better performance in learning, optimization, and user interface.
Advantages to using the Python stack will become clear as various deep learning packages allow the decomposition of the loss function described above, making implementation as seamless as possible.

# Intended audience

Machine learning (ML), especially deep learning (DL), engineers and practitioners who are interested in generative models will be able to benefit the most out of this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML/DL such as training data, loss functions, gradient descent, normal distributions, etc.

# Objectives

By the end of the talk, you will:
1. Understand the motivation behind autoencoders and variational autoencoders.
2. Know how to incorporate labels into the training of a VAE, making it semisupervised.
3. Gain practical insights on how to actually implement a semisupervised VAE.
4. See the various benefits of using such a model, including better predictive and optimization performance as well as interactivity.

# Detailed outline

Motivation (3 minutes)
- Dimensionality reduction via generative models

Basic autoencoders (5 minutes)

Variational autoencoders (5 minutes)
- to enforce Gaussianity

Semisupervised variational autoencoders (7 minutes)

Benefits of the semisupervised approach (7 minutes)

Q&A (3 minutes)

# Other notes

This talk is a cleaned-up version of a project I took part in for my graduate research, so this is a topic I'm quite familiar with and very excited to talk about!
The idea is fairly simple and natural, but it doesn't seem like it's widely known among the deep learning community, so my hope is to help popularize this framework further.

Parts of the talk will contain contents of [Martin Krasser's excellent blog post](https://github.com/krasserm/bayesian-machine-learning/blob/dev/autoencoder-applications/variational_autoencoder_opt.ipynb), which covers the same idea and some of the benefits of the model.

This is my first time submitting to PyCon US!
Some talks I have given at other venues are [this talk on Bayesian machine learning](https://discourse.pymc.io/t/bayesian-machine-learning-a-pymc-centric-introduction-by-quan-nguyen/5985) at last year's PyMCon and [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) at this year's PyData Global.
