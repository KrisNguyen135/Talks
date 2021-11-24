# Title

Semisupervised Variational Autoencoders: Learning the Best of Both Worlds

# Abstract

Variational autoencoders (VAEs) are a class of powerful deep generative models that could learn expressive, continuous latent spaces of complex objects (e.g., images, protein structures) in an unsupervised manner.
In many applications, labels of datapoints that are used to train a VAE are also available; this raises the natural question of whether these labels could be incorporated into the training of the VAE to improve the quality of the learned latent space.

By simply training on these labels an additional predictor whose domain is the latent space, we can combine the VAE's reconstruction loss with supervised signals in one objective function.
The resulting model produces a latent space that is both conducive to object generation and expressive with respect to available labels, leading to better performance in learning, optimization, and user interface.

Advantages to using the Python stack will become clear in examples throughout the talk, where the loss function described above is implemented as seamlessly as possible.

# Description

VAEs are deep generative models that have proven to be able to learn from extremely complex objects such as images and protein structures in an unsupervised manner.
In most use cases, labels are also available but are only used in downstream supervised learning tasks, after a VAE has already been trained.
Is it possible to incorporate these labels into the training of the VAE, and what are the benefits in doing so?

We first show that the loss function used to train a VAE could be naturally extended to take into account labels---even if they are only present in a fraction of the entire dataset---making training a _semisupervised_ procedure.
(This setting is even more appropriate in "big data" use cases where unlabeled data are abundant but labeling is expensive, such as document annotation or protein synthesis.)
With this new training objective in hand, we then demonstrate that the new model effectively learns from the labels, leading to improved performance in various downstream tasks such as prediction, blackbox optimization, and interactivity with a human user.

# Intended audience

Machine learning (ML), especially deep learning (DL), practitioners who are interested in generative models will benefit the most from this talk.
While most background knowledge necessary to follow the talk will be covered, the audience should be familiar with common concepts in ML/DL such as training data, loss functions, gradient descent, normal distributions, etc.

By the end of the talk, the audience will:
1. Understand the motivation behind autoencoders as deep generative models.
2. Know how to incorporate labels into the training of a VAE, making it semisupervised.
3. Gain practical insights on how to implement a semisupervised VAE.
4. See the various benefits of using such a model, including better predictive and optimization performance, as well as interactivity.

# Detailed outline

**Motivation (3 minutes)**
- With an autoencoder, dimensionality reduction is done by bottlenecking the middle portion of a neural network whose output produces reconstructed objects.
- The space produced by the middle portion, a.k.a. the _latent space_, may also be used for downstream tasks such as supervised learning, blackbox optimization, and visual analytics.

**Basic autoencoders (5 minutes)**
- An autoencoder is implemented and trained on the MNIST handwritten digit dataset.
- The learned latent space shows separation between the classes, suggesting that this approach is effective at capturing the variation between the datapoints.

**Variational autoencoders (5 minutes)**
- A drawback of the latent space learned by a vanilla autoencoder is that it could become disjoint and non-continuous during training, which is undesirable.
- Variational autoencoders address this problem by imposing a normal probability distribution over the latent vectors and reconstructing objects from samples drawn from this distribution.
- This modification comes at a low cost thanks to the so-called reparameterization trick in machine learning.
- The resulting latent space is more condensed and closer to a normal distribution.

**Semisupervised variational autoencoders (7 minutes)**
- So far, the models we have been working with are trained in an unsupervised manner.
- The neural network architecture could be extended to contain an MLP predictor starting from the latent space for prediction jointly with reconstruction.
- The joint objective loss function decomposes into a sum of the VAE loss and the prediction loss, making interpretation and implementation straightforward.

**Benefits of the semisupervised approach (7 minutes)**
- The resulting latent space shows further separation between the classes, while maintaining compactness and normality from the VAE.
- By increasing the weight of the predictive loss in the summed objective, one may control how much predictive performance should drive optimization. A good visualization tool for this is a slider for this weight showing the resulting latent space, directly allowing a user to visually observe the effect of the weight.
- One could take the gradient of the objective with respect to the vectors in the latent space to show how the space would change when conditioned on fictitious data. This "counterfactual" latent space is another tool to help a user interpret what is being learned in a VAE.
- While the MNIST dataset poses a classification problem, we also show what will happen if an MLP regressor is trained in lieu of a classifier: the resulting latent space will show a smooth gradient across the classes going from 1 to 9. This demonstrates the flexibility of the approach.
- Aside from these interactivity-centric benefits, we show that a semisupervised VAE also leads to better performance in prediction and blackbox optimization.

**Q&A (3 minutes)**

# Other notes

This talk is a cleaned-up version of a project I am taking part in for my graduate research, so it's a topic I'm quite familiar with and very excited to talk about!
The idea is fairly simple and natural, but it hasn't been widely adopted by the deep learning community, so the hope/goal is to help popularize this framework further.

Some parts of the talk will overlap with [Martin Krasser's excellent blog post](https://github.com/krasserm/bayesian-machine-learning/blob/dev/autoencoder-applications/variational_autoencoder_opt.ipynb), which covers the same idea and some of the benefits of the model.

This is my first time submitting to PyCon US!
Some talks I have given at other venues are [this talk on preference learning and optimization](https://pydata.org/global2021/schedule/presentation/133/making-the-perfect-cup-of-joe-active-preference-learning-and-optimization-under-uncertainty/) at this year's PyData Global and [this talk on Bayesian machine learning](https://discourse.pymc.io/t/bayesian-machine-learning-a-pymc-centric-introduction-by-quan-nguyen/5985) at last year's PyMCon.
