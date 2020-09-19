# Bayesian methods in machine learning: a brief introduction
## Transcripts

### Slide 1: Speaker info and links
- Hello everyone, thanks for joining my talk, _Bayesian methods in machine learning: a brief introduction_.
- My name is Quan, I'm a Ph.D. student in computer science at Washington University in St. Louis.
- My research partially focuses on the topics discussed in this video so I'm very excited to go through it with you.
- Overall, I'll be talking about ideas and techniques used in machine learning that are influenced or even powered by Bayes' theorem and Bayesian statistics.
- The goal I have in mind for this talk is to cover quite a number of topics at a relatively superficial level, and the materials covered might not be as detailed as some of you had hoped.
- This is so that we can maximize the likelihood that a person finds a specific topic in this talk interesting.
- That said, I will be including references to more in-depth readings throughout so that you can navigate materials available online on what you're interested in better.
- With that out of the way, let's get started.

### Slide 2: Agenda
- So in this talk, we will be specifically focusing on two main topics: Bayesian modeling and Bayesian decision making.
- Both are essential to Bayesian machine learning as well as any Bayes-centric decision-making system.
- The first topic goes through some of the most common choices in terms of modeling a latent variable, which includes making inferences,
- If you are new to Bayesian statistics and inference, this discussion will help get you up to speed with the main ideas in this broad field.
- Otherwise, if you are already familiar with the topic, the first part of the video can still give you a quick refresher, or you can skip to the second part about decision-making.

### Slide 3: Intro to Bayes
- Most of you might are tired of seeing this picture, but I'll be remiss if I don't include this obligatory Bayes' theorem neon sign.
- All things Bayesian-related start with this formula, which calculates the conditional probability of event A happening, given that event B happens.
- You the statisticians might already be familiar with a more specific interpretation: the probability of a hypothesis being true given the observed data.
- Here the hypothesis can be about anything of interest: the latent variable being equal to a specific number, the latent variable being in a specific range, etc.
- We also maintain a collection of exhaustive and mutually exclusive competing hypotheses, and the probability of each hypothesis being true before observing any data.
- This is also call the prior probability to denote your state of knowledge _prior_ to making any observation.
- And finally, let's say we have observed some data and have access to a likelihood function, denoting the probability of that observed data given a specific hypothesis being true.
- Bayes' theorem will help us calculate the probability that a specific hypothesis being true, given the observed data.
- This quantity is often called the _posterior_ to contrast the prior.

### Slide 4: Bayesian vs. Frequentist
- If you are familiar with the frequentist approach to hypothesis testing, already you can see the difference in the two ways of thinking.
- As a frequentist statistician, you will only be concerned with the likelihood function.
- While as Bayesians, we treat the latent quantity about what we care about as a random variable, which can take on values as defined by the set of our hypotheses.
- You can make the argument, and in fact many do, that thinking about posterior probability is more natural than thinking about the likelihood, as the latent variable is what we are unsure of, as opposed to the data, which we can observe.
- Not only a more natural way to think about uncertainty, a Bayesian framework can also help address many problems in machine learning, as we will see in this talk.
- And with that, let's move on to our first topic: Bayesian modeling.


### Slide 5: A starting example
- We will start with a simple problem: estimating the proportion that prefers Google Chrome as a web browser to Firefox, out of the entire population of Internet users.
- We will call this proportion, which is a real number between 0 and 1, $\theta$.
- To know the exact value of $\theta$ we need to perform a survey on every Internet user, which is impossible.
- But we can collect this preference information on a sample of this population.
- We denote our data as $\mathcal{D}$, which is simply a collection of 0's and 1's.
- A 0 indicates that a person in our sample does not prefer Chrome to Firefox, while a 1 indicates that that person does.
- $n$ here is the size of our sample.
- With the data from this sample, we would like to say something about this $\theta$, maybe what value it most likely takes, or between what range it is most likely in.
- In other words, we would like to perform _inference_ on this random variable.

### Slide 6: Making inferences: the prior distribution
- The first component in our Bayesian framework that we need is our prior belief about $\theta$, which is expressed as a probability distribution, defined across its possible values.
- In our specific case, it is a distribution with support $[0, 1]$.
- Now, the prior distribution, or prior belief, varies from person to person.
- If you have some reason to believe that most people prefer Chrome, maybe because you yourself really like Chrome, then you prior distribution will look something like this.
- If you have a reason to believe otherwise, your prior distribution will have a lot more mass on the left hand side, indicating that $\theta$ should be a large.
- If you, on the other hand, don't have an idea about what value $\theta$ should take, you will want to express that using a uniform distribution between 0 and 1, which simply says to you, all values are equally likely.
- Again, I want to emphasize the point that your choice of priors is unique to you and depends on your personal belief about the latent variable.
- How to choose an appropriate prior is a deep problem in and of itself, and I believe there is a talk at this PyMCon on this very topic.
- So if you'd like to learn more, definitely check it out.
- For now, we will move on with our current problem of inferring $\theta$.

### Slide 7: Making inferences: the likelihood
- Again, in addition to the prior, we also need the likelihood function to denote the probability of the observed data given a specific hypothesis.
- Here we need the function computing the probability of $\mathcal{D}$ given a specific value of $\theta$.
- This we can calculate fairly easily: if the true proportion of people preferring Chrome is $\theta$, then each person in our sample has $\theta$ probability of preferring Chrome.
- Moreover, each person's preference does not affect another's; in other words, each of the numbers is $\mathcal{D}$ is independent from one another.
- So, using the product rule, we can compute the probability of entire $\mathcal{D}$ as the product of the individual probabilities, denoted here.

### Slide 8: Making inference: Bayes' theorem
- So, we have our prior distribution and our likelihood function.
- Now, we can apply Bayes' theorem to compute the probability of $\theta$ being a specific value, given our observed data as denoted here.
- We can compute the numerator easily, as we already have access to both the prior and the likelihood.
- The question remains: how to compute the denominator in this question?
- First, we see that using the sum rule, we can rewrite the denominator as the sum of the numerator for all values of $\theta$.

**If need more content**: conjugacy

- Still, computing the integral is often intractable except for a very specific set of cases.
- This is where we turn to sampling techniques to approximate the quantity.
- Specifically, say we draw samples from the prior distribution
- Then, we approximate the integral as the sum of the corresponding terms of these samples.
- The idea is that, if we can draw the samples in a way that the aggregate shape roughly resembles the shape of the whole distribution, then our approximation will roughly be equal to the true integral.
- Numerous sampling techniques have been developed, and MCMC is one of the most commonly used, hence the name of the tool that we are all using: PyMC.
- So either using samples as an approximation or actually computing the integral, we obtain the posterior distribution.
- This value indicates the probability of $\theta$ being a specific number having observed the data, computed as the product between the prior and the likelihood.
- A natural interpretation here is that the posterior is influenced by both the prior and the likelihood function, which makes sense: our posterior belief about a quantity is driven by both our prior belief and the data that we observe.

### Slide 9: the posterior distribution
- This calculated posterior probability defines the posterior distribution for $\theta$.
- This is a function with the domain between 0 and 1 denoting our posterior belief about the possible values of $\theta$.
- And that is the complete Bayesian inference procedure for this specific example; let's now look at some visualizations.
- Let's say the true, unknown value of $\theta$ is $0.7$, so 70 percent of Internet users prefer Chrome to Firefox.
- Our sample data is this collection of 100 values, which I generated randomly using NumPy.
- Each of the 100 values has a $0.7$ probability of being 1.
- Now, if our prior is the uniform distribution, and after performing this Bayesian inference, our posterior distribution will become this.
- We see that the posterior now gives more mass to values around $0.7$ and less mass to values far from it.
- This is exactly what we want, as our posterior belief is pointing us to the right direction of where the true $\theta$ really is.

**If need more content**: different priors resulting in different posteriors

- While this example is fairly simple,

### Slide 10:
