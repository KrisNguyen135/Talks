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
- If we treat the outcome of a person's preference as a binary random variable, we say that the variable follows a Bernoulli distribution with parameter $\theta$.
- We usually call this the predictive distribution, as it is the belief about the _outcome_ that we'd like to model.

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
- Now, to estimate $\theta$, we can use a central tendency statistics such as the mean or the mode of the posterior distribution.
- But more importantly, this posterior distribution allows us quantify our uncertainty about $\theta$ as well: for example, if the posterior is highly concentrated around some number, then we are more certain about our belief; conversely, if the posterior is more spread out, then we are uncertain about $\theta$.
- This component of uncertainty quantification is what sets Bayesian statistics apart from frequentism and power applications where uncertainty is an important measure.
- For example, instead of the mean or the mode, sometimes we'd like to quantify the uncertainty of a variable by reporting the 95% credible interval, which we can easily compute from the posterior distribution.
- Contrasting this with the 95% confidence interval from the frequentist perspective, which in itself is a far more confusing idea, we see that the Bayesian framework allows us to naturally quantify our belief about an unknown variable.

**If need more content**: different priors resulting in different posteriors

- Overall, while this example is fairly simple, it demonstrates the power of Bayesian statistics in allowing the statistician to incorporate their prior belief in the form of the prior distribution and combine it with the observed data to  obtain a posterior distribution, which offers a natural way to quantify uncertainty.
- Moreover, even in more complicated applications, the Bayesian framework remains the same: choosing your prior and working out your likelihood function, applying Bayes' theorem to obtain your posterior distribution, and finally extract your analysis from that posterior distribution.
- So throughout this example, we work with one single latent variable $\theta$, but the application of modeling a single variable can be limited.
- What if you instead want to model a function, which is in a sense a collection of infinite variables, each defined on a point?
- Here we turn to Gaussian processes, which are a really convenient mathematical object that allows us to use Bayesian inference to model a function.

### Slide 10: Modeling a latent function
- Gaussian processes, or GPs, have a special place in Bayesian machine learning, because they offer so much flexibility in modeling functions of various shapes and smoothness.
- In the simplest sense, having a GP belief on a latent function first means we place a normal distribution prior on every point inside the domain of the latent function.
- A GP consists of a mean function, which specifies the central tendency of the latent function, and a covariance matrix, which defines covariance of any pair of points within the domain.
- This covariance matrix in effect controls how smooth we'd like to model our latent function as.
- By using this normal distribution-based prior on the function, we can actually derive the analytical expression of the posterior GP, conditioned on a set of observations as shown here.
- This math involves some matrix algebra, especially taking the Cholesky decomposition here.
- Luckily for us, most Bayesian software, including PyMC, already takes care of this computation.
- Overall, all we need to know is that after this conditioning, the posterior GP give us access to the posterior distribution of any point inside the domain of the function.

### Slide 11: GP visualizations: prior
- Let's now look at some examples so that we can have a visual understanding of what a GP can help us do.
- Say we have an arbitrary function, defined on $[0, 1]$, and we'd like to model it using a GP.
- A constant or even zero mean function is typically used, while there are many more choices to be made regarding the covariance matrix, with the Matern 5 / 2 being one of the most commonly used, so those are our choices for this example as well.
- Now, notice that the covariance matrix has its own parameters $\eta$ and $\ell$, which respectively scale the input and output of the function, thus controlling the smoothness of the function.
- For now, we will set $\eta = 3$ and $\ell = 1$.
- Before observing any data, our belief about the function entirely depends on the GP prior we are placing on it, which is shown here.
- The bold line is the mean function plotted across the domain, while the shaded region is plotted by connecting the 95% credible interval (or CI) of each point in the domain, as defined by the prior.
- This means if you consider any point $x$ here, its prior mean is 0, and its CI is between these two points.
- For now, there's nothing interesting going on as we haven't observed any data yet, and this visualization is simply of the prior GP.

### Slide 12: GP visualizations: posterior
- Now, say we observe the value of the function at this specific point, as shown here.
- After conditioning on this observation, we visualize the posterior GP again using the posterior mean and CI.
- We see that the posterior mean nicely goes through the observation and the CI is squeezed in the region around it.
- This is because within a smooth function, values in a small region are relatively similar, so after observing the value of the point here, the uncertainty about the values of surrounding points decreases.
- The further we move away from the observation, the larger the CI gets, and if we are far away enough, the posterior pretty much reverts back to the prior.
- Now, as we have more and more observations, our posterior changes in a similar way where uncertainty, encoded in the CI, decreases around the observations.
- If we take a vertical slice of this plot at any unobserved point $x$, we obtain our posterior belief about $x$.
- And that is the general procedure of using a GP to model a function, or in other words, regression.

### Slide 13: GP hyper-parameters
- One note that I brushed over earlier is the choice of the covariance function parameters, which can be viewed as the hyper-parameters of our GP model.
- Again, these hyper-parameters specify the smoothness of the GP and therefore hugely influences the behavior of the posterior.
- For example, here are the posterior GPs when these hyper-parameters have different values.
- Now, within a specific context, a scientist can use their domain knowledge to find the best values for these hyper-parameters, but most of the time, this choice can be inaccurate.
- A better, more Bayesian strategy is to go hierarchical and place a prior on each of these hyper-parameters, and perform Bayesian inference on them as well.
- After conditioning on some observations, some point-estimate technique can help us set these hyper-parameters.
- It is at this point that PyMC sets itself apart from other Bayesian modeling tools using GP.
- For example, if you were to use scikit-learn's otherwise excellent GP implementation `GaussianProcessRegressor`, you would either need to set the hyper-parameters beforehand or use an optimizer to find their values, which doesn't allow you to encode your preference/expertise via a prior in the optimization.

### Slide 14: GPs in PyMC3
- Specifically, using PyMC3, I place a Gamma prior on $\ell$ and a Half Cauchy prior on $\eta$ for our running exmaple; this results in the posterior GP here.
- Now, as I show the true latent function that we're trying to model here, which is a piece of the Gramacy and Lee function, we see that by having this hierarchical structure for our GP, the fit improves from what we had before.
- As some of you might already know, defining a hierarchical structure for your Bayesian model, by placing priors on parameters that define the priors for other variables, is something that PyMC3 in general allows us to do quite easily, and it is actually what attracts me the most to the library, and I suspect the same goes for many others.

### Slide 15: Bayesian inference on machine learning parameters
- Okay so far we have talked about building Bayesian models on mathematical objects, either individual variables or functions, and perform inference on them.
- These methods are not unique to machine learning, and in fact, I'm pretty sure the statisticians in the room are more familiar with them than I am.
- What is unique to machine learning or ML, on the other hand, is the technique of placing priors on parameters of an ML model and then use Bayes' theorem to obtain the posterior distributions of these parameters.
- These posteriors then induce a _posterior predictive distribution_ on the target variable that we'd like to perform predictions on, which gives us the same benefit of having a belief on the target as well as uncertainty quantification.
- The classical example of this is Bayesian linear regression, which we will consider next.

### Slide 16: Linear regression
- As a refresher, in a linear regression problem, we have a set of features or predictors and a target variable, and we assume there's a linear relationship between them.
- We also have some observations on these features, which we denote matrix $\textbf{X}$, and the corresponding target variable, vector $\textbf{y}$.
- We can encode our assumption using this equation, where $\textbf{w}$ is a vector of random coefficient or weight variables and $\varepsilon$ is the residual or noise.
- Our goal is to find the value for $\textbf{w}$ that will result in a good fit for our linear assumption.
- The exact definition of a good fit is up for interpretation, but the simplest way to solve this problem is to find $\textbf{w}$ such that the sum of squares of the residuals is minimized.
- This method is called least squares and if we were to do a bit of algebra, the optimal solution for $\textbf{w}$ can actually be found.
- With the optimized $\textbf{w}$, what we have in the end is a hyperplane that roughly goes through the points corresponding to our training data.
- In the 2D case where we have one feature, this becomes the so-called best-fit line that many of us are familiar with.

### Slide 17: Bayesian linear regression
- So how would we solve the same linear regression problem the Bayesian way?
- I mentioned that $\textbf{X}$ is considered to be a vector of random variables, so the first thing that comes to mind is to place a prior on this object.
- The most common way to do this is to say $\textbf{X}$ follows a multivariate normal distribution, so each of the coefficients in $\textbf{X}$ has a normal prior.
- We also assume $\varepsilon$ is Gaussian noise with unknown standard deviation $\sigma$.
- This $\sigma$, which is a positive random variable, can also have its own prior; for now I will use an _Inverse Gamma_ distribution.
- So with all of these priors set up, we can compute the likelihood of our observations $\textbf{y}$ using our linear assumption, and from there use Bayes' theorem to compute the posterior for each of our variables $\textbf{w}$ and $\sigma$.
- Again, what we will have in the end is a posterior belief for each variable, which together induce a posterior predictive distribution for $\textbf{y}$.
- This, as we mentioned before, helps us do point estimation if we'd like to exact out just a single value for $\textbf{w}$.

### Slide 18: Credible interval in linear regression
- More importantly, we can quantify the uncertainty that we have for our target variable $\textbf{y}$ that we're predicting on.
- For example, still using the 2D toy example, we can plot out the credible interval for $\textbf{y}$, which somewhat looks like what we have with a GP, but here our belief about $\textbf{y}$ is constraint to be linear with respect to $\textbf{X}$, so our credible interval looks a bit different.
- However, the benefits we obtain are still the same: for each point that we're predicting on, we can take naturally extract out a belief as a probability distribution about $y$.
- Here we can compute the mean or the mode for $y$ and that will give us the same output as the least squares method in many cases, but we also can say how much uncertainty we have about this output.
- For example, over here where our credible interval is small, we can say that we're fairly certain about our output guess, but more uncertainty exists in other areas where we don't have many data points.
- Again, I want to make a point about how what we have here is objectively better than the output from the least squares method, which again highlights the benefit of a Bayesian framework.
- A last note I want to include about this topic is that the procedure extends beyond linear regression and to polynomial regression and other models.
- As long as we have a defined relationship between our features $\textbf{X}$ and our target $\textbf{y}$, a we can place a prior on the parameters that we have and establish a likelihood for our observations, and everything else can follow.

### Slide 19: Bayesian neural networks
- An example of this is in neural networks.
- Some of you might be familiar with this machine learning model, but in a classic neural net, we have a graphical structure that consists of many layers of nodes.
- Inside each node, the linear relationship we just talked about is applied to the input of the node and the output $\textbf{y}$ is put through a non-linear transformation.
- The final output of the network is what's used to perform predictions.
- The collection of these weights $\textbf{w}$ in all of the nodes is the parameters of the neural net that are optimized during training.
- So, if we were to apply the same Bayesian framework to this neural net model, we could simply again place a joint prior on all of these weights and perform Bayesian inference to obtain first the posteriors for the weights and second the posterior predictive for the final output.
- And this will give us the exact same benefit that we've been talking about: an updated belief about the output variable that defines not only the most likely value but also the uncertainty in that value.
- The case of Bayesian neural nets is a bit more special, however, since maintain and updating our belief about the collection of network weights, which is typically very large, is infeasible using simple sampling strategies, so we would need to use variational inference, which is quite well implemented in PyMC3.
- On this topic, Thomas Wiecki, one of the organizers of this conference, has a great talk at PyData, which I'm linking here so you could check it out yourself.

### Slide 19: Bayesian neural networks: an illustration
- I will end this topic and stealing one of his visualizations in that talk to really highlight the power of this method.
- Here the problem is the classify these points into two groups as colored here.
- A vanilla neural net could do this quite well, but when applying this method, we also have access to the _variance_ of the posterior predictive of the output, which is shown here.
- The variance is large in the boundary area, which makes intuitive sense since classification is typically harder in the edge cases.
- This helps us avoid unintuitively confident predictions that neural networks have been known to make.
- Here's a classic example in adversarial machine learning, where by introducing some random noise to pictures, researchers found that neural nets can get confused in the worst way possible: it's wrong, but it's very confident that it's right.
- With the natural ability to quantify uncertainty, the Bayesian method is one of the best candidates to combat this problem.

### Slide 20: The multi-armed bandit problem
- And that's the end of what I wanted to talk about regarding Bayesian modeling, specifically making Bayesian-informed predictions.
- Now we will transition to Bayesian decision theory, where we consider the process of making Bayesian-informed decisions.
- We will start with one of the canonical problems in the Bayesian decision theory literature: the multi-armed bandit problem.
- The setting we have is the following:
- We have $k$ slot machines $1, 2, ..., k$, each of which spits out a coin with probability $\theta_i$ when its arm is pulled.
- We only have a limited number of pulls $N$ available to us.
- Our goal is to sequentially pull the arms of these machines so that our expected number of coins, which is typically called the reward, is maximized when we run out of pulls.
- The difficulty of the problem lies in the randomness of the reward returned by each machine.
- Say at a given point, we have identified a machine that returns a coin half of the time that it is pulled, and it's the best return rate out of the machines we have pulled.
- But there are also machines that we haven't pulled, or have only pulled once or twice, so we don't even know what their true return rate is roughly equal to.
- This is generally called the trade-off between exploration and exploitation that is inherent in many decision-making problems in machine learning.
- A solution to the multi-armed bandit problem is a way, or we will call a policy, of deciding which arm we should pull so that our chances of identifying the true best machine as quickly as possible is maximized.
- Now, there are some naive policies we can immediately think of:
- A greedy policy will always choose the arm of the machine with the highest observed return rate so far at the risk of not exploring enough and failing to identify the true best machine.
- A purely explorative policy will choose a random arm every time, so its expected performance is simply the average return rate across all machines, which is not great.
- The question is, can we come up with a principle way of choosing an arm to pull at each iteration to balance between exploration and exploitation?
- Many real-world problems can be posed as multi-armed bandit, such as designing clinical trials where we'd like to investigate the effect of a drug while minimizing adverse effects on patients, or personalized recommendations where we want to suggest customized products to a customer.

### Slide 21: Bayesian modeling of the return rates
- Now, one way we could immediately apply a Bayesian framework to this problem is to model the return rate of each machine and its outcome in the same manner as our Chrome vs. Firefox example.
- Specifically, we place a prior, for example uniform, on each $\theta_i$, so the outcome of each machine, whether the machine will return a coin if pulled, follows a Bernoulli distribution with unknown parameter $\theta_i$.
- Now, at each iteration, we can obtain a posterior predictive distribution for this outcome with respect to every machine, and from there have a way to quantify out uncertainty about them.
- That still leaves open the question of how to design a policy to pick one arm to pull at each iteration.
- A good policy should prioritize arms that have high expected return rate and/or high variance in the posterior prediction distribution of the outcome, corresponding to exploitation and exploration respectively.

### Slide 22: The Bayesian optimal policy
- One potential solution is to design the Bayesian optimal policy, which basically chooses the arm that in expectation will result in maximum current plus future reward.
- For example, if we only have one pull left, the Bayesian optimal decision is to greedily pick the arm with the best return rate.
- However, if there are two or more pulls left, the value of pulling an arm depends on not only the immediately expected reward from that arm, but also the expected impact it will have on our future beliefs and decisions, conditioned on the outcome of that arm.
- Overall, computing the Bayesian optimal decision is intractable when the number of pulls remaining is greater than 2.
- So we need some other policies that approximate this optimal policy.
- Here we will discuss two of the most popular policies for this multi-armed bandit problem: UCB and Thompson Sampling.

### Slide 23: The Upper-Confidence Bound policy
- There are many versions of the Upper-Confidence Bound, or UCB, policy, but in one Bayesian variant, we assign a point to each arm using this formula, which is a credible interval upper bound of the posterior predictive distribution of the outcome variable.
- At each iteration, the arm with the highest score will be pulled.
- We see that this policy naturally balances exploration and exploitation by using this upper credible interval bound, since if an arm has a high empirical return rate, then the whole posterior predictive credible interval will take on high values.
- On the other hand, an arm that has not been pulled many times will have a fairly flat corresponding posterior predictive belief for the outcome, and the upper bound of the credible interval will also have a high value.
- So by using the scoring rule, we will sequentially explore the set of available machines and converge on one that gives out the best reward.
- And that is the idea behind UCB.

### Slide 24: The Thompson Sampling policy
- The second policy we will take a look at is called Thompson Sampling.
- Unlike 
