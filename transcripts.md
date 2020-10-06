# Transcripts

### Slide 1
- We will start with the canonical problem in Bayesian interference: estimating the success rate of a binary event.
- This event can be the face of a flipped coin, whether or not a new drug will have a positive effect on a specific patient, and so on.
- We will call this rate, which is a real number between 0 and 1, $\theta$.
- To know the exact value of $\theta$ we need to perform a survey on all possible instances of the problem, which is impossible.
- But we can collect the information in question on a sample of a population.
- We denote our data as $\mathcal{D}$, which is simply a collection of 0's and 1's.
- A 0 indicates failure, while a 1 indicates a success in whatever sense our question is regarding.
- $n$ here is the size of our sample.
- With the data from this sample, we would like to say something about this $\theta$, maybe what value it most likely takes, or between what range it is most likely in.
- In other words, we would like to perform _inference_ on this random variable.

### Slide 2
- The first component in our Bayesian framework that we need is our prior belief about $\theta$, which is expressed as a probability distribution, defined across its possible values.
- In our specific case, it is a distribution with support $[0, 1]$.
- Now, the prior distribution, or prior belief, varies from person to person.
- If you have some reason to believe that the success rate is fairly high, then you prior distribution will look something like this.
- Here, your prior distribution will have a lot more mass on the left hand side, indicating that $\theta$ should be a large.
- If you, on the other hand, don't have an idea about what value $\theta$ should take, you will want to express that using a uniform distribution between 0 and 1, which simply says to you, all values are equally likely.
- Again, I want to emphasize the point that your choice of priors is unique to you and depends on your personal belief about the latent variable.
- How to choose an appropriate prior is a deep problem in and of itself, and I believe there is a talk at this PyMCon on this very topic.
- For now, we will move on with our current problem of inferring $\theta$.

### Slide 3
- In addition to the prior, we also need the likelihood function to denote the probability of the observed data given a specific hypothesis.
- Here we need the function computing the probability of $\mathcal{D}$ given a specific value of $\theta$.
- We assume first that there are $k$ successes and $(n - k)$ failures in our sample dataset.
- This we can calculate fairly easily: if the true success rate is some $\theta$, then each test in our sample has $\theta$ probability of being a success.
- Moreover, the tests don't affect one another; in other words, each of the numbers is $\mathcal{D}$ is independent from one another.
- So, using the product rule, we can compute the probability of entire $\mathcal{D}$ as the product of the individual probabilities like so.
- If we treat the outcome of each test as a binary random variable, we say that the variable follows a Bernoulli distribution with parameter $\theta$.
- We usually call this the predictive distribution, as it is the belief about the _outcome_ that we'd like to model.

### Slide 4
- So, we have our prior distribution and our likelihood function.
- Now, we can apply Bayes' theorem to compute the probability of $\theta$ being a specific value, given our observed data as denoted here.
- We can compute the numerator easily, as we already have access to both the prior and the likelihood.
- The question remains: how to compute the denominator in this question?
- First, we see that using the sum rule, we can rewrite the denominator as the sum of the numerator for all values of $\theta$.
- Still, computing the integral is often intractable except for a very specific set of cases.
- This is where we turn to sampling techniques to approximate the quantity.
- Specifically, say we draw samples from the prior distribution.
- Then, we approximate the integral as the sum of the corresponding terms of these samples.
- The idea is that, if we can draw the samples in a way that the aggregate shape roughly resembles the shape of the whole distribution, then our approximation will roughly be equal to the true integral.
- Numerous sampling techniques have been developed, and MCMC is one of the most commonly used, hence the name of the tool that we are all using: PyMC.
- So either using samples as an approximation or actually computing the integral, we obtain the posterior distribution.
- This value indicates the probability of $\theta$ being a specific number having observed the data, computed as the product between the prior and the likelihood before being normalized.
- A natural interpretation here is that the posterior is influenced by both the prior and the likelihood function, which makes sense: our posterior belief about a quantity driven by both our prior belief and the data that we observe.
- This calculated posterior probability defines the posterior distribution for $\theta$.
- This is a function with the domain between 0 and 1 denoting our posterior belief about the possible values of $\theta$.
- And that is the complete Bayesian inference procedure for this specific example; let's now look at some visualizations.

### Slide 5
- Let's say the true, unknown value of $\theta$ is $0.3$, so 30 percent of the time, the event we're considering will result in a success.
- Say our prior belief about $\theta$ is encoded by this blue line, so before seeing any data, we believe that $\theta$ is around 0.8.
- Our sample data is this collection of 100 values, which I generated randomly using NumPy.
- It's visualized by these two bar plots.
- Each of the 100 values has a $0.3$ probability of being 1.
- Now, after performing this Bayesian inference, our posterior distribution will become this green line.
- We see that the posterior now gives more mass to values around $0.4$ and less mass to values far from it.
- This is exactly what we want, as our posterior belief is pointing us to the right direction of where the true $\theta$ really is.
- Now, to estimate $\theta$, we can use a central tendency statistics such as the mean or the mode of the posterior distribution.
- But more importantly, this posterior distribution allows us quantify our uncertainty about $\theta$ as well: for example, if the posterior is highly concentrated around some number, then we are more certain about our belief; conversely, if the posterior is more spread out, then we are uncertain about $\theta$.
- Instead of the mean or the mode, sometimes we'd like to quantify the uncertainty of a variable by reporting the 95% credible interval, which we can easily compute from the posterior distribution.
- Contrasting this with the 95% confidence interval from the frequentist perspective, which in itself is a far more confusing idea, we see that the Bayesian framework allows us to naturally quantify our belief about an unknown variable.
- Overall, while this example is fairly simple, it demonstrates the power of Bayesian statistics in allowing the statistician to incorporate their prior belief in the form of the prior distribution and combine it with the observed data to obtain a posterior distribution, which offers a natural way to quantify uncertainty.
- Moreover, even in more complicated applications, the Bayesian framework remains the same: choosing your prior and working out your likelihood function, applying Bayes' theorem to obtain your posterior distribution, and finally extract insights from that posterior distribution.
- Throughout this example, we work with one single latent variable $\theta$, but the application of modeling a single variable can be limited.
- What if you instead want to model a function, which is in a sense a collection of infinite variables, each defined on a point?
- Here we turn to Gaussian processes, which are a really convenient mathematical object that allows us to use Bayesian inference to model a function.

### Slide 6
- Gaussian processes, or GPs, have a special place in Bayesian machine learning, because they offer so much flexibility in modeling functions of various shapes and smoothness.
- In the simplest sense, having a GP belief on a latent function first means we place a normal distribution prior on every point inside the domain of the latent function.
- A GP consists of a mean function, which specifies the central tendency of the latent function, and a covariance matrix, which defines covariance of any pair of points within the domain.
- This covariance matrix in effect controls how smooth we'd like to model our latent function as.
- By using this normal distribution-based prior on the function, we can actually derive the analytical expression of the posterior GP, conditioned on a set of observations as shown here.
- This math involves some matrix algebra.
- Luckily for us, most Bayesian software, including PyMC, already takes care of this computation.
- Overall, all we need to know is that after this conditioning, the posterior GP give us access to the posterior distribution of any point inside the domain of the function.

### Slide 7
- Let's now look at some examples so that we can have a visual understanding of what a GP can help us do.
- Say we have an arbitrary function, defined on $[0, 1]$, and we'd like to model it using a GP, which I'm showing the PyMC3 code for here.
- A constant or even zero mean function is typically used, while there are many more choices to be made regarding the covariance matrix, with the Matern 5 / 2 being one of the most commonly used, so those are our choices for this example as well.
- Now, notice that the covariance matrix has its own parameters $\eta$ and $\ell$, which respectively scale the input and output of the function, thus controlling the smoothness of the function.
- For now, we will set $\eta = 3$ and $\ell = 1$.
- Before observing any data, our belief about the function entirely depends on the GP prior we are placing on it, which is shown here.
- The bold line is the mean function plotted across the domain, while the shaded region is plotted by connecting the 95% credible interval (or CI) of each point in the domain, as defined by the prior.
- This means if you consider any point $x$ here, its prior mean is 0, and its CI is between these two points.
- For now, there's nothing interesting going on as we haven't observed any data yet.

### Slide 8
- Now, say we observe the value of the function at these specific points, as shown here.
- After conditioning on these observations, we visualize the posterior GP again using the posterior mean and CI.
- We see that the posterior mean nicely goes through the observation and the CI is squeezed in the regions around the observations.
- This is because within a smooth function, values in a small region are relatively similar, so after observing the value of a point, the uncertainty about the values of surrounding points decreases.
- The further we move away from the observation, the larger the CI gets, and if we are far away enough, the posterior pretty much reverts back to the prior.
- If we take a vertical slice of this plot at any unobserved point $x$, we obtain our posterior belief about $x$.
- And that is the general procedure of using a GP to model a function, or in other words, regression.

### Slide 9
- One note that I brushed over earlier is the choice of the covariance function parameters, which can be viewed as the hyper-parameters of our GP model.
- Again, these hyper-parameters specify the smoothness of the GP and therefore hugely influence the behavior of the posterior.
- For example, here are the posterior GPs when these hyper-parameters have different values.
- Now, within a specific context, a scientist can use her domain knowledge to find the best values for these hyper-parameters, but most of the time, this choice can be inaccurate.
- A better, more Bayesian strategy is to go hierarchical and place a prior on each of these hyper-parameters, and perform Bayesian inference on them as well.
- After conditioning on some observations, some point-estimate technique can help us set these hyper-parameters.
- It is at this point that PyMC sets itself apart from other Bayesian modeling tools using GP.
- For example, if you were to use scikit-learn's otherwise excellent GP implementation `GaussianProcessRegressor`, you would either need to set the hyper-parameters beforehand or use an optimizer to find their values, which doesn't allow you to encode your preference/expertise via a prior in the optimization.

### Slide 10
- Specifically, using PyMC3, I place a Gamma prior on $\ell$ and a Half Cauchy prior on $\eta$ in our running example; this results in the posterior GP here.
- Now, we see that by having this hierarchical structure for our GP, the fit improves from what we had before.
- As some of you might already know, defining a hierarchical structure for your Bayesian model by placing priors on parameters that define other variables is something that PyMC3 in general allows us to do quite easily, and it is actually what attracts me the most to the library.
- Okay so far we have talked about building Bayesian models on mathematical objects, either individual variables or functions, and perform inference on them.
- These methods are not unique to machine learning, and in fact, I'm pretty sure the statisticians watching this are more familiar with them than I am.
- What is unique to machine learning, on the other hand, is the technique of placing priors on parameters of an ML model and then use Bayes' theorem to obtain the posterior distributions of these parameters.
- These posteriors then induce a _posterior predictive distribution_ on the target variable that we'd like to perform predictions on, which gives us the same benefit of having a belief on the target as well as uncertainty quantification.
- The classical example of this is Bayesian linear regression, which we will consider next.

### Slide 11
- As a refresher, in a linear regression problem, we have a set of features or predictors $\textbf{x}$ and a target variable $y$, and we assume there's a linear relationship between them.
- We can encode our assumption using this equation, where $\textbf{w}$ is a vector of random coefficients or weight variables and $\varepsilon$ is the residual or noise.
- Our goal is to find the value for $\textbf{w}$ that will result in a good fit for our linear assumption.
- The exact definition of a good fit is up for interpretation, but the simplest way to solve this problem is to find $\textbf{w}$ such that the sum of squares of the residuals is minimized.
- This method is called least squares and if we were to do a bit of algebra, the optimal solution for $\textbf{w}$ can actually be found.
- With the optimized $\textbf{w}$, what we have in the end is a hyperplane that roughly goes through the points corresponding to our training data.
- In the 2D case where we have one feature, this becomes the so-called best-fit line that many of us are familiar with.
- So how would we solve the same linear regression problem the Bayesian way?
- I mentioned that $\textbf{X}$ is considered to be a vector of random variables, so the first thing that comes to mind is to place a prior on this object.
- The most common way to do this is to say $\textbf{w}$ follows a multivariate normal distribution, so each of the coefficients in $\textbf{w}$ has a normal prior.
- We also assume $\varepsilon$ is Gaussian noise with unknown standard deviation $\sigma$.
- This $\sigma$, which is a positive random variable, can also have its own prior.
- So with all of these priors set up, we can compute the likelihood of our observations $\textbf{y}$ using our linear assumption, and from there use Bayes' theorem to compute the posterior for each of our variables $\textbf{w}$ and $\sigma$.
- Again, what we will have in the end is a posterior belief for each variable, which together induce a posterior predictive distribution for $\textbf{y}$.
- This, as we mentioned before, helps us do point estimation if we'd like to exact out just a single value for $\textbf{w}$.
- More importantly, we can quantify the uncertainty that we have for our target variable $\textbf{y}$ that we're predicting on.

### Slide 12
- For example, still using a 2D toy example, we can plot out the credible interval for $\textbf{y}$, which somewhat looks like what we have with a GP, but here our belief about $\textbf{y}$ is constraint to be linear with respect to $\textbf{x}$, so our credible interval looks a bit different.
- However, the benefits are still the same: for each point that we're predicting on, we can naturally extract out a belief as a probability distribution about $y$.
- Here we can compute the mean or the mode for $y$ and that will give us an output that is almost equal to the least squares method in many cases, but we also can say how much uncertainty we have about this output.
- Again, I want to make a point about how what we have here is objectively better than the output from the least squares method, which again highlights the benefit of a Bayesian framework.
- A last note I want to include about this topic is that the procedure extends beyond linear regression and to polynomial regression and other models.
- As long as we have a defined relationship between our features $\textbf{x}$ and our target $\textbf{y}$, a we can place a prior on the parameters that we have and establish a likelihood for our observations, and everything else can follow.
- An example of this is in neural networks.
- Some of you might be familiar with this machine learning model, but in a classic neural net, we have a graphical structure that consists of many layers of nodes.
- Inside each node, the linear relationship we just talked about is applied to the input of the node and the output $\textbf{y}$ is put through a non-linear transformation.
- The final output of the network is what's used to perform predictions.
- The collection of these weights $\textbf{w}$ in all of the nodes is the parameters of the neural net that are optimized during training.
- So, if we were to apply the same Bayesian framework to this neural net model, we could simply again place a prior on all of these weights and perform Bayesian inference to obtain first the posteriors for the weights and second the posterior predictive for the final output.
- And this will give us the exact same benefit that we've been talking about: an updated belief about the output variable that defines not only the most likely value but also the uncertainty in that value.
- The case of Bayesian neural nets is a bit more special, however, since maintain and updating our belief about the collection of network weights, which is typically extremely large, is infeasible using simple sampling strategies, so we would need to use variational inference, which is quite well implemented in PyMC3.
- On this topic, Thomas Wiecki has a great talk at PyData, which I'm linking here so you could check it out yourself.
- And that's the end of what I wanted to talk about regarding Bayesian modeling, specifically making Bayesian-informed predictions.

### Slide 13
- Now we will transition to Bayesian decision theory, where we consider the process of making Bayesian-informed decisions.
- As we will see throughout this section, Bayesian decision theory consists of two main components: maintaining Bayesian beliefs about unknown quantities, which we just talked about, and using those beliefs to maximize the utility function.
- The term _utility function_ denotes our valuation for different outcomes of an unknown event.
- The function $u$ basically maps a decision leading to an outcome, which is random, to a specific numerical value, thus giving us a way to compare and express preference over the different decisions and outcomes.
- The concept is unique to each problem and sometimes even to each statistician, but as long as we have a well-defined utility function, we will be able to say things about the Bayesian optimal decision $d^*$, which is the decision that will in expectation lead to the best outcome, according to the current state of the world and utility function.
- To become more familiar with the idea of using the utility function, we will go through a quick toy problem, where we derive the best strategy for _The Price is Right_.

### Slide 14
- Our setup is a modified version of _The Price is Right_ as follows.
- We are competing against another player in a game where we have to guess the price $P$ of a product with a single number.
- Our opponent has already made her guess to be $\overline{p}$.
- Now, if our guess is above the actual price then we won't get anything, in which case our utility is 0. The same goes for our opponent.
- If neither of the guesses are above the actual price, the person with the closest guess will win the product.
- If we win, our utility will be the actual value of $P$, otherwise it will  be 0.
- Say we do have a belief about $p$, expressed as a normal probability distribution $\mathcal{N}(p; \mu, \sigma^2)$.
- Our task is to also make the Bayesian optimal guess, which is the guess that has the highest expected utility.

### Slide 15
- Now, the technique to derive the optimal decision in a Bayesian framework is relatively the same across problems.
- We first consider the utility of each action given the actual price, and then marginalize out that price according to our belief.
- Specifically, our utility is 0 if our guess $g$ is greater than $P$, regardless of what our opponent's guess is.
- In other words, $u(g \mid P) = 0$ if $g > P$.
- If $g < \overline{p} < P$, then our utility is also 0.
- Otherwise, $u(g \mid P) = P$ if $\overline{p} < g < P$.
- Then, for each value of $g$, we compute the expected utility as $\mathbb{E}[u(g)] = \int u(g \mid P) dP$.
- This can be approximated by drawing samples from our predictive belief about $P$ and computing $u$ with the rules that we have.
- Finally, we choose our guess $g$ to maximize this expected utility.
- So that is the general process of deriving the Bayesian optimal decision.
- Using the same procedure, we could write a function that takes in values sampled from the predictive distribution to represent our belief and plot out the expected utility as a function of our guess.

### Slide 16
- Let's say for our specific example, we have a predictive belief about $P$ as a gaussian with mean $100 and standard deviation 10, and our opponent has guesses at 75.
- Here I'm showing that plot that denotes the expected utility for potential guesses.
- We see that everything below 75 has an expected utility of 0, since our guess is lower than our opponent's, and we are most likely not winning given our belief.
- To obtain the Bayesian optimal decision, we simply locate the maximum of the plot and get the corresponding guess.
- In this case, it is a number just above 75.
- This makes sense since we only need to make sure that our guess is above our opponent's guess, so there's no value in going too far away from 75 and risking going above the actual price.
- This is why the expected utility quickly drops to 0 after that point.
- Of course, here our opponent is at a total disadvantage since she has to guess first, so the format of the actual show is not this simplified version.
- But overall, I hope that via this example, we can understand the concept of the Bayesian optimal decision and how to derive it better.
- Moving, we will start talking about actual problems in machine learning that can greatly benefit from a Bayesian perspective.

### Slide 17
- We will start with one of the canonical problems in the Bayesian decision theory literature: the multi-armed bandit problem.
- The setting we have is the following:
- We have $k$ slot machines $1, 2, ..., k$, each of which returns a coin with probability $\theta_i$ when its arm is pulled.
- We only have a limited number of pulls $N$ available to us.
- Our goal is to sequentially pull the arms of these machines so that our expected number of coins, which is typically called the reward, is maximized when we run out of pulls.
- The difficulty of the problem lies in the randomness of the reward returned by each machine.
- Say at a given point, we have identified a machine that returns a coin half of the time that it is pulled, and it's the best return rate out of the machines we have pulled.
- But there are also machines that we haven't pulled, or have only pulled once or twice, so we don't even know what their true return rate is roughly equal to.
- This is generally called the trade-off between exploration and exploitation that is inherent in many decision-making problems in machine learning.
- A solution to the multi-armed bandit problem is a way, which we will call a policy, of deciding which arm we should pull so that our chances of identifying the true best machine as quickly as possible are maximized.
- There are some naive policies we can immediately think of:
- A greedy policy will always choose the arm of the machine with the highest observed return rate so far at the risk of not exploring enough and failing to identify the true best machine.
- A purely explorative policy will choose a random arm every time, so its expected performance is simply the average return rate across all machines, which is not great.
- The question is, can we come up with a principle way of choosing an arm to pull at each iteration to balance between exploration and exploitation?
- Overall, many real-world problems can be posed as multi-armed bandit, such as designing clinical trials where we'd like to investigate the effect of a drug while minimizing adverse effects on patients, or personalized recommendations where we want to suggest customized products to a customer.
- Now, one way we could immediately apply a Bayesian framework to this problem is to model the return rate of each machine and its outcome in the same manner as the classic coin flipping example.
- Specifically, we place a prior, for example uniform, on each $\theta_i$, so the outcome of each machine, whether the machine will return a coin if pulled, follows a Bernoulli distribution with unknown parameter $\theta_i$.
- Now, at each iteration, we can obtain a posterior predictive distribution for this outcome with respect to every machine, and from there have a way to quantify our uncertainty about them.
- That still leaves open the question of how to design a policy to pick one arm to pull at each iteration.
- A good policy should prioritize arms that have high expected return rate and/or high variance in the posterior prediction distribution of the outcome, corresponding to exploitation and exploration.

### Slide 18
- One potential solution is to design the Bayesian optimal policy, which basically chooses the arm that in expectation will result in maximum current plus future reward.
- For example, if we only have one pull left, the Bayesian optimal decision is to greedily pick the arm with the best return rate.
- However, if there are two or more pulls left, the value of pulling an arm depends on not only the immediately expected reward from that arm, but also the expected impact it will have on our future beliefs and decisions, conditioned on the outcome of that arm.
- Overall, computing the Bayesian optimal decision is intractable when the number of pulls remaining is greater than 2.
- So we need some other policies that approximate this optimal policy.
- To quantify how good a policy is, we typically use the measure called regret, which is the difference in utility between a given policy and always pulling the unknown optimal arm.
- It has been proven that the best we could do is a regret that behaves like a logarithmic function of the number of iterations $t$, so our goal is to have policies that have regret of this behavior.
- Here we will discuss two of the most popular policies for this multi-armed bandit problem: UCB and Thompson Sampling.

### Slide 19
- There are many versions of the Upper-Confidence Bound, or UCB, policy, but in one Bayesian variant, we assign a point to each arm using this formula, which is a credible interval upper bound of the posterior distribution of the return rate variable.
- At each iteration, the arm with the highest score will be pulled.
- We see that this policy naturally balances exploration and exploitation by using this upper credible interval bound, since if an arm has a high empirical return rate, then the whole posterior credible interval will take on high values.
- On the other hand, an arm that has not been pulled many times will have a fairly flat posterior belief, and the upper bound of the credible interval will also have a high value.
- Of course, if an arm is obviously not optimal, its posterior distribution will take on low values and its UCB score will be low, de-prioritizing it from being chosen.
- By using the scoring rule, we will sequentially explore the set of available machines and converge on one that gives out the best reward.
- And that is the idea behind UCB.

### Slide 20
- The second policy we will take a look at is called Thompson Sampling.
- Unlike UCB where the procedure of choosing the next arm to pull is deterministic, Thompson Sampling is a randomized policy, meaning that when faced with the same information, it is not guaranteed that the policy will make the same choice every time.
- Again, at each iteration, we maintain a posterior belief about each $\theta_i$ as a probability distribution.
- Thompson Sampling tells us to sample from these posteriors, which will give us a rough estimate for each $\theta_i$, denoted $\overline{\theta_i}$.
- Then, we simply choose the arm corresponding to the largest estimate $\overline{\theta_i}$, and that will be our decision for that iteration.
- As you can see, this procedure is randomized, as the sampling step is inherent not deterministic.
- The intuition behind Thompson Sampling is quite simple but also elegant:
- If we are very certain in our belief about a specific $\theta_i$ being the max return rate, then its posterior distribution is very concentrated around a relatively large number, which means a sample from this distribution will be very likely to be close to that large number as well, thus making arm $i$ more likely to be chosen.
- On the other hand, if we are very uncertain about $\theta_i$, its posterior distribution is more widespread, and it is also likely that a sample from this distribution is a large value, again causing arm $i$ to be chosen.
- So this sampling and choosing the largest procedure naturally favors posteriors that either are concentrated around a large value, in other words exploitation, or have more uncertainty and are widespread, or exploration.
- Sid Ravinutala recently published a great blogpost on Thompson Sampling in the context of Covid testing, which I'm linking here and highly recommend you check it out.
- Okay, so that's the problem of multi-armed bandit and the two most common solutions to it.
- Theoretically, we could actually prove that the expected regret resulting from applying either of these solutions does have a logarithmic trend, so it is a good guarantee to have when these policies are applied in real-life problems.

### Slide 21
- Now I want to move on to a somewhat related topic called Bayesian optimization.
- In general, the term _Bayesian optimization_ denotes not a specific algorithm but a Bayesian framework of decision-making for optimization problems.
- The setup for such a problem is simple.
- We have access to the output of a function via queries but not its gradients or its functional form, and sometimes the function we want to optimize doesn't even have a functional form, so all the gradient-based optimization routines are not applicable.
- Also, querying this function is expensive, either costing a lot of money or time-consuming or other definitions of cost, so the number of queries we can make to the function is very limited.
- What's more, our observations might be noisy and not exactly the function values.
- The go-to example to motivate this problem is hyper-parameter tuning of neural network models.
- Most of the time, we want to set these hyper-parameters so that the predictive performance of the model increases.
- This could be accuracy, area under the curve, or some other metrics, but it is almost impossible to tell the functional form of these metrics in terms of the hyper-parameters, or if one even exists.
- What's more, rerunning the whole neural net with a new set of hyper-parameters can be quite time-consuming, since neural nets can take a long time to train.
- The question laid out before us is that, how can we design an optimization policy of sequentially making queries to the objective function so that the maximizer of the function is identified when our queries run out.
- Recall that in the multi-armed bandit problem, we model our belief about the return rate $\theta_i$ of each machine using the Bayesian framework.
- The same can be done in this problem as well, but here we don't have individual latent variables $\theta_i$ that we need to worry about.
- Instead, we have an entire function that we'd like to model and update our belief with new observations.
- As you can already guess, this is where Gaussian processes come in.
- As we discussed, a GP prior can be placed on a function to help us model our belief about the value of the function at each point in the domain.
- Additionally, this gets updated with new observations to reflect our posterior belief about the function at unobserved points.
- Now, our task is again to use this probabilistic belief about the function, modeled via a GP, to make informed decisions as to where to query the function, so that when our budget is depleted, we can identify the maximizer of the function.
- So far, you might have noticed that there is a parallel between Bayesian optimization and multi-armed bandit.
- In both problems, we are faced with a decision of choosing where to evaluate next, either of an objective function or among a set of slot machine arms to pull.
- In multi-armed bandit, we model individual return rates using the Bayesian framework, while in Bayesian optimization, we use a GP as the equivalent for a function.
- In multi-armed bandit, the Bayesian optimal decision is intractable except when we only have one or two iterations left.
- In Bayesian optimization, it's quite easy to see that the optimal decision is intractable except for the very last decision as well.
- If we have more than one query left, we would need to iterate through all possible queries, which are infinitely many, but also condition each possible outcome of each query to look ahead.
- Therefore, in Bayesian optimization, we also need to design policy that approximates the Bayesian optimal one.

### Slide 22
- At each iteration, we have the posterior predictive distribution of the function value of each point in the domain.
- This object gives rise to some of the most common Bayesian optimization policies.
- For example, the _Probability of Improvement_ policy calculates the probability that the function value of each point is greater than the current maximum function value that we have observed, $\overline{y}$.
- This requires us to compute the CDF of a normal distribution among points in the domain, which is quite easy to do.
- The _Expected Improvement_ policy, on the other hand, calculates its score as the expected improvement from the current maximum function value that we observed and chooses the point with the largest score.
- The posterior predictive distribution also allows us to use the GP equivalent of the UCB policy, which computes the credible interval upper bound of each unobserved point in the domain and chooses the maximizer.
- We see that similar to what we have seen, all of these policies again balances the trade-off between exploration and exploitation by prioritizing points with large expected value or large uncertainty in its posterior distribution.

### Slide 23
- Given the posterior predictive distribution of the objective function, we can also consider the distribution of the location of the true maximizer of the function, $x^*$.
- This object motivates several other optimization policies that seek to minimize the uncertainty about $x^*$ via their queries such as _Entropy Search_ and _Predictive Entropy Search_.
- If this idea sounds strange to you, a minimal equivalent case is binary search of a sorted array.
- By querying the number in the middle of the current sub-array at each iteration, we effectively minimize our uncertainty about where the number we are looking for is.
- _Entropy Search_ and _Predictive Entropy Search_ work roughly in a similar way, but the target of the search is the location of the function maximizer.
- Interestingly, Thompson Sampling has its own analog in the Bayesian optimization problem, where we first assign each unobserved point a score equal to the probability that the true function maximizer is at that point.
- Then, we randomly sample the points with probabilities proportional to these scores and query at the point we sampled.
- Again, these policies are motivated by finding the location of $x^*$, powered by the GP belief that we maintain on the objective function, which is quite a unique technique that is only possible within a Bayesian framework.
- Implementation-wise, Python offers a wide range of good libraries that implement Bayesian optimization policies.
- Here I want to highlight a relatively new library called pyGPGO, which integrates a PyMC backend with GP modeling.
- Dan Foreman-Mackey is one of the authors of this tool, so I highly recommend checking out this work.
- And with that, we reach the end of what I wanted to cover regarding Bayesian optimization.

### Slide 24
- Now, I want to briefly go over some other, more exotic, problems in machine learning where Bayesian decision theory thrives.
- First, we have the subfield called active learning, where the goal is to identify minimal training data that will give a given ML model the best predictive performance.
- If you are familiar with _Support Vector Machine_ classifiers, you might remember that if the training data is reduced to just the support vectors, which are the data points that are on the classification boundary, the resulting trained model will not change.
- The idea is then generalized in active learning, where we want to identify the smallest set of data points to train our models on without sacrificing any predictive performance.
- Typically this is used when obtaining the label of a data point is costly in the same way as Bayesian optimization: it is either expensive, time-consuming, or undesirable to do many times in some other way.
- Related is the active search problem, where we still have a constraint on the number of queries we can make, but the goal is to identify as many members of a rare class as possible.
- In other words, this is an active learning problem where we'd like to maximize recall.
- The ability to actively search for and make queries that are optimal in expectation with respect to a specific goal comes straight from our belief about our environment, represented as probability distributions and worked out using Bayesian decision theory.
- Results of these subfields of ML can lead to cheaper and more efficient ML training in applications where, again, labeling is costly and uncertainty plays an important role such as clinical trials, fraud detection, or scientific experiments.

### Conclusion
- That also concludes my talk on Bayesian methods in machine learning.
- Overall, we have discussed two specific topics: Bayesian modeling and Bayesian decision theory in the context of ML.
- I hope I have convinced you that going from Bayes' theorem, we can design Bayesian frameworks to address problems in ML in a more informed, principled way.
- A second point that I wanted to make is PyMC3's ability to allow fast, easy building of probabilistic models, especially ones with hierarchical structures.
- Thanks for tuning in, and if you have any questions about the material, please head over to discourse and participate in the conversation there.
