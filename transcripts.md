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
- Then we will maintain a collection of exhaustive and mutually exclusive competing hypotheses
