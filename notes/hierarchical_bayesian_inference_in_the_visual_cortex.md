# Gist

The gist of the paper is to propose a new theory of visual cortex inference, and you guessed it, the theory is using Hierarchical Bayesian inference. As an incredibly brief review of the visual cortex, we have five layers of importance: LGN, V1, V2, V4, and IT (whatever happened to V3?). Explaining the activations of neurons in this model can be phrased in a bayesian way as below:

P(x_0 , x_{v1}, x_{v2}, x_{v4}, x_{IT}) = P(x_0 | x_{v1}) P(x_{v1} | x_{v2}) ... P(x_{IT})

Thus they are able to conditionally factorize the above. They have three ideas:

1. Their idea is that each layer seeks to maximize the above via belief propagation. 
2. The systems use particle filtering to keep multiple hypothesis alive
3. That V1 is a high resolution geometric buffer

They back their claims up with experimental evidence (eg. experiments on monkeys and college students):

1. They show that V1 and IT activate concurrently 
2. They show that IT processes high level course data and V1 low level fine data
3. They show that V1 can adopt high level processes (like edge completion) after receiving feedback from higher levels
4. And they partially showed support for the multiple hypothesis by showing that low level (V1) processing is dampened when there is a concrete high level hypothesis that can explain the data
