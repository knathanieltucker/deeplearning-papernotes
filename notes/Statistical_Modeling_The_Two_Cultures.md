Breiman, aka Random Forests, presents an interesting story about two communities: the statistical one and the machine learning one. There is a subtle and important difference between these two communities and it all revolves around modeling data. If you are modeling the data distribution (assuming the data is distributed normally or that the target given the factors is distributed Bernoulli) then you fall in the statistics camp. And if you don’t make any assumptions about your data (other than it is sampled independently) then you fall into the machine learning came. Breiman falls far into the ML camp with some words of caution for those that will only consider the stats camp.

He goes through a lot of specific examples of data modeling flaws: the omnibus goodness of fit test has little power when it tests in all directions simultaneously, if you tinker with variable selection goodness of fit tests are not applicable, residual analysis only works in small number of dimensions, there is no standard for the comparison of models, etc. The list goes on. The fact is you can’t have it both ways. You cannot rely on troves of literature and results about data models without making sure your data fits the model. 

He elaborates on three big points:

1. The multiplicity of good models. There can be many possible models with a test error rate within 1% of each other. And each paints a very different picture of how data is generated.
2. Simplicity vs. Accuracy. This talks about how predictive a model is vs. how interpretable.
3. The curse of dimensionality. Here he shows that increasing the dimensions may help in some models (check out C. Olah’s blog for a great example of this). 

Overall really good. The message really was that there should be more statisticians working on ML and that without it some of the biggest problems will not be solved.
