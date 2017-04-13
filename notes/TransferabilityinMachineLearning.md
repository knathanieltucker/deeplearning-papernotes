Oh man, this one is filled with goodies! I remember reading the first adversarial training paper by Goodfellow, really super cool and it is good to see him continuing to pursue this line of work.

So this paper is a bag of tricks that can teach you to become an ML hacker. 

They explore adversarial sample transferability. This is the idea that algorithms trained on different samples of the same dataset have the same adversarial examples, but they go a lot further. They show that not only different architectures, but different algorithms altogether have similar adversarial examples. I will refer you to the paper itself to see which algorithms are the most resilient.

They then talk about black box hacking. This is the idea that if you don’t know anything about the algorithm but are able to query its API, you can still hack it. The introduce two tricks to help out in this task: periodical step size and reservoir sampling (this first has to do with the step size along the sign Jacobean that you take when generating adversarial samples, and the second seems to be just like sampling… I don’t really get it). Anyways, they use DNN and LR substitutes to hack both Amazon and Google.

Finally they invent ways to generate adversarial samples for SVMs and DT. One by marching towards the decision boundary, and the other presumably by marching up the tree and then perturbing.

On the whole pretty cool (but a lot of the tricks were very underwhelming). It is almost a nice survey paper of adversarial ML attacks. 
