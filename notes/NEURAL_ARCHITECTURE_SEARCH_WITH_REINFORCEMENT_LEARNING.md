Another one by Quoc Le (super prolific).

The fundamental idea is to use an RNN controller to generate a model string (I’ll go into depth for the CNN model string). In their basic implementation they predict a number of CNN layers each represented by Filter Hight, Width, Stride Height, Width and Number of filters via an RNN where the outputs at each step are softmaxed and then fed into the next time stamp. The network stops after generating a certain number of layers (which they extend as training goes on). And they train the resulting network and get it’s validation set accuracy.

As of now the model is not differentiable. So they phrase the entire thing as an RL problem (specifically using the REINFORCE rule from Williams (1992)). In brief, they maximize the probability of making specific layers (weighting the layers by their validation set accuracy) with respect to the parameters of the controller RNN. 

That is the bulk of it.

There are a couple of tricks they use to get reasonable performance like adding in skip layers, adding a baseline to the RL formulation, and a ton of little heuristics (see page 5 directly under figure 4). But that is the gist.

They trained by training ~13k networks in a distributed fashion. The results were good. They beat SOTA on CIFAR-10 and PTB after piling a ton of heuristics on top.

What was perhaps most interesting to me is that they generated a performant RNN Cell called NAS through this (which is actually in TF now). 

Pretty cool, not sure how practical this is for anybody other than Google to do though…
