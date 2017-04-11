This is my second time going through this paper. This is another Bengio one. The idea is kind of in the title, they make a hierarchical multi scale RNN. The big difference between this and previous work is the fact that they use a binary boundary condition.

The network is a series of stacked LSTMs (the stack size is a hyper parameter). Lower levels upon hitting a boundary condition will pass their information to higher levels and then flush out their own memories. The lowest level reads in the input and the highest spits out the output.

There are three operations thusly: Update, Flush and Copy (for when there is not a boundary condition at the level below you, you just stay the same).

In order to get a discrete (non differentiable) boundary condition to work they use a technique called the straight through estimator where they replace that non differentiable function with a differentiable one in the backward pass. 

It does really well and the visualizations are pretty convincing that it found the boundary. However it is a pretty complex formulation for not too big of gains…


