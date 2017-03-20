So a good one from Microsoft Research. This is my second time reading this, so I thought I would go ahead and write something up.

The central question of the paper is: is learning better networks as easy as stacking more layers? And their answer is a resounding yes for the most part. But there are some challenges that we need to overcome first:

1. Vanishing gradients (mostly solved by batch normalization in this paper)
2. Degradation problem (solved by this paper)

The degradation problem is simply put: the training accuracy for deep networks degrades. To understand the degradation problem one just needs to consider this example. Shallow networks are a strict subset of deeper networks. So to achieve a shallow network’s accuracy all a deep network has to do is to copy the shallow network in the first few layers and then identity function your way to victory. This is also the inspiration behind resnets. 

All the residual network does is provide shortcut connections that simply perform the identity mapping. This work has been done before (most notably in highway networks), but they have invariably complicated this mapping by adding projections or gating. This while the whole idea of the degradation problem suggests that solvers have problems approximating the identity mapping.

There are a couple of tricks they use in the paper, but the majority of the contribution can be seen in this one equation:

y = sigmoid(x) + F(x)

The output of a residual layer is a function of the input plus the input itself.

In the end this model won ImageNet detection, localization COCO detection, and segmentation and a ton of different competitions. So you know it’s good.
