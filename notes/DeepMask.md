This is a good old FAIR goody that tries to do the really quite hard task of instance segmentation. The paper should be named DeepMask.

The architecture is simple. There is a VGG trunk that preprocesses the images. This feeds two networks: a segmentation and a scoring network. The segmentation network is not a decoder network but is rather a single classifier layer that outputs a binary mask. The scoring network will predict the objectness of the image. They are jointly learned. 

During the full scene inference this network is run densely over the image (sliding at multiple scales). 

There are a ton of implementation details so watch out for any of those brave enough to replicate these results!
