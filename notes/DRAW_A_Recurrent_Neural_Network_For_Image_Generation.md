Another really cool one out of Deepmind. 

We have an encoder network that determines a distribution over latent codes that capture salient information about the input data and a decoder that receives samples from the code and uses them to condition its own distribution over images. This is interesting for three reasons:

1. First is that the encoder is privy to the decoder’s previous outputs, so that it can tailor the code it sends according to previous output
2. Second, the decoder writes in steps
3. Third, they can both use attention

This network does not have to be used just for reading and writing digits, it is just the example of choice. 

The architecture is really well described. First there is an encoding of the image and the recorder’s previous state. Second we condition on this encoding to draw a sample from a normal. Finally we use that sample to act with the decoder. The loss is the same as a variational auto-encoder. 

The attention mechanism is interesting. They use a similar formulation to NTM, but use 2D Gaussian Filters. 

The results are okay, and have some impressive pictures of focusing in on relevant features of images (3.3% error on cluttered MNIST). But the generation is not as good for images as other stuff out there.
