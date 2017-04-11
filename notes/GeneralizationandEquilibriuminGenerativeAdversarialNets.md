An interesting if not on the cusp of being too theoretical paper from Princeton. This one goes over a motley of GAN properties and questions that have not been addressed thus far.

The first of these is to show that using the given GAN objective functions, small discriminators can only do so much, and it is generally better for them to memorize the empirical distribution rather than generalizing. They then introduce a new distance metric for the objective function that will lead to better generalization (and prove it). It is called the neural network divergence. 

Next they prove that there is a stable equilibrium in certain GAN cases. If the discriminator is a deep net with n params then a generator with n log n params will be able to fool it 1 - epsilon times to 1. 

And finally they add some empirical results and show that using a mixture of generators and discriminators can lead to more stable training.

Another good bag of tricks to add to the old GAN training book.
