A good old NYU LeCun paper. I won’t go into too much detail here but the main idea is to view the discriminator in a GAN as an energy function and the generator a function that tries to achieve low energy. They prove that this formulation can lead to Nash equilibrium between the generator and the discriminator that result in equal data and generated data distributions. And the coolest part is to formulate the discriminator as an Auto-Encoder (where [Dec(Enc(x)) - x] is the energy. 

They then show that this formulation is more stable etc.

Kind of cool… But that was about it…
