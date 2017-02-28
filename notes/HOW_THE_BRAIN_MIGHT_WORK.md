200 pages in incredible brevity.

The thesis hinges on the No Free Lunch Theorem and the Neocortex seemingly in conflict (no free lunch saying that no learning algoritm can be good at everything and the Neocortex being and learning algorithm and being good at everything). So the Neocortex really must have the right inductive bias. 

Dileep tries to replicate what the Neocortex is doing by making Hierachical Temporal Memory (HTM) Networks. These guys try to learn an object’s manifold (how it looks under all sorts of conditions) by watching little movies of the objects (the temporal part). And it learns things in a hierarchy because parameter sharing.

Simply put the model does three things:

1. Memorize patterns (at the bottom level this means pixels and at higher levels this means activations of lower levels
2. Update a Markov graph (which memorized object will likely go to which other one)
3. Extract temporal groups from the graph (which memorized objects go together, and then these are passed to the higher levels

They do some fuzzy matching because of noise and some other tricks but that is how the model works. You can even do belief propagation over the model to make it generative!

They talk about how this model generalizes better than CNNs because of learned pooling etc.

The rest of the paper is about how the model jives with what we know biologically about the Neocortex.
