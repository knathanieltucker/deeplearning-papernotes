A deep mind Graves one. Fun fun. This one reminds me a ton of target prop. The basic idea is to do backprop without hearing from the layers above you (as in you don’t know what the error was). 

There is no magic here. All they do is have small networks estimate the utility of your activation (as in have a little network estimate what the error was from the upper layers) and then as you slowly get the true errors trickling down from the top, update the little network. 

The interesting thing here is the application to RNNs where you are able to estimate really long timescales. They get okay results, I don’t imagine that we are going to be using this technique anytime soon.
