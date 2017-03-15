Another interesting deep mind Graves paper. This time the idea is to have a sequential decision process of a goal directed agent interacting with a visual environment. Let me explain.

At each time step the agent only has access to a local view of the data via the glimpse sensor. This sensor is centered around some location L and is able to see around that location at different scales. The data from the glimpse sensor (along with the current location L) is then fed into the glimpse network. This network processes both with a set of FC layers and spits out an encoding G. Finally the model architecture runs recurrently over outputs G. At each step spiting out a new location and an action. That’s that.

The network is trained using reinforcement learning (literally the same formulation as this [guy](https://github.com/knathanieltucker/deeplearning-papernotes/blob/master/notes/NEURAL_ARCHITECTURE_SEARCH_WITH_REINFORCEMENT_LEARNING.md)) to classify MNIST digits (getting a respectable performance of 4% error). 

The idea might be cool. I can’t think of any uses….
