Another deep mind paper (based heavily on this [one](https://github.com/knathanieltucker/deeplearning-papernotes/blob/master/notes/Recurrent_Models_of_Visual_Attention.md)). Though there might be a little bit of a better use for this one.

The idea is simple, we want to direct a read head over an image and find all instances in the image. 

The architecture is simple. We feed an image patch (where we currently are) and location into small CNN and FC network. We multiply the high frequency and low frequency vectors together (sigh….) and we feed that into an RNN, let’s call this guy RNN1. After we take N steps we use the output from RNN1 to classify an instance in the image. But before that we are using it as an input to RNN2 whose job it is to determine the next location. I have left out some details, but for the most part I wanted to impress on you that nothing out of the ordinary is being done.

They do some optimization to improve performance and actually run this one over some important data sets (recognizing signs). This is a bit more promising than the last paper but leaves something to be desired. The entire formulation seems contrived… But more on that later.
