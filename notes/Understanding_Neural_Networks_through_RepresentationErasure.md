The idea here is to delete dimensions of word vectors in trained models (eg. POS tagging and sentiment). We can then measure how important a dimension is by averaging the difference in the log likelihood of deleting over all examples. The word vectors used were word2vec and GloVe and for each task a 4 layer neural model was trained.

They then looked at something really interesting, erasing entire words from sentiment analysis to find the most important words (or words that often mess the detection up!). And for one step forward they find the minimum number of words to mislabel a sentence.
