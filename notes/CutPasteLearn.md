A deceptively simple CMU paper for training data greedy methods in a small data regime. The paper is incredibly simple. The idea is to:

1. Collect object instances
2. Collect scene images
3. Segment the object instances
4. Paste the objects into the scene
5. Blend the areas around the object borders

This simple procedure forgets about global consistency (where there is a great deal of work happening), but has really good results.
