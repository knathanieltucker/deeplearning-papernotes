So we are trying to reconstruct 3D images from line drawings etc. We do this using a simple 6 step procedure.

1. Find Keypoints across scale
2. Connect proximate keypoints
3. Select candidates by local appearance
4. Computed compatibility between local layers
5. Infer most likely patches
6. Fit surface to patches

Now let’s go over each step. Finding the key points is by computing an interest map (sum of the eigenvalues of the 2x2 structure of the test image) and then dampening around selected points. Connections are done based on hyper parameter based scale and location proximity. Shape candidates are selected in two steps:

1. Get a large number of candidates that are close in euclidian distance from the test patch
2. PCA them and find a diverse set of the candidates above (maximum minimum distance)

The scoring is somewhat complex. First they take into account occluding contours (those should not be proximate to the surfaces they are occluding) using image segmentation (edge detection, flood filling, etc.). Then they find depth and normal compatibility of overlapping patches. The final step in scoring is putting a prior on patches (high probability to those that occur frequently). 

They train (memorize patches) and test on synthetic images and that is about that!
