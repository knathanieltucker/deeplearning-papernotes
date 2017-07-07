A great little paper by Fisher Yu. Again this is partly expounding on the dilated convolutions that the lab is famous for. 

We are doing two things at once (and the paper makes and argument that these two must be done in tandem for best performance): (1) scene completion and semantic segmentation. They also make some arguments on why this approach is better than model fitting, but that is for another time. 

In short they encode 3D space with a signed distance function and do voxel-wise semantic segmentation on a 3D room using a 3D dilated CNN. One thing that I really enjoy here is using dilated layers in order to do multi scale context aggregation (again in the first paper). 

The results show that they did quite well and they validate all their steps with concrete tests. Great paper over all.
