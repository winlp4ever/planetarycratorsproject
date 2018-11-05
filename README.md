### MC Detection

#### Approaches taken

* Use Deep Net + Transfer Learning: As indicated in the guide notebook, 3 promising 
network architectures are:
1. Faster RCNN (read the paper [here]())
2. SSD (Single Shot MultiBox Detector, [paper](https://arxiv.org/abs/1512.02325))
3. YOLO (You Only Look Once, [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf))

Among these three, _Google_ has actually implemented the first two archis and train them with
different popular datasets (COCO and ImageNet). More info should be read at [Google Object
Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
So, the idea is taking advantages of these pretrained model (as these are trained with a large 
amount of data and for diversified classes of objects, they would clearly contain
more information than networks trained for limited dataset and a sole class like our case)
 by creating our model on top of these, fine-tuning it with our data (which is basically the
 idea of transfer learning).
 
 My work so far follows what proposed in this [tutorial](https://becominghuman.ai/tensorflow-object-detection-api-tutorial-training-and-evaluating-custom-object-detector-ed2594afcf73)
 with some changes to adapt to our problem's requisites.
#### What I have done so far:

* Transforming original data (`npy` format) to image data (`jpeg` format) and then
to _tensorflow_ data format (`tfrecord`). 

* Fine-tuning `ssd-mobinet-v1-coco` and `faster-rcnn-resnet101-coco` with our data.
The training process hasn't finished yet.

#### Difficulties to be discussed

1. Google's trained models are for _full-color_ dataset, ours is _grey_.
2. Google use _bounding rectangles_ in their models, our problem asks for _circle-bounding_.

More questions should be added to this list the more we proceed into the project.

For now, I solved the first one by duplicating values to all three channels (_RGB_)
and the second one by taking the tangential squares of the circles.
