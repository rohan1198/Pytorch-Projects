<p align="center"><b><ins> YOLOv1 </ins></b></p>

- In the paper [You Only Look Once: Unified, Real-Time Object Detection (2016)](https://arxiv.org/pdf/1506.02640.pdf), the authors propose a novel approach for object detection.
- The authors frame object detection as a <ins> regression problem to spatially separated bounding boxes and associated class probabilities </ins>.
- A single neural network predicts bounding boxes and class probabilities directly from entire images in one evaluation.

<br>

---

<br>

<b><ins> Introduction </ins></b>

- The proposed object detection methods at the time repurposed classifiers to perform detectionby evaluating the classifier at various locations and scales.
- Other popular methods included sliding window approaches, region proposal networks.
- These pipelines are complex, slow, and hard to optimize, as each component has to be trained separately.

<br>

- The proposed architecture, YOLO, uses a single convolutional neural network that predicts multiple bounding boxes and class probabilities for the detected boxes.
- This simplification in the pipeline, as well as the different approach make YOLO fast.
- Unlike the window-based approaches YOLO also reasons globally about the image when making predictions, as it can see the larger context.
- YOLO also learns generalizable represemtations of objects.

<br>

---

<br>

<b><ins> Unified Detection </ins></b>

- The proposed network uses features from the entire image to predict multiple bounding boxes.
- The image is first divided into an  $S$ x $S$ grid.
- The grid is responsible for detecting the object if it lies in the center of the grid.
- Each grid cell $B$ predicts bounding boxes and confidence scores for the boxes.
- Each bounding box consists of 5 predictions: $x$, $y$, $w$, $h$ and $confidence$.
- The <ins>confidence score is defined as the intersection over union between the predicted box and the ground truth </ins>.
- The <ins>width and height are predicted relative to the whole image</ins>, while the <ins>(x, y) coordinates represent the center of the box relative to the bounds of the grid cell</ins>.

<br>

- Each grid cell also predicts conditional class probabilities, $Pr(Class_{i} | Object)$.
- Only one set of probabilities are predicted per cell, regardless of the number of boxes B.
- During testing, the conditional class probabilities are are multiplied with the individual box confidence scores to give the class-specific score for each box:

$$
Pr(Class_{i} | Object) * Pr(Object) * IoU^{truth}_{pred} = Pr(Class_{i}) * IoU^{truth}_{pred}
$$

- These scores encode the probability of the class appearing in the box, and how well the prediction fits the ground truth.

<br>

---

<br>

<b><ins> Network Design </ins></b>

- The initial layers of the CNN extract features from the image, while the fully-connected layers predict the output probabilities and coordinates.
- The architecture was inspired by GoogleNet, and has 24 convolutional layers followed by 2 fully connected layers.
- The authors also propose using a 1x1 convolution followed by a 3x3 convolution instead of using the inception modules in GoogleNet.
- The final output is the $7$x$7$x$30$ tensor of predictions.

<br>

---

<br>

<b><ins> Training </ins></b>

- The convolutional layers are pretrained on the ImageNet-1000 dataset.
- A combination of the first 20 convolutional layers followed by an average pooling layer and a fully connected layer is used for this
- The leaky-ReLU activation function is used for all layers except the final layer.

<br>

- The model is optimized for the sum-squared error in the output, but this does not align with the goal of <ins>maximizing average precision</ins>.
- And since many grid cells do not contain any objects, the scores of those get pushed to zero, often affecting the gradient, and further leading to model instability.
- <ins>To remedy this, the authors increase the loss from bounding box coordinate predictions, and decrease the loss from confidence predictions for boxes that don't contain any objects</ins>.

<br>

---

<br>

<b><ins> Loss Function </ins></b>

- The following multi-part sum-squarred loss function is optimized during training:

$$
    \displaylines{
    \lambda_{coord} \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} 1^{obj}_{i j}[(x_{i} - \hat{x_{i}})^2 + (y_{i} - \hat{y_{i}})^2] \\
    + \lambda_{coord} \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} 1^{obj}_{i j} [(\sqrt{w_{i}} - \sqrt{\hat{w_{i}}})^2 + (\sqrt{h_{i}} - \sqrt{\hat{h_{i}}})^2] \\
    + \lambda_{coord} \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} 1^{obj}_{i j}(c_{i} - \hat{c_{i}})^{2} \\
    + \lambda_{noobj} \sum_{i = 0}^{S^{2}} \sum_{j = 0}^{B} 1^{obj}_{i j}(c_{i} - \hat{c_{i}})^{2} \\
    + \sum_{i = 0}^{S^{2}} 1^{obj}_{i} \sum_{classes} (p_{i}(c) - \hat{p_{i}}(c))^{2}
    }
$$

- Here, $1^{obj}_{i}$ denotes is object appears in cell $i$ and $1^{obj}_{ij}$ denotes the $j^{th}$ bounding box predictor in cell $i$ is responsible for that prediction.
- Line 1 shows the sum-squared error of the ground truth midpoint and the predicted midpoint.
- Line 2 shows the sum-squared error of the ground truth (width and height) and the predicted (width and height). An additional square root operation is performed on the height and width to normalize the difference between the losses of very tall/wide bounding boxes (basically, taking a square of a tall bounding box will lead to a larger loss as compared to a small bounding box).
- Line 3 shows the sum-squared error of the ground truth and prediction of the probability that an object is present in a given cell.
- Line 4 shows the sum-squared error if there is no object present in the cell.
- Line 5 shows the sum-squared error of the ground truth and prediction of the predicted class in each cell.

<br>

---

<br>

References

- https://arxiv.org/pdf/1506.02640.pdf
- https://www.youtube.com/watch?v=n9_XyCGr-MI
- https://pyimagesearch.com/2022/04/11/understanding-a-real-time-object-detection-network-you-only-look-once-yolov1/
- https://universe.roboflow.com/jacob-solawetz/pascal-voc-2012/dataset/1
