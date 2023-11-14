<h1 align="center">Multi-scale feature decoupling and similarity distillation for class-incremental defect detection of photovoltaic cells</h1>
<p align="center">
This is the code of distillation losses in mFDSD
</p>
### Base detector and　enviroment
Our method can theoretically be added to any detector, default is Yolov5.　The enviroment of our mFDSD follows that of the choosen detector.

### Introduction of functions in [distill-loss.py](https://github.com/wsjxdy/mFDSD/blob/master/distill-loss.py)
* function:　[mask_feature()](https://github.com/wsjxdy/mFDSD/blob/master/distill-loss.py) is the function to perform MFD
* function: [calculate_pred_distillation_loss()](https://github.com/wsjxdy/mFDSD/blob/master/distill-loss.py) is the function of calculating distillation losses between teacher responses and student responses
* function: [calculate_neck_distillation_loss()](https://github.com/wsjxdy/mFDSD/blob/master/distill-loss.py) is the function of calculating distillation losses between feature maps outputted from the necks
* function: [calculate_back_distillation_loss()](https://github.com/wsjxdy/mFDSD/blob/master/distill-loss.py) is the function of calculating distillation losses between feature maps outputted from the backs
### Readers can apply the file [distill-loss.py](https://github.com/wsjxdy/mFDSD/blob/master/distill-loss.py) to any deep learning-based detectors, so as to perform the class-incremental object detection.
 
