this is the code of distillation losses in mFDSD
# Our method can theoretically be added to any detector, default is Yolov5.
# The enviroment of our mFDSD follows that of the choosen detector.
\function: mask_feature() is the function to perform MFD
\function: calculate_pred_distillation_loss() is the function of calculating distillation losses between teacher responses and student responses
\function: calculate_neck_distillation_loss() is the function of calculating distillation losses between feature maps outputted from the necks
\function: calculate_back_distillation_loss() is the function of calculating distillation losses between feature maps outputted from the backs
