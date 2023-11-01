# distillation loss of mFDSD

import torch
import torch.nn as nn
import torch.nn.functional as F

class Compute_distill_loss:
    def __init__(self, device):
        self.device = device
        self.gamma1 = torch.ones(1, device=self.device) * 0.5
        self.gamma2 = 1 - self.gamma1

    def mask_feature(self, stu_out_list=None, tea_out_list=None, mask=None):
        # stu_out_list: the list of feature maps from backbone and neck of the student detector
        # tea_out_list: the list of feature maps from backbone and neck of the teacher detector
        # mask: GT labels (format xyxy)
        device = stu_out_list[4].device  # device

        # take out the feature map that needs to be distilled; here (YOLOV5) we chooose 4, 6, 10 of backbone, and 17,20,23 of neck
        teacher_output = [tea_out_list[4], tea_out_list[6], tea_out_list[10], tea_out_list[17],
                          tea_out_list[20], tea_out_list[23]]  # 80,40,20
        student_output = [stu_out_list[4], stu_out_list[6], stu_out_list[10], stu_out_list[17],
                          stu_out_list[20], stu_out_list[23]]

        # generate mask of GT labels (new categories) and perform MFD
        if mask is not None:
            mask_large = torch.ones([tea_out_list[17].shape[0], 1, tea_out_list[17].shape[2],
                                     tea_out_list[17].shape[3]], device=device)
            mask_middle = torch.ones([tea_out_list[20].shape[0], 1, tea_out_list[20].shape[2],
                                      tea_out_list[20].shape[3]], device=device)
            mask_little = torch.ones([tea_out_list[23].shape[0], 1, tea_out_list[23].shape[2],
                                      tea_out_list[23].shape[3]], device=device)

            for i in range(mask.shape[0]):
                #  begin generate multi-scale MASKs about new categories
                mask_now = mask[i, :]
                label = int(mask_now[0].item())  # label ： xmin ymin xmax ymax
                large_H_max = (mask_now[5] // 8) + 1 if (mask_now[5] // 8) + 1 < 80 else mask_now[5] // 8
                large_H_min = mask_now[3] // 8
                large_W_max = (mask_now[4] // 8) + 1 if (mask_now[4] // 8) + 1 < 80 else mask_now[4] // 8
                large_W_min = mask_now[2] // 8

                middle_H_max = (mask_now[5] // 16) + 1 if (mask_now[5] // 16) + 1 < 40 else mask_now[5] // 16
                middle_H_min = mask_now[3] // 16
                middle_W_max = (mask_now[4] // 16) + 1 if (mask_now[4] // 16) + 1 < 40 else mask_now[4] // 16
                middle_W_min = mask_now[2] // 16

                little_H_max = (mask_now[5] // 32) + 1 if (mask_now[5] // 32) + 1 < 20 else mask_now[5] // 32
                little_H_min = mask_now[3] // 32
                little_W_max = (mask_now[4] // 32) + 1 if (mask_now[4] // 32) + 1 < 20 else mask_now[4] // 32
                little_W_min = mask_now[2] // 32
                large_H_min = int(large_H_min.item())
                large_H_max = int(large_H_max.item())
                large_W_min = int(large_W_min.item())
                large_W_max = int(large_W_max.item())
                middle_H_min = int(middle_H_min.item())
                middle_H_max = int(middle_H_max.item())
                middle_W_min = int(middle_W_min.item())
                middle_W_max = int(middle_W_max.item())
                little_H_min = int(little_H_min.item())
                little_H_max = int(little_H_max.item())
                little_W_min = int(little_W_min.item())
                little_W_max = int(little_W_max.item())
                mask_large[label, :, large_H_min:large_H_max + 1, large_W_min:large_W_max + 1] = 0
                mask_middle[label, :, middle_H_min:middle_H_max + 1, middle_W_min:middle_W_max + 1] = 0
                mask_little[label, :, little_H_min:little_H_max + 1, little_W_min:little_W_max + 1] = 0
                # end generate multi-scale MASKs about new categories

            # begin MFD
            teacher_output[0] = torch.mul(teacher_output[0], mask_large)
            student_output[0] = torch.mul(student_output[0], mask_large)
            teacher_output[1] = torch.mul(teacher_output[1], mask_middle)
            student_output[1] = torch.mul(student_output[1], mask_middle)
            teacher_output[2] = torch.mul(teacher_output[2], mask_little)
            student_output[2] = torch.mul(student_output[2], mask_little)
            teacher_output[3] = torch.mul(teacher_output[3], mask_large)
            student_output[3] = torch.mul(student_output[3], mask_large)
            teacher_output[4] = torch.mul(teacher_output[4], mask_middle)
            student_output[4] = torch.mul(student_output[4], mask_middle)
            teacher_output[5] = torch.mul(teacher_output[5], mask_little)
            student_output[5] = torch.mul(student_output[5], mask_little)
            del stu_out_list
            del tea_out_list
            torch.cuda.empty_cache()
        return teacher_output, student_output  # output masked teacher output and student output

    def calculate_pred_distillation_loss(self, stu_list, tea_list, old_classes, new_classes):
        # stu_list: predictions from the prediction heads of the student detector
        # tea_list: predictions from the prediction heads of the teacher detector
        # old_classes: the list of old classes
        # new_classes: the list of new classes added in current task
        device = self.device
        final_pre_dis_loss = torch.zeros(1, device=device)

        # To split box, obj, cls information in stu_list
        stu_pre_obj_list = []
        stu_pre_box_list = []
        stu_pre_cls_list = []
        for cnt in range(len(stu_list)):
            stu_pre_box_list.append(stu_list[cnt][:, :, :, :, 0:4])  # box
            stu_pre_obj_list.append(stu_list[cnt][:, :, :, :, 4].unsqueeze(-1))  # obj
            stu_pre_cls_list.append(stu_list[cnt][:, :, :, :, 5:])  # cls

        # To split box, obj, cls information in tea_list
        tea_pre_box_list = []
        tea_pre_obj_list = []
        tea_pre_cls_list = []
        for cnt in range(len(tea_list)):
            tea_pre_box_list.append(tea_list[cnt][:, :, :, :, 0:4])
            tea_pre_obj_list.append(tea_list[cnt][:, :, :, :, 4].unsqueeze(-1))
            tea_pre_cls_list.append(tea_list[cnt][:, :, :, :, 5:])

        multi_scale_pro_distillation_loss = []  # init distillation loss of pro
        multi_scale_box_distillation_loss = []  # init distillation loss of box
        multi_scale_cls_distillation_loss = []  # init distillation loss of cls

        for i in range(0, len(tea_pre_obj_list)):
            def mFDSD_model():
                scale_teacher_obj = tea_pre_obj_list[i]
                scale_student_obj = stu_pre_obj_list[i]
                single_scale_obj_difference = scale_teacher_obj - scale_student_obj
                obj_mask = torch.gt(single_scale_obj_difference,
                                    torch.zeros(single_scale_obj_difference.shape).to(device)) * 1
                single_scale_obj_difference = single_scale_obj_difference * obj_mask
                obj_distillation_loss = torch.mul(single_scale_obj_difference, single_scale_obj_difference)
                obj_distillation_loss = torch.mean(obj_distillation_loss)  # dim=0 
                multi_scale_pro_distillation_loss.append(obj_distillation_loss)
                return obj_mask

            obj_mask = mFDSD_model()  # obtain mask of responses that needs to be distilled

            unzero_mask = torch.gt(obj_mask, torch.zeros(obj_mask.shape).to(device)) * 1 + 1e-6

            # distilation of cls
            num_category_need_dis = len(old_classes) + len(new_classes)
            singer_scale_current_teacher_cls = tea_pre_cls_list[i] * unzero_mask
            singer_scale_current_student_cls = stu_pre_cls_list[i] * unzero_mask
            singer_scale_current_teacher_cls = singer_scale_current_teacher_cls[:, :, :, :, :num_category_need_dis]
            singer_scale_current_student_cls = singer_scale_current_student_cls[:, :, :, :, :num_category_need_dis]

            # 原版cls损失
            mean_of_teacher_cls = torch.mean(singer_scale_current_teacher_cls, dim=-1, keepdim=True)
            mean_of_student_cls = torch.mean(singer_scale_current_student_cls, dim=-1, keepdim=True)
            normalized_teacher_cls = torch.sub(singer_scale_current_teacher_cls, mean_of_teacher_cls)
            normalized_student_cls = torch.sub(singer_scale_current_student_cls, mean_of_student_cls)
            cls_l2_loss = nn.MSELoss(size_average=False, reduce=False)
            cls_distillation_loss = cls_l2_loss(normalized_teacher_cls, normalized_student_cls) * obj_mask
            cls_distillation_loss = torch.mean(cls_distillation_loss)

            multi_scale_cls_distillation_loss.append(cls_distillation_loss)

            # distilation of box
            singer_scale_current_teacher_box = tea_pre_box_list[i] * unzero_mask
            singer_scale_current_student_box = stu_pre_box_list[i] * unzero_mask
            box_l2_loss = nn.MSELoss(size_average=False, reduce=False)
            bbox_distillation_loss = box_l2_loss(singer_scale_current_teacher_box,
                                                 singer_scale_current_student_box) * obj_mask
            bbox_distillation_loss = torch.mean(bbox_distillation_loss)
            multi_scale_box_distillation_loss.append(bbox_distillation_loss)

        final_pro_distillation_loss = sum(multi_scale_pro_distillation_loss) if len(stu_pre_obj_list) == 0 else sum(
            multi_scale_pro_distillation_loss) / len(stu_pre_obj_list)
        final_box_distillation_loss = sum(multi_scale_box_distillation_loss) if len(stu_pre_obj_list) == 0 else sum(
            multi_scale_box_distillation_loss) / len(stu_pre_obj_list)  ##
        final_cls_distillation_loss = sum(multi_scale_cls_distillation_loss) if len(stu_pre_obj_list) == 0 else sum(
            multi_scale_cls_distillation_loss) / (len(stu_pre_obj_list) * num_category_need_dis)
        final_pre_dis_loss += final_pro_distillation_loss + final_box_distillation_loss + final_cls_distillation_loss
        del stu_list
        del tea_list
        torch.cuda.empty_cache()

        return final_pre_dis_loss  # return distillation losses between predictions from head

    def calculate_neck_distillation_loss(self, teacher_output, student_output, stu_out_list, tea_out_list, mask=None):
        # teacher_output: the list of feature maps from backbone and neck of the teacher detector (output of function "mask_feature()")
        # student_output: the list of feature maps from backbone and neck of the student detector (output of function "mask_feature()")
        # stu_out_list: the list of feature maps from backbone and neck of the student detector (not the output of function "mask_feature()")
        # tea_out_list: the list of feature maps from backbone and neck of the teacher detector (not the output of function "mask_feature()")
        # mask: GT labels (format xyxy)
        # dis_device: cos
        # dis_attention
        device = self.device
        if mask is None:  # mask is None represents only have SFD module, no MFD module
            del teacher_output
            del student_output
            torch.cuda.empty_cache()

        final_neck_distillation_loss = torch.zeros(1, device=device)
        teacher_neck_output_current = [tea_out_list[17], tea_out_list[20], tea_out_list[23]]  # 80,40,20
        student_neck_output_current = [stu_out_list[17], stu_out_list[20], stu_out_list[23]]

        mask_cos = []
        spa_attention_list = []
        spa_attention_stu_list = []
        cha_attention_list = []
        cha_attention_stu_list = []

        # decouple old categories and background (SFD)
        mask_cos_cha = []
        mask_cos_spa = []
        for i in range(0, len(teacher_neck_output_current)):
            now_teacher = teacher_neck_output_current[i]
            now_student = student_neck_output_current[i]
            num_batch = now_teacher.shape[0]
            num_channel = now_teacher.shape[1]
            num_hight = now_teacher.shape[2]
            num_weight = now_teacher.shape[3]
            now_teacher_cha = torch.reshape(now_teacher, [num_batch, num_channel, num_hight * num_weight])
            now_student_cha = torch.reshape(now_student, [num_batch, num_channel, num_hight * num_weight])
            now_teacher_spa = now_teacher.permute(0, 2, 3, 1)
            now_teacher_spa = torch.reshape(now_teacher_spa, [num_batch, num_hight * num_weight, num_channel])
            now_student_spa = now_student.permute(0, 2, 3, 1)
            now_student_spa = torch.reshape(now_student_spa, [num_batch, num_hight * num_weight, num_channel])
            mask_cha = F.cosine_similarity(now_teacher_cha, now_student_cha, -1)
            mask_spa = F.cosine_similarity(now_teacher_spa, now_student_spa, -1)
            mask_cha = torch.reshape(mask_cha, [num_batch, num_channel, 1, 1]).expand_as(now_teacher)
            mask_spa = torch.reshape(mask_spa, [num_batch, num_hight, num_weight, 1]).permute(0, 3, 1, 2).expand_as(
                now_student)
            mask_cos_cha.append(mask_cha)
            mask_cos_spa.append(mask_spa)
            del now_teacher
            del now_student
            del now_teacher_spa
            del now_student_spa
            del now_teacher_cha
            del now_student_cha
            del mask_cha
            del mask_spa

            mask_cos.append(mask_cos_cha)
            mask_cos.append(mask_cos_spa)
            del stu_out_list
            del tea_out_list
            torch.cuda.empty_cache()

        if mask is not None:  # represents have MFD module
            # decouple new categories areas an others
            del teacher_neck_output_current
            del student_neck_output_current
            torch.cuda.empty_cache()
            teacher_neck_output = [teacher_output[3], teacher_output[4], teacher_output[5]]
            student_neck_output = [student_output[3], student_output[4], student_output[5]]
            del teacher_output
            del student_output
            torch.cuda.empty_cache()
        else:  # represents have not MFD module
            teacher_neck_output = teacher_neck_output_current
            student_neck_output = student_neck_output_current

        num_scales = len(teacher_neck_output)
        multi_scale_neck_distillation_loss = []  # init distillation loss between feature maps from neck

        # the calculation of distillation losses
        for i in range(0, num_scales):
            current_teacher_neck_output = teacher_neck_output[i]
            current_student_neck_output = student_neck_output[i]

            teacher_neck_feature_avg = torch.mean(current_teacher_neck_output)
            student_neck_feature_avg = torch.mean(current_student_neck_output)

            normalized_teacher_feature = current_teacher_neck_output - teacher_neck_feature_avg  # normalize features
            normalized_student_feature = current_student_neck_output - student_neck_feature_avg


            feature_difference = normalized_teacher_feature - normalized_student_feature
            feature_distillation_loss = torch.abs(feature_difference)
            feature_distillation_loss = torch.mul(torch.mul(feature_distillation_loss, 1 - mask_cos[0][i]),
                                                  1 - mask_cos[1][i])
            multi_scale_neck_distillation_loss.append(torch.mean(feature_distillation_loss))

        del mask_cos
        torch.cuda.empty_cache()
        final_neck_distillation_loss += sum(multi_scale_neck_distillation_loss) / len(multi_scale_neck_distillation_loss)
        return final_neck_distillation_loss  # return distillation losses of feature maps from neck

    def calculate_back_distillation_loss(self, teacher_output, student_output, stu_out_list, tea_out_list, mask=None):
        # teacher_output: the list of feature maps from backbone and neck of the teacher detector (output of function "mask_feature()")
        # student_output: the list of feature maps from backbone and neck of the student detector (output of function "mask_feature()")
        # stu_out_list: the list of feature maps from backbone and neck of the student detector (not the output of function "mask_feature()")
        # tea_out_list: the list of feature maps from backbone and neck of the teacher detector (not the output of function "mask_feature()")
        # mask: GT labels (format xyxy)
        # dis_device: cos
        # dis_attention

        device = stu_out_list[4].device
        if mask is None:
            del teacher_output
            del student_output
        final_back_distillation_loss = torch.zeros(1, device=device)  # generate a tensor, the shape is 1, like this:[0]
        teacher_back_output_current = [tea_out_list[4], tea_out_list[6], tea_out_list[10]]  # 80,40,20
        student_back_output_current = [stu_out_list[4], stu_out_list[6], stu_out_list[10]]
        mask_cos = []
        spa_attention_list = []
        spa_attention_stu_list = []
        cha_attention_list = []
        cha_attention_stu_list = []

        mask_cos_cha = []
        mask_cos_spa = []
        for i in range(0, len(teacher_back_output_current)):
            now_teacher = teacher_back_output_current[i]
            now_student = student_back_output_current[i]
            num_Batch = now_teacher.shape[0]
            num_Channel = now_teacher.shape[1]
            num_Hight = now_teacher.shape[2]
            num_Weight = now_teacher.shape[3]
            now_teacher_cha = torch.reshape(now_teacher, [num_Batch, num_Channel, num_Hight * num_Weight])
            now_student_cha = torch.reshape(now_student, [num_Batch, num_Channel, num_Hight * num_Weight])
            now_teacher_spa = now_teacher.permute(0, 2, 3, 1)
            now_teacher_spa = torch.reshape(now_teacher_spa, [num_Batch, num_Hight * num_Weight, num_Channel])
            now_student_spa = now_student.permute(0, 2, 3, 1)
            now_student_spa = torch.reshape(now_student_spa, [num_Batch, num_Hight * num_Weight, num_Channel])
            mask_cha = F.cosine_similarity(now_teacher_cha, now_student_cha, -1)
            mask_spa = F.cosine_similarity(now_teacher_spa, now_student_spa, -1)
            mask_cha = torch.reshape(mask_cha, [num_Batch, num_Channel, 1, 1]).expand_as(now_teacher)
            mask_spa = torch.reshape(mask_spa, [num_Batch, num_Hight, num_Weight, 1]).permute(0, 3, 1, 2).expand_as(
                now_student)
            mask_cos_cha.append(mask_cha)
            mask_cos_spa.append(mask_spa)
            del mask_cha
            del mask_spa
            del now_teacher
            del now_student
            del now_teacher_spa
            del now_student_spa
            del now_teacher_cha
            del now_student_cha
            torch.cuda.empty_cache()

            mask_cos.append(mask_cos_cha)
            mask_cos.append(mask_cos_spa)
            del stu_out_list
            del tea_out_list
            torch.cuda.empty_cache()

        if mask is not None:
            del teacher_back_output_current
            del student_back_output_current
            torch.cuda.empty_cache()
            teacher_back_output = [teacher_output[0], teacher_output[1], teacher_output[2]]
            student_back_output = [student_output[0], student_output[1], student_output[2]]
            del teacher_output
            del student_output
            torch.cuda.empty_cache()
        else:
            teacher_back_output = teacher_back_output_current
            student_back_output = student_back_output_current
        num_scales = len(teacher_back_output)
        multi_scale_back_distillation_loss = []
        for i in range(0, num_scales):
            current_teacher_back_output = teacher_back_output[i]
            current_student_back_output = student_back_output[i]
            teacher_back_feature_avg = torch.mean(current_teacher_back_output)
            student_back_feature_avg = torch.mean(current_student_back_output)
            normalized_teacher_feature = current_teacher_back_output - teacher_back_feature_avg  # normalize features
            normalized_student_feature = current_student_back_output - student_back_feature_avg

            feature_difference = normalized_teacher_feature - normalized_student_feature
            feature_distillation_loss = torch.abs(feature_difference)
            feature_distillation_loss = torch.mul(torch.mul(feature_distillation_loss, 1 - mask_cos[0][i]),
                                                  1 - mask_cos[1][i])
            multi_scale_back_distillation_loss.append(torch.mean(feature_distillation_loss))  

        del mask_cos
        torch.cuda.empty_cache()

        final_back_distillation_loss += sum(multi_scale_back_distillation_loss) / len(multi_scale_back_distillation_loss)
        return final_back_distillation_loss

