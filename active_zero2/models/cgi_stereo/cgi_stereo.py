# https://github.com/gangweiX/CGI-Stereo/blob/main/models/CGI_Stereo.py

from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from active_zero2.models.cgi_stereo.submodule import *
from active_zero2.utils.reprojection import compute_reproj_loss_patch
import math
import gc
import time
import timm

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        try:
            self.act1 = model.act1 # timm 0.5.4
        except AttributeError:
            self.act1 = model.bn1.act # latest timm

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class Context_Geometry_Fusion(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(Context_Geometry_Fusion, self).__init__()

        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.att = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                                 padding=(0,2,2), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,5,5),
                             padding=(0,2,2), stride=1, dilation=1)

        self.weight_init()

    def forward(self, cv, feat):
        '''
        '''
        feat = self.semantic(feat).unsqueeze(2)
        att = self.att(feat+cv)
        cv = torch.sigmoid(att)*feat + cv
        cv = self.agg(cv)
        return cv


class hourglass_fusion(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))


        self.CGF_32 = Context_Geometry_Fusion(in_channels*6, 160)
        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3 = self.CGF_32(conv3, imgs[3])
        conv3_up = self.conv3_up(conv3) # [B, C, D//4//4, H, W]
        if conv3_up.shape[-2:] != conv2.shape[-2:]:
            conv3_up = F.interpolate(conv3_up, size=conv2.shape[-3:], mode='trilinear', align_corners=True)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)
        if conv2_up.shape[-2:] != conv1.shape[-2:]:
            conv2_up = F.interpolate(conv2_up, size=conv1.shape[-3:], mode='trilinear', align_corners=True)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.CGF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)

        return conv
    
class normal_predictor(nn.Module):
    def __init__(self, in_channels, n_disp):
        super(normal_predictor, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))    

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.CGF_16 = Context_Geometry_Fusion(in_channels*4, 192)
        self.CGF_8 = Context_Geometry_Fusion(in_channels*2, 64)
        
        self.normal_head = nn.Sequential(
            BasicConv(in_channels * n_disp, in_channels * n_disp // 4, 
                      deconv=True, is_3d=False, kernel_size=4, padding=1, stride=2),
            nn.ConvTranspose2d(in_channels * n_disp // 4, 3, kernel_size=4, padding=1, stride=2, bias=False)            
        )

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv2 = self.CGF_16(conv2, imgs[2])
        conv2_up = self.conv2_up(conv2)
        if conv2_up.shape[-2:] != conv1.shape[-2:]:
            conv2_up = F.interpolate(conv2_up, size=conv1.shape[-3:], mode='trilinear', align_corners=True)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.CGF_8(conv1, imgs[1])
        conv = self.conv1_up(conv1)

        B, C, D, H_div4, W_div4 = conv.shape
        conv = conv.reshape(B, C*D, H_div4, W_div4)
        
        normal = self.normal_head(conv) # [B, 3, H, W]
        normal = normal / (normal.norm(dim=1, keepdim=True) + 1e-8)
        return normal


class CGI_Stereo(nn.Module):
    def __init__(self, maxdisp, disparity_mode='regular',
                 loglinear_disp_min_depth=0.04, loglinear_disp_max_depth=3.0, loglinear_disp_c=0.01,
                 predict_normal=False):
        super(CGI_Stereo, self).__init__()
        self.maxdisp = maxdisp 
        self.disparity_mode = disparity_mode
        assert self.disparity_mode in ['regular', 'log_linear']
        if self.disparity_mode == 'log_linear':
            self.loglinear_disp_min_depth = loglinear_disp_min_depth
            self.loglinear_disp_max_depth = loglinear_disp_max_depth
            self.loglinear_disp_c = loglinear_disp_c
            # self.loglinear_prioritize_near_factor = 2.5 # prioritize losses where the camera is near the object

        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.hourglass_fusion = hourglass_fusion(8)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
        
        self.predict_normal = predict_normal
        if self.predict_normal:
            self.normal_predictor = normal_predictor(8, self.maxdisp // 4)
        else:
            self.normal_predictor = None

    def forward(self, data_batch, predict_normal=None):
        if predict_normal is None:
            predict_normal = self.predict_normal
        
        left, right = data_batch['img_l'], data_batch['img_r']
        if left.shape[1] == 1:
            left = left.tile((1,3,1,1))
        if right.shape[1] == 1:
            right = right.tile((1,3,1,1))
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)


        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        if self.disparity_mode == 'regular':
            corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        else:
            corr_volume = build_loglinear_correlation_volume(
                match_left, match_right, self.maxdisp//4, 4,
                data_batch['focal_length'], data_batch['baseline'], 
                self.loglinear_disp_min_depth, self.loglinear_disp_max_depth, self.loglinear_disp_c
            )
        corr_volume = self.corr_stem(corr_volume)
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)
        volume = self.agg(feat_volume * corr_volume)
        cost = self.hourglass_fusion(volume, features_left)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)
        pred_up = context_upsample(pred, spx_pred)

        pred_dict = {
            "left_div4_feats": features_left[0], # [bs, C, H/4, W/4]
            "right_div4_feats": features_right[0], # [bs, C, H/4, W/4]
            "cost_prob": F.interpolate(F.softmax(cost.squeeze(1), dim=1), size=pred_up.shape[-2:], mode="nearest"), # [bs, maxdisp/4, H, W]
            "pred_orig": pred_up * 4, # [bs, H, W]
            "pred_div4": pred.squeeze(1) * 4, # [bs, H/4, W/4]
        }
        
        if predict_normal:
            pred_dict["normal"] = self.normal_predictor(volume, features_left)
        
        return pred_dict
        
    def to_processed_disparity(self, raw_disp, focal_length, baseline):
        # raw_disp: [B, H, W], focal_length: [B], baseline: [B]
        assert len(raw_disp.shape) == 3
        depth = (focal_length * baseline)[:, None, None] / (raw_disp + 1e-8)
        disp_t = (
                (torch.log(depth + self.loglinear_disp_c) - np.log(self.loglinear_disp_min_depth + self.loglinear_disp_c))
                / (np.log(self.loglinear_disp_max_depth + self.loglinear_disp_c) - np.log(self.loglinear_disp_min_depth + self.loglinear_disp_c))
        ) # a valid range of disp_t is [1/maxdisp, 1]
        return self.maxdisp - self.maxdisp * disp_t

    def to_raw_disparity(self, processed_disp, focal_length, baseline):
        # processed_disp: [B, H, W], focal_length: [B], baseline: [B]
        assert len(processed_disp.shape) == 3
        disp_t = (self.maxdisp - processed_disp) / self.maxdisp
        depth = (
            ((self.loglinear_disp_min_depth + self.loglinear_disp_c) ** (1 - disp_t)) 
            * ((self.loglinear_disp_max_depth + self.loglinear_disp_c) ** disp_t) 
            - self.loglinear_disp_c
        )
        return (focal_length * baseline)[:, None, None] / (depth + 1e-8)
        

    def compute_disp_loss(self, data_batch, pred_dict):
        disp_gt = data_batch["img_disp_l"]
        disp_gt_div4 = F.interpolate(disp_gt, size=pred_dict["pred_div4"].shape[-2:], mode="nearest")
        disp_gt = disp_gt.squeeze(1) # [B, H, W]
        disp_gt_div4 = disp_gt_div4.squeeze(1) # [B, H//4, W//4]
        if self.disparity_mode == "log_linear":
            disp_gt = self.to_processed_disparity(disp_gt, data_batch["focal_length"], data_batch["baseline"])
            disp_gt_div4 = self.to_processed_disparity(disp_gt_div4, data_batch["focal_length"], data_batch["baseline"])

        pred_orig = pred_dict['pred_orig']
        pred_div4 = pred_dict['pred_div4']
        # Get stereo loss on sim
        # Note in training we do not exclude bg
        if self.disparity_mode == 'regular':
            mask = (disp_gt < self.maxdisp) * (disp_gt > 0)
        else:
            mask = (disp_gt <= self.maxdisp - 1) * (disp_gt >= 0)
        mask.detach()
        if self.disparity_mode == 'regular':
            mask_div4 = (disp_gt_div4 < self.maxdisp) * (disp_gt_div4 > 0)
        else:
            mask_div4 = (disp_gt_div4 <= self.maxdisp - 1) * (disp_gt_div4 >= 0)
        mask_div4.detach()
        
        """
        if self.disparity_mode == "log_linear":
            mul_orig = torch.ones_like(disp_gt)
            mul_orig[disp_gt > self.maxdisp / 2] = self.loglinear_prioritize_near_factor
            mul_div4 = torch.ones_like(disp_gt_div4)
            mul_div4[disp_gt_div4 > self.maxdisp / 2] = self.loglinear_prioritize_near_factor
            pred_orig, disp_gt = pred_orig * mul_orig, disp_gt * mul_orig
            pred_div4, disp_gt_div4 = pred_div4 * mul_div4, disp_gt_div4 * mul_div4
        """

        loss_disp = 0.0
        loss_disp += 1.0 * F.smooth_l1_loss(pred_orig[mask], disp_gt[mask], reduction="mean")
        loss_disp += 0.3 * F.smooth_l1_loss(pred_div4[mask_div4], disp_gt_div4[mask_div4], reduction="mean")

        return loss_disp

    def compute_reproj_loss(self, data_batch, pred_dict, use_mask: bool, patch_size: int, only_last_pred: bool):
        if use_mask:
            disp_gt = data_batch["img_disp_l"] # [B, 1, H, W]
            # Get stereo loss on sim
            # Note in training we do not exclude bg
            if self.disparity_mode == 'regular':
                mask = (disp_gt < self.maxdisp) * (disp_gt > 0)
            else:
                disp_gt_processed = self.to_processed_disparity(disp_gt.squeeze(1), data_batch["focal_length"], data_batch["baseline"])[:, None ,:, :]
                mask = (disp_gt_processed <= self.maxdisp - 1) * (disp_gt_processed >= 0)
            mask.detach()
        else:
            mask = None
        

        if only_last_pred:
            pred_disp_l = pred_dict["pred_orig"] # [B, H, W]
            if self.disparity_mode == 'log_linear':
                pred_disp_l = self.to_raw_disparity(pred_disp_l, data_batch["focal_length"], data_batch["baseline"])
            pred_disp_l = pred_disp_l[:, None, :, :]
            """
            # filter out pixels close to the camera when calculating the reprojection loss
            mask_dist_filter = ((data_batch["focal_length"] * data_batch["baseline"])[:, None, None] / (pred_disp_l + 1e-8)) > 0.3
            pred_disp_l = pred_disp_l[:, None, :, :]
            mask_dist_filter = mask_dist_filter[:, None, :, :]
            if mask is None:
                mask = mask_dist_filter
            else:
                mask = mask * mask_dist_filter
            """
            loss_reproj = compute_reproj_loss_patch(
                data_batch["img_pattern_l"],
                data_batch["img_pattern_r"],
                pred_disp_l=pred_disp_l,
                mask=mask,
                ps=patch_size,
            )

            return loss_reproj
        else:
            loss_reproj = 0.0
            for pred_name, loss_weight in zip(["pred_div4", "pred_orig"], [0.5, 1.0]):
                if pred_name in pred_dict:
                    pred_disp_l = pred_dict[pred_name]
                    if self.disparity_mode == 'log_linear':
                        pred_disp_l = self.to_raw_disparity(pred_disp_l, data_batch["focal_length"], data_batch["baseline"])
                    pred_disp_l = pred_disp_l[:, None, :, :]
                    """
                    # filter out pixels close to the camera when calculating the reprojection loss
                    mask_dist_filter = ((data_batch["focal_length"] * data_batch["baseline"])[:, None, None] / (pred_disp_l + 1e-8)) > 0.3
                    pred_disp_l = pred_disp_l[:, None, :, :]
                    mask_dist_filter = mask_dist_filter[:, None, :, :]
                    if mask is None:
                        mask = mask_dist_filter
                    else:
                        mask = mask * mask_dist_filter
                    """
                    loss_reproj += loss_weight * compute_reproj_loss_patch(
                        data_batch["img_pattern_l"],
                        data_batch["img_pattern_r"],
                        pred_disp_l=pred_disp_l,
                        mask=mask,
                        ps=patch_size,
                    )
            return loss_reproj

    def compute_normal_loss(self, data_batch, pred_dict):
        normal_gt = data_batch["img_normal_l"]
        normal_pred = pred_dict["normal"] # [B, 3, H, W]
        cos = F.cosine_similarity(normal_gt, normal_pred, dim=1, eps=1e-6)
        loss_cos = 1.0 - cos
        if "img_disp_l" in data_batch:
            disp_gt = data_batch["img_disp_l"]
            if self.disparity_mode == 'regular':
                mask = (disp_gt < self.maxdisp) * (disp_gt > 0)
            else:
                disp_gt_processed = self.to_processed_disparity(disp_gt.squeeze(1), data_batch["focal_length"], data_batch["baseline"])[:, None, :, :]
                mask = (disp_gt_processed <= self.maxdisp - 1) * (disp_gt_processed >= 0)
            mask = mask.squeeze(1) # [B, H, W]
            mask.detach()
        else:
            mask = torch.ones_like(loss_cos)

        if "img_normal_weight" in data_batch:
            img_normal_weight = data_batch["img_normal_weight"]  # (B, H, W)
            loss_cos = (loss_cos * img_normal_weight * mask).sum() / ((img_normal_weight * mask).sum() + 1e-8)
        else:
            loss_cos = (loss_cos * mask).sum() / (mask.sum() + 1e-8)
        return loss_cos
