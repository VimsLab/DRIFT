from .resnet import resnet50, resnet101
import torch.nn as nn
import torch
from .globalNet import globalNet
# from .refineNet import refineNet
from .segNet import segNet
from .multiNet_for_real import multiNet_for_real

__all__ = ['CPN50', 'CPN101', 'MultiTask']


class CPN(nn.Module):
    def __init__(self, output_shape):
        super(CPN, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        # self.resnet = resnet
        self.globalNet = globalNet(channel_settings, output_shape, 2)

        self.offsetNet = globalNet(channel_settings, output_shape, 2)
        # self.segNet = segNet(channel_settings, output_shape, 1)
        # self.offsetNet = globalNet(channel_settings, output_shape, 4)
        self.multiNet = multiNet_for_real(channel_settings[3], output_shape)
        # self.refineNet = refineNet(channel_settings[-1], output_shape, 2)

    def forward(self, warped_encoded):
        
        #[x4, x3, x2, x1] = warped_encoded # x4,x3,x2,x1 from 

        feature_foward, _, _, _ = self.globalNet(warped_encoded)

        _, _, _, offsets = self.offsetNet(warped_encoded)
        end_short_offset_pred = offsets[3][:,0:2,:,:]
        # import pdb; pdb.set_trace()
        # _, _, _, offsets = self.offsetNet(warped_encoded)

        # control_short_offset_pred = offsets[3][:,0:2,:,:]
        # end_short_offset_pred = offsets[3][:,4:6,:,:]

        # _, _, mask_pred= self.segNet(warped_encoded)
        # mask_pred = mask_pred[3]

        control_area_pred, end_area_pred, mask_pred, control_point_embeding, end_point_embeding = self.multiNet(feature_foward)

        # refine_out = self.refineNet(global_fms)

        # return mask_pred, control_short_offset_pred, end_short_offset_pred, control_area_pred, end_area_pred
        return control_area_pred, end_area_pred, mask_pred, control_point_embeding, end_point_embeding\
                ,end_short_offset_pred

def MultiTask(out_size):
    model = CPN(output_shape=out_size)
    return model

def CPN50(out_size,num_class,pretrained=True):
    res50 = resnet50(pretrained=pretrained)
    model = CPN(res50, output_shape=out_size,num_class=num_class, pretrained=pretrained)
    return model

# def CPN101(out_size,num_class,pretrained=True):
#     res101 = resnet101(pretrained=pretrained)
#     model = CPN(res101, output_shape=out_size,num_class=num_class, pretrained=pretrained)
#     return model

