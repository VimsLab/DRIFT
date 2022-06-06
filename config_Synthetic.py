import os
import os.path
import sys
import numpy as np
import cv2
import torch

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    this_dir_name = cur_dir.split('/')[-1]
    root_dir = os.path.join(cur_dir, '.')
    multi_gpu = False
    # save_weights_name = '_run_offset'
    # save_weights_name = '_run_offset_small_all'
    # save_weights_name = '_run_offset_small_only_offset'
    # save_weights_name = '_run_offset_small_only_offset_old_way_calculateoffset'
    # save_weights_name = '_run_offset_small_only_offset_old_way_calculateoffset_64'
    # save_weights_name = '_run_offset_small_only_offset_old_way_calculateoffset_patch_encoder'
    # save_weights_name = '_run_offset_small_only_offset_old_way_calculateoffset_patch_encoder_with_segmentation'
    # save_weights_name = '_run_offset_small_only_offset_old_way_calculateoffset_patch_encoder_with_segmentation_no_image_encoding'
    # save_weights_name = '_no_imcoding_thick_5'
    epochs = 20
    # kernel_size = 5
    kernel_size = 11
    numofimage= 100
    image_size = 256
    batch_size = 5
    workers = 16 
    step = 15
    patch_size = 32
    visual = False
    save_weights_name = 'for_paper' + str(image_size) + '_numofimage_' + \
        str(numofimage) + '_thick_' + str(kernel_size)+'_step_' + str(step) + 'patch_size' +\
        str(patch_size)\

    data_folder_p = '/home/****/work/codeToBiomix/intersection_detection'  # base name shared by data files
    
    jason_file_p = 'dataset/curve_coco_train_lstm_'+str(numofimage)+'_'+ \
        str(image_size)+'_kernel_'+ str(kernel_size) +'_step_' + str(step) + '.json'
    image_folder = '/home/****/work/codeToBiomix/intersection_detection/dataset/curve_coco_train_lstm_'\
        +str(numofimage)+'_'+\
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step) +'/images'
    offset_folder = '/home/****/work/codeToBiomix/intersection_detection/dataset/curve_coco_train_lstm_'\
        +str(numofimage)+'_'+\
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step) +'/offset'

    data_folder_val = '/home/****/work/codeToBiomix/intersection_detection' 
    jason_file_val ='dataset/curve_coco_val_lstm_'+str(numofimage)+'_'+ \
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step)+ '.json'
    image_folder_val = '/home/****/work/codeToBiomix/intersection_detection/dataset/curve_coco_val_lstm_'\
        +str(numofimage)+'_'+\
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step) +'/images'

    offset_folder_val = '/home/****/work/codeToBiomix/intersection_detection/dataset/curve_coco_val_lstm_'\
        +str(numofimage)+'_'+\
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step) +'/offset'
  
    data_folder_test = '/home/****/work/codeToBiomix/intersection_detection' 
    jason_file_test ='dataset/curve_coco_test_lstm_'+str(numofimage)+'_'+ \
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step)+ '.json'
    image_folder_test = '/home/****/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_'\
        +str(numofimage)+'_'+\
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step) +'/images'

    offset_folder_test= '/home/****/work/codeToBiomix/intersection_detection/dataset/curve_coco_test_lstm_'\
        +str(numofimage)+'_'+\
        str(image_size)+'_kernel_'+ str(kernel_size)+'_step_' + str(step) +'/offset'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc_radius = 20


    # device = torch.device("cpu" )
    # data_shape = (608, 800)
    # output_shape = (608, 800)

    # data_shape = (448, 608)
    # output_shape = (448, 608)
    # data_shape = (256, 512)
    # output_shape = (64, 128)

    # tensorboard_path = './runs/radius_12_k_12_8_6_3'

    # tensorboard_path = './runs/radius_12_only'
    # batch_size = 1
    # data_shape = (576, 768)
    # output_shape = (576, 768)
    # info = 'Training with disc size 12 both keypoint and offset'

    ####################################
    ### CUDA_VISIBLE_DEVICES=0 python test.py --workers=12 -c checkpoint_longshort -t epoch50checkpoint
    # tensorboard_path = './runs/radius_12_in_384_out_192_short_long'
    # batch_size = 4
    # data_shape = (384, 496)
    # output_shape = (192, 256)
    # info = 'Training with data_shape = (384, 496) output_shape = (192, 256) No batch Norm last layer'
###########################################################################
    # tensorboard_path = './runs/radius_12_in_384_out_192_short_long_disc_scale_mid_offset'
    # batch_size = 4
    # data_shape = (384, 496)
    # output_shape = (192, 256)
    # info = 'Training with data_shape = (384, 496) output_shape = (192, 256) No batch Norm last layer'
 #############################################################################
    # tensorboard_path = './runs/radius_double_offset'
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)
    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # ############################################################################
    # tensorboard_path = './runs/radius_double_offset_para_nobn_v2'
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)
    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # #############################################################################
    # tensorboard_path = './runs/radius_double_offset_para_nobn_v2'
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)
    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # # #############################################################################
    # tensorboard_path = './runs/mask_keypoints'

#### Jan 6 ###############
    # batch_size = 1
    # data_shape = (384, 512)
    # output_shape = (384, 512)

    # disc_radius = 20
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'

#### smaller size
    # batch_size = 16
    # data_shape = (256, 256)
    # output_shape = (256, 256)

    # disc_radius = 7
    # info = 'Training with data_shape =(384, 512) output_shape = (384, 512) No batch Norm last layer'
    # # tensorboard_path = './runs/radius_12_in_576_out_768_short_long_disc'
    # # batch_size = 1
    # # data_shape = (576, 768)
    # # output_shape = (576, 768)
    # # info = 'Training with data_shape = (384, 496) output_shape = (192, 256) No batch Norm last layer'

    # crop_width = 0

    # # disc_radius = 12

    # gaussain_kernel = (7, 7)

    # # disc_kernel_points_16 = np.ones((16,16))
    # # disc_kernel_points_12 = np.ones((12,12))
    # # disc_kernel_points_8 = np.ones((8,8))
    # # disc_kernel_points_3 = np.ones((3,3))

    # disc_kernel_points_16 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    # disc_kernel_points_12 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    # disc_kernel_points_8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    # disc_kernel_points_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    # gk15 = (15, 15)
    # gk11 = (11, 11)
    # gk9 = (9, 9)
    # gk7 = (7, 7)
    # # folders = [x[0] for x in os.walk('/usa/****1/colab/data/instance_labeled_file/')]

    # #
    # # gt_path = []
    # # for i in range(2, len(folders),3):
    # #     # import pdb;pdb.set_trace()
    # #     img_root = folders[i]
    # #
    # #     anno_root = folders[i + 1] + '/'
    # #
    # #     target_file = 'ann_keypoint_instance_mask_no_crop.json'
    # #     if 'fb_second_100_test_common' in anno_root:
    # #         continue
    # #     if 'fb_fifth_common' in anno_root:
    # #         continue
    # #     gt_path.append(os.path.join(anno_root, target_file))
    # #
    # # gt_path.append(os.path.join('/usa/****1/colab/work/fiberPJ/data/fiber_labeled_data/', 'ann_keypoint_instance_mask_no_crop.json'))

    # # gt_path = os.path.join('/home/****/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/', 'curves_pool_t_junction_zigzag_di_5.json')
    # gt_path = os.path.join('/home/****/work/colab/work/fiberPJ/pytorch-cpn/data/synthetic/', 'curves_pool_50_step.json')


cfg = Config()
add_pypath(cfg.root_dir)
