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

    batch_size = 5
    workers =16
    disc_radius = 20
    patch_size = 64
    step = 20
    epochs = 500

    visual = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_weights_name = 'Final_for_paper_worm' +'_step_' + str(step) + 'patch_size' +\
        str(patch_size)
    
    data_folder_p = '/home/****/work/colab/work/fiberPJ/data/worm_data' 
    binary_folder = '/home/****/work/colab/work/fiberPJ/data/worm_data/BBBC010_v1_foreground' 
    image_folder = '/home/****/work/colab/work/fiberPJ/data/worm_data/BBBC010_v2_images'  
    
    jason_file_p = 'PATH TO GROUND TRUTH JASON FILE'
    
    jason_file_p = 'worm_ann_keypoint_sequence.json'

    offset_folder = 'worm_ann_keypoint_sequence_test_step_'+'_'+ \
                 str(step) + 'offset'

    data_folder_val = '/home/****/work/colab/work/fiberPJ/data/worm_data' 

    # jason_file_val ='worm_ann_keypoint_sequence_test_step_'+\
    #                 str(step) +'.json'
    jason_file_val ='worm_ann_keypoint_sequence_test_one_way.json'

    image_folder_val = image_folder

    offset_folder_val = offset_folder

    data_folder_test = '/home/****/work/colab/work/fiberPJ/data/worm_data' 
    jason_file_test ='worm_ann_keypoint_sequence_test_one_way.json'
    image_folder_test = image_folder

cfg = Config()
add_pypath(cfg.root_dir)
