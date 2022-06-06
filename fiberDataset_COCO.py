import os
import time
from torch.utils.data import Dataset
import numpy as np
import json
import random
import math
import cv2
import skimage
import skimage.transform
import copy
from tqdm import tqdm
import torch
import copy
from cocoapi.PythonAPI.pycocotools.coco import COCO
import matplotlib.pyplot as plt
from scipy.io import loadmat
# import torch.utils.data as data
# from cocoapi.PythonAPI.pycocotools.coco import COCO

from utility.preprocess import get_keypoint_discs, get_keypoint_discs_offset, compute_short_offsets, compute_mid_offsets,sequence_patches,sequence_segmentation_patches
from utility.preprocess import draw_mask, visualize_offset,draw_mask_color,sequence_patches_with_offset_matrix
from utility.preprocess import gen_gaussian_map
from config_Synthetic import cfg
import imageio

class fiberDataset_COCO(Dataset):
    def __init__(self, data_folder, jason_file, image_folder, offset_folder, transform, train=True):
        self.image_folder = image_folder
        self.offset_folder = offset_folder
        self.is_train = train
        self.data_folder = data_folder
        self.disc_radius = 10
        self.transform = transform
        self.max_len = 100
        self.debug = False
        # self.debug = True
        self.gt_path = data_folder + '/' + jason_file
        self.fiber_coco = COCO(self.gt_path)
        self.ids = list(self.fiber_coco.anns.keys())
        self.img_ids = list(self.fiber_coco.imgs.keys())

        if train:
            self.anno = []
            with open(self.gt_path) as anno_file:
                self.anno.extend(json.load(anno_file))

        # else:
        #     self.anno = []

        #     with open(cfg.gt_path) as anno_file:
        #         self.anno.extend(json.load(anno_file))

    def __getitem__(self, index):
        # import pdb;pdb.set_trace()

        if self.is_train:
            coco_fiber = self.fiber_coco
            ann_id = self.ids[index]
            img_id = coco_fiber.anns[ann_id]['image_id']
            file_name= coco_fiber.loadImgs(img_id)[0]['file_name']

            img_path =  os.path.join(self.image_folder, file_name)

            image = imageio.imread(img_path)
            a = coco_fiber.anns[ann_id]

            image = np.asarray(image)

            image_shape = (image.shape[0], image.shape[1])
            
            #####################   
            # tic = time.time()
            # import pdb; pdb.set_trace()
            image_control_points = coco_fiber.loadImgs(img_id)[0]['control_points']
            image_end_points = coco_fiber.loadImgs(img_id)[0]['end_points']

            image_level_gt = np.zeros((2,image.shape[0], image.shape[1]), dtype = np.float32)

            end_points_off_sets_shorts_map_h_final = np.zeros(image_shape, dtype = np.float32)
            end_points_off_sets_shorts_map_v_final = np.zeros(image_shape, dtype = np.float32)
            end_points_map_final = np.zeros(image_shape, dtype = np.float32)
            end_points_label = np.reshape(np.asarray(image_end_points), (-1, 2))

            control_points_off_sets_shorts_map_h_final = np.zeros(image_shape, dtype = np.float32)
            control_points_off_sets_shorts_map_v_final = np.zeros(image_shape, dtype = np.float32)
            control_points_map_final = np.zeros(image_shape, dtype = np.float32)
            control_points_label = np.reshape(np.asarray(image_control_points), (-1, 2))
            
            heatmap_cotrol = gen_gaussian_map(control_points_label, image_shape, 3)
            heatmap_end = gen_gaussian_map(end_points_label, image_shape, 3)
            # cmap = plt.get_cmap('jet')
            # rgba_img = cmap(heatmap)
            # cv2.imshow('rgba_img',rgba_img)
            # cv2.waitKey(0)

            # for idx, i in enumerate(end_points_label):
            #     # print(idx)
                
            #     end_points_map_final=cv2.circle(end_points_map_final, (i[0], i[1]), self.disc_radius, 1, -1)
            #     # end_points_map_final[i[1], i[0]] = 1

            # for idx, i in enumerate(control_points_label):
            #     # print(idx)
            #     control_points_map_final = cv2.circle(control_points_map_final, (i[0], i[1]), self.disc_radius, 1, -1)

                # control_points_map_final[i[1], i[0]] = 1

            # end_points_discs = get_keypoint_discs(end_points_label, image_shape, radius = self.disc_radius)
            # control_points_discs = get_keypoint_discs(control_points_label, image_shape, radius = self.disc_radius)

            # control_points_short_offset, canvas = compute_short_offsets(control_points_label, control_points_discs, map_shape = image_shape, radius =  self.disc_radius)
            # end_points_short_offset, canvas = compute_short_offsets(end_points_label, end_points_discs, map_shape = image_shape, radius =  self.disc_radius)

            image_level_gt[0,:,:] = heatmap_end
            image_level_gt[1,:,:] = heatmap_cotrol
            # image_level_gt[2:4,:,:] = np.transpose(end_points_short_offset\
            #                                         ,(2,0,1))
            # image_level_gt[4:6,:,:] = np.transpose(control_points_short_offset\
            #                                         ,(2,0,1))
            # print(f"global time: {time.time() - tic:.4f}")
            #########################################
            # cv2.imshow('control_points_discs', image_level_gt[1,:,:]  * 255.)
            # cv2.imshow('end_points_discs', image_level_gt[0,:,:] * 255.)

            # canvas = np.zeros(image_shape)
            # canvas = visualize_offset(canvas, image_level_gt[2,:,:], image_level_gt[3,:,:])

            # combined_show = draw_mask_color(np.tile(image,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])
            # cv2.imshow('endpoints_offset', combined_show)
            
            # canvas = np.zeros(image_shape)
            # canvas = visualize_offset(canvas, image_level_gt[4,:,:], image_level_gt[5,:,:])
            # combined_show = draw_mask_color(np.tile(image,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])
            # cv2.imshow('control_points_offset', combined_show)
            # cv2.waitKey(0)

            #################################################
            ##############################

            sequence_x = np.zeros(self.max_len)
            sequence_y = np.zeros(self.max_len)
            stop_sequence = np.zeros(self.max_len)

            sequence_len = len(a['seq_x_col'])
            sequence_x[:len(a['seq_x_col'])] = a['seq_x_col']
            # sequence_x[len(a['seq_x_col'])] = 9999
            sequence_y[:len(a['seq_y_row'])] = a['seq_y_row']
            # sequence_y[len(a['seq_y_row'])] = 9999
            stop_sequence[len(a['seq_x_col']) - 1] = 1.

            sequence_x_offset = np.zeros(self.max_len)
            sequence_y_offset = np.zeros(self.max_len)
            
            sequence_x_offset[:-1] = sequence_x[1:] - sequence_x[:-1] 
            sequence_y_offset[:-1] = sequence_y[1:] - sequence_y[:-1] 

            # import pdb; pdb.set_trace()
            if self.debug:
            # if True:

                canvas_image = copy.deepcopy(image)
                cv2.imwrite('demo_output/lstmdemo/'+ 'org.png', canvas_image)
                canvas_image = np.tile(canvas_image,(3,1,1)).transpose(1,2,0)
                canvas = np.zeros(image.shape)
                canvas_rec = np.zeros(image.shape)
                for pp in tqdm(range(len(sequence_x))):
                    # import pdb; pdb.set_trace()
                    if pp == 0:
                        pre = (int(sequence_x[pp]), int(sequence_y[pp]))
                        canvas = cv2.circle(canvas, (int(sequence_x[pp]), int(sequence_y[pp])), 5, (200,0,0), 3)
                        test_show = draw_mask_color(canvas_image, canvas, (0,255,0))
                        canvas_rec  = cv2.rectangle(canvas_rec, (int(sequence_x[pp])-32,int(sequence_y[pp])-32),(int(sequence_x[pp])+32,int(sequence_y[pp])+32), (200,0,0), 2) 
                        canvas = (canvas > 0) * 255.
                        test_show = draw_mask_color(test_show, canvas_rec, (0,0,255))
                        cv2.imshow('test', test_show)
                        cv2.imwrite('demo_output/lstmdemo/'+str(pp) + '.png', test_show)
                        cv2.waitKey()

                        continue

                    cc = 0
                    if sequence_x[pp] == 9999:
                        break

                    if pp % 2 == 1:
                        continue            
                    canvas = cv2.circle(canvas, (int(sequence_x[pp]), int(sequence_y[pp])), 5, (200,0,0), 3)
                    test_show = draw_mask_color(canvas_image, canvas, (0,255,0))

                    canvas_rec  = cv2.rectangle(canvas_rec, (int(sequence_x[pp])-32,int(sequence_y[pp])-32),(int(sequence_x[pp])+32,int(sequence_y[pp])+32), (200,0,0), 2) 
                    pre = (pre[0] + int(sequence_x_offset[pp-1]), pre[1] + int(sequence_y_offset[pp-1]))
                    canvas = (canvas > 0) * 255.
                    test_show = draw_mask_color(test_show, canvas_rec, (0,0,255))
                    cv2.imshow('test', test_show)
                    cv2.imwrite('demo_output/lstmdemo/'+str(pp) + '.png', test_show)
                    cv2.waitKey()





            canvas = np.zeros((image.shape[0], image.shape[1]))
            # hard coded
            '''
            # set origion point with a bounding box mask

            radius = 10

            canvas[int(max(0,sequence_y[0] - radius)) : int(min(sequence_y[0] + radius, 255)), int(max(0,sequence_x[0] - radius)): int(min(sequence_x[0] + radius, 255))] = 255.

            canvas = np.expand_dims(canvas,2)
            image = np.expand_dims(image,2)
            image = np.concatenate((image, canvas, canvas), 2)
            image = np.asarray(image)
            '''
            # import pdb; pdb.set_trace()
            if len(image.shape) == 2:
                image = np.tile(image,(3,1,1)).transpose(1,2,0)
                image = np.asarray(image)
            
            # image = cv2.resize(image, (256,256))
            # cv2.imshow('t', image)
            # cv2.waitKey(0)


            def im_to_torch(img):
                img = np.transpose(img, (2, 0, 1)) # C*H*W
                img = torch.from_numpy(img).float()
                if img.max() > 1:
                    img /= 255
                return img

            kernel = np.ones((3,3),np.uint8)
            # import pdb; pdb.set_trace()
            # image =  cv2.dilate(image, kernel)

            # image = cv2.resize(image, (256,256))
            sequence_x = sequence_x * (256/256)
            sequence_y = sequence_y * (256/256)
            # image = im_to_torch(image)  # CxHxW
            img = self.transform(image)

            # img = torch.cat((img,canvas),0)

            sequence_x = torch.FloatTensor(sequence_x)
            sequence_y = torch.FloatTensor(sequence_y)

            sequence_x_offset = torch.FloatTensor(sequence_x_offset)
            sequence_y_offset = torch.FloatTensor(sequence_y_offset)

            sequence = torch.stack((sequence_x, sequence_y)).transpose(0,1)
            sequence_offset = torch.stack((sequence_x_offset, sequence_y_offset)).transpose(0,1)

            sequence_len = torch.LongTensor([sequence_len])
            meta = {'img_path' : img_path}


            #########################################################################
            
            # Segmentation
            skel = (a['skel'][0],a['skel'][1])
            segmentation_mask_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            segmentation_mask_final[skel] = 1
            segmentation_mask_final = cv2.dilate(segmentation_mask_final, np.ones((3,3)))

            #################
            # insatnce_mask = draw_mask_color(canvas_image, segmentation_mask_final, [0,0,255])
            # cv2.imwrite('demo_output/lstmdemo/'+ 'insatnce_mask.png', insatnce_mask)
            ######################
            # control points
            control_points = [a['seq_x_col'], a['seq_y_row']]

            patches_seqeunces = sequence_patches(image, self.max_len, sequence_offset, segmentation_mask_final, control_points, \
                        patch_size= cfg.patch_size, disc_radius= 15)
            # tic = time.time()

            # patches_seqeunces = sequence_segmentation_patches(image, self.max_len, sequence_offset, segmentation_mask_final, control_points, \
            #             patch_size= 64, disc_radius= 10)

            matrix_file = str(ann_id) + '.npy'
            matrix_file_path = os.path.join(self.offset_folder, matrix_file)
            offset_matrix = np.load(matrix_file_path)
            # import pdb; pdb.set_trace()
            # patches_seqeunces = sequence_patches_with_offset_matrix(image, self.max_len, offset_matrix, segmentation_mask_final, control_points, \
            #             patch_size= 128, disc_radius= 10)
            # print(f"patches_seqeunces time: {time.time() - tic:.4f}")
            # import pdb; pdb.set_trace()
            ###########################################################################################################################
            '''
            # end points
            end_points = a['endpoints']

            end_points_map = np.zeros(image_shape, dtype = np.float32)
            end_points_label = np.reshape(np.asarray(end_points), (-1, 2))

            end_points_discs = get_keypoint_discs(end_points_label, image_shape, radius = self.disc_radius)
            for disc in end_points_discs:
                end_points_map += disc * 1.0
            
            # cv2.waitKey(0)
            
            end_points_short_offset, canvas = compute_short_offsets(end_points_label, end_points_discs, map_shape = image_shape, radius =  self.disc_radius)
            # ####################################################################################################################
            # end_points_map_final is for all end point in current image | 
            # end_points_map_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            ######################################################################################################################
            # Control points
            control_points_map = np.zeros(image_shape, dtype = np.float32)

            control_points = [a['seq_x_col'], a['seq_y_row']]
            control_points = np.transpose(control_points)
            control_points_label = np.reshape(np.asarray(control_points), (-1, 2))
            control_points_discs = get_keypoint_discs(control_points_label, image_shape, radius = self.disc_radius)
            for disc in control_points_discs:
                control_points_map += disc * 1.0


            control_points_short_offset, canvas = compute_short_offsets(control_points_label, control_points_discs, map_shape = image_shape, radius =  self.disc_radius) 
            
            #######################################################################################################################
            # Mid offset
            import pdb; pdb.set_trace()
            off_sets_nexts_map_h = np.zeros(image_shape, dtype = np.float32)
            off_sets_nexts_map_v = np.zeros(image_shape, dtype = np.float32)
            
            for idx, points_coor in enumerate(control_points):
                if idx == len(control_points) - 1:
                    off_sets_nexts_map_h[points_coor[1], points_coor[0]] = 0
                    off_sets_nexts_map_v[points_coor[1], points_coor[0]] = 0            
                else:
                    off_sets_nexts_map_h[points_coor[1], points_coor[0]] = sequence_offset[idx][0]
                    off_sets_nexts_map_v[points_coor[1], points_coor[0]] = sequence_offset[idx][1]

            control_points_nexts_offset, canvas = compute_mid_offsets(control_points_label, off_sets_nexts_map_h, off_sets_nexts_map_v, image_shape, control_points_discs)

            #control_points_map_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            # end_points_off_sets_shorts_map_h_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            # end_points_off_sets_shorts_map_v_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            # control_points_off_sets_shorts_map_h_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            # control_points_off_sets_shorts_map_v_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)

            skel = (a['skel'][0],a['skel'][1])
            segmentation_mask_final = np.zeros((image.shape[0], image.shape[1]), dtype = np.float32)
            segmentation_mask_final[skel] = 1
            segmentation_mask_final = cv2.dilate(segmentation_mask_final, np.ones((3,3)))
            if self.debug:

            # if True:
                cv2.imshow('end_points_map', end_points_map * 255.)
                cv2.imshow('control_points_map', control_points_map * 255.)
                cv2.imshow('segmentation', segmentation_mask_final)
                cv2.imshow('image', image)

                canvas = np.zeros_like(end_points_map)
                canvas = visualize_offset(canvas, end_points_short_offset[:,:,0], end_points_short_offset[:,:,1])

                combined_show = draw_mask_color(image, canvas, [0., 255., 0.])
                cv2.imshow('end_points_short_offset',combined_show)

                canvas = np.zeros_like(control_points_map)
                canvas = visualize_offset(canvas, control_points_short_offset[:,:,0], control_points_short_offset[:,:,1])

                combined_show = draw_mask_color(image, canvas, [0., 255., 0.])
                cv2.imshow('control_points_short_offset',combined_show)

                canvas = np.zeros_like(control_points_map)
                canvas = visualize_offset(canvas, control_points_nexts_offset[:,:,0], control_points_nexts_offset[:,:,1])

                combined_show = draw_mask_color(image, canvas, [0., 255., 0.])
                cv2.imshow('control_points_nexts_offset',combined_show)
                cv2.waitKey(0)


            import pdb; pdb.set_trace()
            ground_truth_multi = [segmentation_mask_final, control_points_map, end_points_map, control_points_short_offset, end_points_short_offset, control_points_nexts_offset]
            #########################################################
            '''
            patches_seqeunces = torch.Tensor(patches_seqeunces)
            image_level_gt = torch.Tensor(image_level_gt)
            return img, sequence, sequence_offset, sequence_len, stop_sequence, patches_seqeunces, image_level_gt, meta

        else:
            coco_fiber = self.fiber_coco
            img_id = self.img_ids[index]
            file_name= coco_fiber.loadImgs(img_id)[0]['file_name']
            img_path =  os.path.join(self.image_folder, file_name)
            
            ##########
            # img_path = '~/work/data/mtdata/14_man.png'
            # file_name = '14_man.png'

            # img_path = '~/work/data/zder_s/dd1.jpg'
            # file_name = 'dd1.jpg'
            # img_path = '~/work/data/zder_s/dd4.jpg'
            # file_name = 'dd4.jpg'
            # img_path = '~/work/data/zder_s/dd3.jpg'
            # file_name = 'dd3.jpg'
            # # img_path = '~/work/data/zder_s/ee2.jpg'
            # # file_name = 'ee2.jpg'

            ##########

            image = imageio.imread(img_path)
            # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            image = np.asarray(image)

            ###########################
            # h, w = image.shape[:2]
            # image = image[700:700 + 1000,1000:1000 + 1000]
            # print(image.shape)
            # from utility.color_map import ColorToLabel

            #######################################
            # image = image[:,:,:3]
            # image = np.dot(image, [0.299, 0.587, 0.114])

            # image = (255 * (image > 0)).astype('uint8')
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            ####
            
            # h, w = image.shape[:2]
            
            # crop_height = h % 32
            # crop_width = w % 32
            # image = image[int(crop_height / 2) : int(h - crop_height /2),int(crop_width / 2) : int(w - crop_width /2)]
            # # image = image[int(800) : int(800+256),int(700) : int(700+256)]
            # # image = image[int(300) : int(300+512),int(100):int(100+512)]
            
            # #######

            # meta = {'index': index, 'imgID' : img_info['id'], 'img_path' : file_path, 'out_shape':self.out_res, 'input_shape':input_shape}
            if len(image.shape) == 2:
                image = np.tile(image,(3,1,1)).transpose(1,2,0)
                image = np.asarray(image)
 
            img = self.transform(image)

            meta = {'index' : index,
            'img_path' : img_path,
            'file_name' : file_name,
            'image_id': img_id}##, 'augmentation_details' : details}

            meta['det_scores'] = float(1)

            return img, meta

    def __len__(self):
        if self.is_train:
            return len(self.ids)
            # return 20
        else:
            return len(self.img_ids)
            # return 8
