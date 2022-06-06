
from matplotlib.path import Path
from sys import path
path.append('./utility/')

# path.append('/home/****/work/detectron2/')
# from detectron2.structures import BoxMode
# import scipy.stats as st
import matplotlib.patches as patches
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
import numpy as np 
import random
import cv2
import os
import json
from utils_for_synthetic_data import compute_offsets
from utility.preprocess import draw_mask_color, visualize_offset
from skimage.morphology import skeletonize
from PIL import Image, ImageDraw
from tqdm import tqdm
from get_intersection_and_endpoint import get_skeleton_endpoint, get_skeleton_intersection_and_endpoint, get_skeleton_intersection


def mask_to_pologons(mask):
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    # import pdb;pdb.set_trace()
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        contour = contour + 0.5
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            polygons.append(contour)
    return polygons
'''
This file to create synthetic dataset in COCO format
images: 
    objects id 
    image Path

annotation: 
    image id
    curve id 
    bbox
    area
    segmentation

'''
def create_curves(width, height, num_cp, reversing_points, num_points, L,deform_down = 1, deform_up = 10 ):

    # random angle to rotate
    an_rot_st = 0
    an_rot = 180
    angle = np.deg2rad(an_rot_st -random.randint(0, an_rot))
    # angle = np.deg2rad(90)
    rot_mat = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    rot_mat = np.asarray(rot_mat)

    #random translation
    trans_x = random.randint(-round(width//2), round(width//2))
    trans_y = random.randint(-round(height//2), round(height//2))
    trans_mat = [[trans_x*np.sin(angle)],[trans_y*np.cos(angle)]]

    #rand om deformations
    # d =  np.random.randn(num_cp)
    d =  np.random.random(num_cp)

    # # import pdb; pdb.set_trace()
    deform = np.random.randint(deform_down,deform_up, size = len(d))
    
    # # big rand

    # # small rand    
    rand_index = random.randint(0,num_cp-1)
    # d[rand_index] = d[rand_index] * deform
    if L > 300:
        d[rand_index] = d[rand_index]  * 100
    else:
        d[rand_index] = d[rand_index]  * 10

    # d = d * deform
    xp = np.linspace(-L/2,L/2,num_cp)

    pp = interpolate.splrep(xp,d);

    #generate points
    x = np.linspace(-L/2,L/2,num_points)
    y = interpolate.splev(x, pp);

    points = np.stack((x,y))
    pts = np.dot(rot_mat,points)


    pts = pts + np.tile(trans_mat,(1, num_points))

    #points to rasterize in an image
    im_pts = np.minimum( np.maximum(0,np.rint(pts+width/2 - 1)),width - 1)
    im_pts = im_pts.astype('int64')

    im_pts_v2 = im_pts.transpose()
    im_pts_v2 = list((map(tuple,im_pts_v2)))

    # get the sequnce and remove duplicate points? I don't remmeber; Figure this out later
    from collections import defaultdict
    dic = defaultdict(int)
    im_pts_v3 = []
    for t in im_pts_v2:
        if t not in dic:
            dic[t] += 1
            if not(t[0] == 0 or t[0] == width - 1 or t[1] == 0 or t[1] == width - 1):
                im_pts_v3.append(t)
    # ######################



    im_pts = tuple(map(tuple,im_pts))


    im = np.zeros((width, height))
    im[im_pts] = 1
    im[0 , :] = 0
    im[im.shape[0] - 1, :] = 0
    im[: , im.shape[1] - 1] = 0
    im[: , 0] = 0
    # plt.imshow(im)
    # plt.show()
    # import pdb; pdb.set_trace()

    return im, im_pts_v3


def main():

    number_of_images = 1000

    num_cp = 5

    deform = 30
    disc_radius = 10
    train_data = {}
    category_infos = []
    category_info= {}
    category_info['supercategory'] = 'fl'
    category_info['id'] = 1
    category_info['name'] = 'filaments'
    category_infos.append(category_info)
    train_data['categories'] = category_infos

    status = ['train']

    # kernel_size_list = [3, 5,7,11]
    # cp_steps = [15,30,60] # control points step
    # widths = [256,512]
    
    kernel_size_list = [5]
    cp_steps = [30] # control points step
    widths = [256]

    for cp_step in cp_steps:
        print(cp_step)
        for width in widths:
            print(width)
            height = width
            for status_use in status:
                for kernel_size in kernel_size_list:
                    gt_file = os.path.join('./dataset/', 'embedding_curve_coco_'+status_use + '_lstm_' +\
                        str(number_of_images)+'_' +  str(width) +'_kernel_' + str(kernel_size) +\
                            '_step_' + str(cp_step) + '.json')

                    directory_curve_pool = './dataset/embedding_curve_coco_'+status_use + '_lstm_' +\
                        str(number_of_images)+'_' +  str(width) +'_kernel_' + str(kernel_size)+\
                            '_step_' + str(cp_step)
                    # directory_curve_pool = './dataset/curve_coco_val_lstm_100_256'
                    if status_use == 'test':
                        reverse = False
                    else:
                        reverse = True

                    if not os.path.exists(directory_curve_pool):
                        os.makedirs(directory_curve_pool)

                    directory_curve_pool_images = directory_curve_pool + '/images'
                    if not os.path.exists(directory_curve_pool_images):
                        os.makedirs(directory_curve_pool_images)
                    offset_matrix_normal_dir = directory_curve_pool + '/offset'
                    if not os.path.exists(offset_matrix_normal_dir):
                        os.makedirs(offset_matrix_normal_dir)

                    curves_id = 0
                    image_id = 0
                    images = []
                    annotations = []
                    num_of_curves_statistics = []
                    length_statistics = []
                    for i in tqdm(range(number_of_images)):
                        single_data = {}
                        img_info = {}


                        kernel = np.ones((kernel_size,kernel_size),np.uint8)


                        input_image = np.zeros((width, height))
                        input_image_dilate = np.zeros((width, height))

                        ################
                        if width < 512:
                            num_of_curves = random.randint(2,7)
                            
                        elif width < 700:
                            num_of_curves = random.randint(4,9)
                        else:
                            num_of_curves = random.randint(6,15)
                        
                        # num_of_curves = random.randint(6,15)

                        ####################

                        prev_skel_im_exsit = False
                        all_end_point = []
                        all_control_point = []
                        
                        
                        num_of_curves_statistics.append(num_of_curves)
                        
                        for ii in range(num_of_curves):

                            instance_curve = {}
                            ####################
                            # L = random.randint(100,600)
                            L = random.randint(100,800)
                            num_points = random.randint(1000,5000)
                            num_cp = random.randint(4,6)
                            deform_up = random.randint(8,30)

                            num_of_reversing_point = random.randint(0,4)
                            reversing_points = np.random.randint(0,num_cp+1, num_of_reversing_point)
                            ####################
                            
                            # create curves
                            # 
                            im, im_pts_v3 = create_curves(width, height, num_cp, reversing_points, num_points, L, deform_up=deform_up)

                            im = skeletonize(im)
                            im = im * 1.
                            imskel = im
                            length_statistics.append(np.sum(imskel))
                            im = cv2.dilate(im, kernel)

                            mask_pologons = mask_to_pologons(im>0)

                            contours,_ = cv2.findContours(np.uint8(im),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            # bbox
                            try:
                                px = contours[0].squeeze()[:,0]
                            except:
                                print('invalid')
                                continue
                            py = contours[0].squeeze()[:,1]
                            #bbox-> top left x , top left y , width and height
                            bbox = [int(np.min(px)),int(np.min(py)),int(np.max(px) - np.min(px)), int(np.max(py) - np.min(py))]
                            bboxMode = 'xywh'
                            
                            im_pts_v3 = np.array(im_pts_v3)
                            im_pts_v3[:,[1,0]] = im_pts_v3[:,[0,1]]

                            # get control points with step: 
                            
                            selected_contour = np.vstack((im_pts_v3[::cp_step], im_pts_v3[-1]))

                            skel = np.where(im > 0)

                            

                            prev_skel_im = im
                            prev_skel_im_exsit = True

                            endpoints = get_skeleton_endpoint(imskel)

                            endpoints = [tuple(map(int, endpoint)) for endpoint in endpoints]


                            skel = [list(map(int,skel_one_axis)) for skel_one_axis in skel]
                            
                            instance_curve['id'] = int(curves_id)
                            instance_curve['image_id'] = int(image_id)
                            instance_curve['endpoints'] = endpoints
                            instance_curve['skel'] =  skel
                            instance_curve['segmentation'] =  mask_pologons
                            instance_curve['bbox'] =  bbox
                            instance_curve['area'] =  int(bbox[2] * bbox[3])
                            instance_curve['iscrowd'] =  0
                            instance_curve['category_id'] =  1

                            diff_x = selected_contour[:,0] - np.pad(selected_contour[:,0],(1,0), 'constant')[:-1]
                            diff_y = selected_contour[:,1] - np.pad(selected_contour[:,1],(1,0), 'constant')[:-1]
                            instance_curve['seq_x_col_offset'] = diff_x.tolist()
                            instance_curve['seq_y_row_offset'] = diff_y.tolist()
                            instance_curve['seq_x_col'] = selected_contour[:,0].tolist()
                            instance_curve['seq_y_row'] = selected_contour[:,1].tolist()
                            instance_curve['control_points'] = selected_contour[:].tolist()
                            
                            #############
                            # offsetmap #
                            #############
                            # sequence_x = selected_contour[:,0]
                            # sequence_y = selected_contour[:,1]

                            # sequence_x_offset = np.zeros(len(sequence_x))
                            # sequence_y_offset = np.zeros(len(sequence_x))

                            # sequence_x_offset[:-1] = sequence_x[1:] - sequence_x[:-1] 
                            # sequence_y_offset[:-1] = sequence_y[1:] - sequence_y[:-1] 

                            # sequence_offset = np.stack((sequence_x_offset, sequence_y_offset)).transpose(0,1)
                            # # import pdb; pdb.set_trace()    
                            # control_points_of_current_curve= [instance_curve['seq_x_col'],instance_curve['seq_y_row']]
                            # # import pdb; pdb.set_trace()
                            # # offset_map = compute_offsets((width, height),sequence_offset,control_points_of_current_curve,\
                            # #                 disc_radius = 10)
                            # offset_map = 0                
                            # np.save(offset_matrix_normal_dir + '/' + str(curves_id) + '.npy', offset_map)

                            ###########
                            #visualize
                            ###########
                            # test_offset_map = np.load(offset_matrix_normal_dir + '/' + str(curves_id) + '.npy')
                            # canvas = np.zeros((width, height))
                            # canvas = visualize_offset(canvas, test_offset_map[0,:,:], test_offset_map[1,:,:])
                            # combined_show = draw_mask_color(np.tile(im,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])
                            # cv2.imshow('short', combined_show)
                            # canvas = np.zeros((width, height))
                            # canvas = visualize_offset(canvas, test_offset_map[2,:,:], test_offset_map[3,:,:])
                            # combined_show = draw_mask_color(np.tile(im,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])

                            # cv2.imshow('mid_offsets_map', combined_show)
                            # cv2.waitKey(0)

                            ###############
                            #             #
                            ###############

                            control_points = [tuple(map(int, control_point)) for control_point in  selected_contour[:]]
                            all_control_point.extend(control_points)
                            all_end_point.extend(endpoints)
 
                            curves_id += 1
                            annotations.append(instance_curve)

                            # reverse
                            
                            if reverse:
                                instance_curve = {}
                                instance_curve['id'] = curves_id

                                diff_x = selected_contour[::-1,0] - np.pad(selected_contour[::-1,0],(1,0), 'constant')[:-1]
                                diff_y = selected_contour[::-1,1] - np.pad(selected_contour[::-1,1],(1,0), 'constant')[:-1]
                                instance_curve['seq_x_col_offset'] = diff_x.tolist()
                                instance_curve['seq_y_row_offset'] = diff_y.tolist()
                                ## instance_curve['seq_x_col'] = diff_x.tolist()
                                ## instance_curve['seq_y_row'] = diff_y.tolist()

                                instance_curve['seq_x_col'] = selected_contour[::-1,0].tolist()
                                instance_curve['seq_y_row'] = selected_contour[::-1,1].tolist()

                                instance_curve['id'] = int(curves_id)
                                instance_curve['image_id'] = int(image_id)
                                instance_curve['endpoints'] = endpoints
                                instance_curve['skel'] =  skel
                                instance_curve['segmentation'] =  mask_pologons
                                instance_curve['bbox'] =  bbox
                                instance_curve['area'] =  int(bbox[2] * bbox[3])
                                instance_curve['iscrowd'] =  0
                                instance_curve['category_id'] =  1
                                instance_curve['control_points'] = selected_contour[::-1].tolist()

                                # ###################################
                                # reverse_save
                                # ###################################
                                sequence_x = selected_contour[::-1,0]
                                sequence_y = selected_contour[::-1,1]

                                sequence_x_offset = np.zeros(len(sequence_x))
                                sequence_y_offset = np.zeros(len(sequence_x))

                                sequence_x_offset[:-1] = sequence_x[1:] - sequence_x[:-1] 
                                sequence_y_offset[:-1] = sequence_y[1:] - sequence_y[:-1] 

                                sequence_offset = np.stack((sequence_x_offset, sequence_y_offset)).transpose(0,1)
                                    
                                control_points_of_current_curve= [instance_curve['seq_x_col'],instance_curve['seq_y_row']]
                                # import pdb; pdb.set_trace()
                                offset_map = compute_offsets((width, height),sequence_offset,control_points_of_current_curve,\
                                                disc_radius = 10)
                                np.save(offset_matrix_normal_dir + '/' + str(curves_id) + '.npy', offset_map)

                                ###########
                                #visualize
                                ###########
                                # test_offset_map = np.load(offset_matrix_normal_dir + '/' + str(curves_id) + '.npy')
                                # canvas = np.zeros((width, height))
                                # canvas = visualize_offset(canvas, test_offset_map[0,:,:], test_offset_map[1,:,:])
                                # combined_show = draw_mask_color(np.tile(im,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])
                                # cv2.imshow('short', combined_show)
                                # canvas = np.zeros((width, height))
                                # canvas = visualize_offset(canvas, test_offset_map[2,:,:], test_offset_map[3,:,:])
                                # combined_show = draw_mask_color(np.tile(im,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])

                                # cv2.imshow('mid_offsets_map', combined_show)
                                # cv2.waitKey(0)
                                # ###################################
                                #
                                # ###################################

                                # img_info ['file_name'] = str(i) + '.png'
                                # img_info ['file_path'] = directory_curve_pool
                                # instance_curve ['img_info'] = img_info
                                curves_id += 1
                                annotations.append(instance_curve)

                            # im_dilate = cv2.dilate(im, kernel)
                            # input_image_dilate = input_image_dilate + im_dilate
                            input_image = input_image + im

                        img_info['id'] = int(image_id)
                        img_info['file_name'] = str(image_id) + '.png'
                        img_info['width'] = int(width)
                        img_info['height'] = int(height)
                        img_info['flickr_url'] = ''
                        img_info['coco_url'] = ''
                        img_info['date_captured'] = 'none'
                        img_info['control_points'] = 'none'
                        img_info['end_points'] = all_end_point
                        img_info['control_points'] = all_control_point

                        images.append(img_info)
                        image_id += 1

                        # input_overlapping = input_image_dilate > 1
                        # input_overlapping = input_overlapping * 1
                        # intersections = get_skeleton_intersection(input_overlapping)

                        # input_image = input_image > 0
                        # input_image = skeletonize(input_image)
                        # input_image = input_image * 1.0

                        # intersections, _ = get_skeleton_intersection_and_endpoint(input_image)
                        # intersections = get_skeleton_intersection(input_image)

                        # intersections = [tuple(map(int, intersection)) for intersection in intersections]
                        # intersections.extend(all_end_point)
                        # import pdb;pdb.set_trace()



                        # single_data['instances'] = instances
                        # single_data['intersections'] =  intersections
                        # single_data ['img_info'] = img_info
                        # input_image = input_image_dilate
                        # input_image = cv2.dilate(input_image, kernel)
                        img_debug = Image.fromarray(input_image * 255.)
                        img_debug = np.asarray(img_debug)
                        # img_debug_draw = ImageDraw.Draw(img_debug)
                        # import pdb; pdb.set_trace()
                        ##############################################
                        # for a in intersections:
                        #     # import pdb; pdb.set_trace()
                        #     cc = 0
                            # img_debug = cv2.circle(img_debug, (a[0], a[1]), 3, (0,255,0), 2)
                            # img_debug.show()
                        # for a in instances:
                        #     end_points = a['endpoints']
                        #     for b in end_points:
                        #         # import pdb; pdb.set_trace()
                        #         img_debug = cv2.circle(img_debug, (b[0], b[1]), 5, (200,0,0), 1)
                        # I = np.asarray(img_debug)
                        # cv2.imshow('t', I)
                        # cv2.waitKey(0)
                        ##################################################


                        input_image = Image.fromarray(input_image * 255)
                        input_image.convert('L').save(directory_curve_pool_images + '/' + str(i) + '.png')
                        


                    print('saving transformed annotation...')
                    train_data['images'] = images
                    train_data['annotations'] = annotations

                    with open(gt_file,'w') as wf:
                        json.dump(train_data, wf)
                        print('done')
                    
                    print(np.mean(length_statistics))
                    print(np.std(length_statistics))
                    print(np.mean(num_of_curves_statistics))
                    print(np.std(num_of_curves_statistics))


if __name__ == '__main__':
    main()

