
import sys
sys.path.insert(0, '../../')

import os
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import scipy.ndimage as ndimage
import math
import skfmm

from pycocotools.coco import COCO
from fiber_tools.common.util.color_map import GenColorMap
from fiber_tools.common.util.cv2_util import pologons_to_mask, mask_to_pologons
from get_skeleton_intersection import get_skeleton_intersection_and_endpoint, get_skeleton_endpoint


from skimage.morphology import skeletonize
from least_square import least_squares

from tqdm import tqdm
import json
from fiber_tools.common.util.osutils import isfile
import sys
import numpy as np


from utility.preprocess import get_keypoint_discs
from utility.preprocess import compute_short_offsets, compute_mid_offsets
from utility.preprocess import draw_mask_color, visualize_offset, draw_mask


# end to end Trim
def get_end_to_end_contour(selected_contour, tips):
    tmp_contour = selected_contour
    startIdx = -1
    used0 = False
    used1 = False
    for idx, point in enumerate(tmp_contour):
        # print(point)
        if point[0][0] == tips[0][0] and point[0][1] == tips[0][1] and not used0:
            used0 = True
            if startIdx == -1:
                startIdx = idx
            else:
                endIdx = idx
        elif point[0][0] == tips[1][0] and point[0][1] == tips[1][1] and not used1:
            used1 = True
            if startIdx == -1:
                startIdx = idx
            else:
                endIdx = idx
        if used0 and used1:
            break
    try:
        selected_contour = selected_contour[startIdx:endIdx + 1]
    except:
        import pdb; pdb.set_trace()
    return selected_contour

def compute_offsets(image_shape, sequence_offset, control_points, \
                    disc_radius):
    '''
    shape: image 3; [segmentaiton:1, contro_points_label: 1 ; short_offset:2 ; next_point:2] 
    
    one curve correpond to one fullsize matrix

    input sequence

    return matrix of shape [c, h,w]
    c = 4: shortoffset 2; midoffset 2
    '''
    channel = 5

    offsetMap =np.zeros((channel,image_shape[0],image_shape[1])) 


    control_points = np.transpose(control_points)
    control_points_label = np.reshape(np.asarray(control_points), (-1, 2))

    control_points_discs = get_keypoint_discs(control_points_label, image_shape, radius = disc_radius)
    
    short_offset,_ = compute_short_offsets(control_points_label, control_points_discs, image_shape, disc_radius)

    off_sets_nexts_map_h = np.zeros(image_shape, dtype = np.float32)
    off_sets_nexts_map_v = np.zeros(image_shape, dtype = np.float32)
    for idx, points_coor in enumerate(control_points):

        if idx == len(control_points) - 1:
            off_sets_nexts_map_h[points_coor[1], points_coor[0]] = 0
            off_sets_nexts_map_v[points_coor[1], points_coor[0]] = 0       
        else:
            off_sets_nexts_map_h[points_coor[1], points_coor[0]] = sequence_offset[0][idx]
            off_sets_nexts_map_v[points_coor[1], points_coor[0]] = sequence_offset[1][idx]      

    control_points_nexts_offset, _ = \
        compute_mid_offsets(control_points_label, off_sets_nexts_map_h, off_sets_nexts_map_v, image_shape, control_points_discs)     
    
    control_points_nexts_offset = np.transpose(control_points_nexts_offset, (2,0,1))

    short_offset =  np.transpose(short_offset, (2,0,1))
    offsetMap[0:2] = short_offset
    offsetMap[2:4] = control_points_nexts_offset
    return offsetMap


def get_angle(x, y):
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle_radius=np.arccos(cos_angle)
    angle_degree=angle_radius*360/2/np.pi
    return angle_degree

def draw_mask(im, mask, color):


    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    # import pdb; pdb.set_trace()
    mask = mask>0
    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    #combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    combined = cv2.merge([r, g, b])

    return combined.astype(np.uint8)

def get_keypoints(img, true_image, step, crop_edge, debug):

    end_points = []
    control_points = []
    off_sets_prev = []
    off_sets_next = []
    #################
    skel = skeletonize(img)
    skel = skel.astype(np.uint8)
    skel = skel[crop_edge:skel.shape[0] - crop_edge, crop_edge : skel.shape[1] - crop_edge]
    true_image = true_image[crop_edge:true_image.shape[0] - crop_edge, crop_edge : true_image.shape[1] - crop_edge, :]
    canvas = np.zeros(skel.shape)
    intersections, _ = get_skeleton_intersection_and_endpoint(skel)

    # center = (endpoints_tmp[0][1], endpoints_tmp[0][0])
    # distance = geodesic_distance_transform(skel,center)
    # cv2.imshow('t', distance)
    # cv2.waitKey(0)
    # import pdb; pdb.set_trace()

    # Prune branches caused by skeletonization
    if len(intersections):
        # print('more than one intersections')
        # seperate the branches as individual contours.
        for p in intersections:
            cv2.circle(skel, p , 2, (0), -1)

        # find contours
        im2, contours, hierarchy = cv2.findContours(skel,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        # find the longest contour.
        longest = 0
        ########################################################################
        #canvas_debug = np.zeros(skel.shape)
        #######################################################################
        for idx, contour in enumerate(contours):
            #####################################################################
            # canvas_debug = cv2.drawContours(canvas_debug, contours, idx, 1, 1)
            # cv2.imshow('t', canvas_debug)
            # cv2.waitKey(0)
            #######################################################################
            if cv2.arcLength(contour, False) > cv2.arcLength(contours[longest], False):
                longest = idx

        longest_contour = contours[longest]

    else:
        # find contours
        im2, contours, hierarchy = cv2.findContours(skel,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # canvas_debug = np.zeros(skel.shape)
        # canvas_debug = cv2.drawContours(canvas_debug, contours, 0, 1, 1)
        # cv2.imshow('t', canvas_debug)
        # cv2.waitKey(0)
        if len(contours) < 1:
            return end_points, off_sets_next, control_points, off_sets_prev

        longest_contour = contours[0]
        longest = 0

    #################################################################
    # new_contour_arr = [tuple(row) for row in contours[0]] #remove duplicates
    # uniques = np.unique(new_contour_arr, axis = 0)
    ################################################################
    # import pdb; pdb.set_trace()
    # To find the end points
    reset_contour_canvas = np.zeros(skel.shape)
    reset_contour_canvas = cv2.drawContours(reset_contour_canvas, contours, longest, 1, 1)
    _, tips = get_skeleton_intersection_and_endpoint(reset_contour_canvas)
    endpoints_tmp = get_skeleton_endpoint(reset_contour_canvas)

    try:
        stpt = endpoints_tmp[0] #start point default
    except:
        return end_points, off_sets_next, control_points, off_sets_prev

    if len(tips) < 2:
        
        return end_points, off_sets_next, control_points, off_sets_prev

    longest_contour = get_end_to_end_contour(longest_contour, tips)
    #left most and down most point as start point
    # for ept in endpoints_tmp:
    #     if (ept[0] < stpt[0]):
    #         stpt = ept
    #     elif(ept[0] == stpt[0]):
    #         if(ept[1] < stpt[1]):
    #             stpt = ept


    # #swap: start from start point
    # cut_point_x = np.where(longest_contour[:,0,0] == stpt[0])
    # cut_point_y = np.where(longest_contour[:,0,1] == stpt[1])
    # cut_point = cut_point_x and cut_point_y
    # cut_point = cut_point[0][0]
    # part1 = longest_contour[cut_point:]
    # part2 = longest_contour[:cut_point]
    # longest_contour = np.concatenate((part1, part2))



    # Sample contour with consistent interval
    longest_contour_sampled = longest_contour[0 : int(len(longest_contour)): step]

    if debug:
        cv2.circle(canvas, (longest_contour[0][0][0],longest_contour[0][0][1]) , 3, 1, -1)
        cv2.circle(canvas, (longest_contour[int(len(longest_contour))][0][0],longest_contour[int(len(longest_contour))][0][1]) , 5, 1, 1)

    # Deal with the last point.
    # if the distance between the last point and the end point is less than one third of step
    # Use the end point directly.
    if len(longest_contour_sampled) == 1:
        if debug:
            print('one point in sampled')
        end_points.append(longest_contour[0][0][0])
        end_points.append(longest_contour[0][0][1])
        end_points.append(longest_contour[int(len(longest_contour)-1)][0][0])
        end_points.append(longest_contour[int(len(longest_contour)-1)][0][1])

        control_points.append(longest_contour[0][0][0])
        control_points.append(longest_contour[0][0][1])
        control_points.append(longest_contour[int(len(longest_contour)-1)][0][0])
        control_points.append(longest_contour[int(len(longest_contour)-1)][0][1])
        off_sets_next.append(longest_contour[0][0][0] - longest_contour[int(len(longest_contour)-1)][0][0])
        off_sets_next.append(longest_contour[0][0][1] - longest_contour[int(len(longest_contour)-1)][0][1])
        off_sets_next.append(0)
        off_sets_next.append(0)
        off_sets_prev.append(0)
        off_sets_prev.append(0)
        off_sets_prev.append(longest_contour[int(len(longest_contour)-1)][0][0] - longest_contour[0][0][0])
        off_sets_prev.append(longest_contour[int(len(longest_contour)-1)][0][1] - longest_contour[0][0][1])

    elif((len(longest_contour) ) % step < (step / 2)):
        #merge
        if debug:
            print('less than 1/3')
        for pt in range(len(longest_contour_sampled)):
            if pt == (len(longest_contour_sampled) - 1):
                end_points.append(longest_contour[int(len(longest_contour)-1)][0][0])
                end_points.append(longest_contour[int(len(longest_contour)-1)][0][1])
                control_points.append(longest_contour[int(len(longest_contour)-1)][0][0])
                control_points.append(longest_contour[int(len(longest_contour)-1)][0][1])

                off_sets_next.append(0)
                off_sets_next.append(0)
                off_sets_prev.append(longest_contour[int(len(longest_contour)-1)][0][0] - longest_contour_sampled[pt - 1][0][0] )
                off_sets_prev.append(longest_contour[int(len(longest_contour)-1)][0][1] - longest_contour_sampled[pt - 1][0][1] )

            elif pt == 0:
                end_points.append(longest_contour_sampled[pt][0][0])
                end_points.append(longest_contour_sampled[pt][0][1])
                control_points.append(longest_contour_sampled[pt][0][0])
                control_points.append(longest_contour_sampled[pt][0][1])

                off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0])
                off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1])
                off_sets_prev.append(0)
                off_sets_prev.append(0)

            elif pt == (len(longest_contour_sampled) - 2):

                control_points.append(longest_contour_sampled[pt][0][0])
                control_points.append(longest_contour_sampled[pt][0][1] )


                off_sets_prev.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                off_sets_prev.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )
                off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour[int(len(longest_contour)-1)][0][0])
                off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour[int(len(longest_contour)-1)][0][1])

            else:
                control_points.append(longest_contour_sampled[pt][0][0])
                control_points.append(longest_contour_sampled[pt][0][1])
                off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0] )
                off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1] )
                off_sets_prev.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                off_sets_prev.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )
    else:
        # import pdb; pdb.set_trace()

        if debug:
            print('more than 1/3')
        for pt in range(len(longest_contour_sampled)):
            if pt == (len(longest_contour_sampled) - 1):
                control_points.append(longest_contour_sampled[pt][0][0])
                control_points.append(longest_contour_sampled[pt][0][1])

                off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour[int(len(longest_contour)-1)][0][0])
                off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour[int(len(longest_contour)-1)][0][1])

                off_sets_prev.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                off_sets_prev.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )
            elif pt == 0:
                end_points.append(longest_contour_sampled[pt][0][0])
                end_points.append(longest_contour_sampled[pt][0][1])
                control_points.append(longest_contour_sampled[pt][0][0])
                control_points.append(longest_contour_sampled[pt][0][1])

                off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0])
                off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1])
                off_sets_prev.append(0)
                off_sets_prev.append(0)
            else:
                control_points.append(longest_contour_sampled[pt][0][0])
                control_points.append(longest_contour_sampled[pt][0][1])
                off_sets_next.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt + 1][0][0] )
                off_sets_next.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt + 1][0][1] )
                off_sets_prev.append(longest_contour_sampled[pt][0][0] - longest_contour_sampled[pt - 1][0][0] )
                off_sets_prev.append(longest_contour_sampled[pt][0][1] - longest_contour_sampled[pt - 1][0][1] )

        end_points.append(longest_contour[int(len(longest_contour) -1)][0][0])
        end_points.append(longest_contour[int(len(longest_contour) -1)][0][1])
        control_points.append(longest_contour[int(len(longest_contour) -1)][0][0])
        control_points.append(longest_contour[int(len(longest_contour) -1)][0][1])
        off_sets_next.append(0)
        off_sets_next.append(0)
        off_sets_prev.append(longest_contour[int(len(longest_contour) -1)][0][0] - longest_contour_sampled[-1][0][0] )
        off_sets_prev.append(longest_contour[int(len(longest_contour)-1)][0][1] - longest_contour_sampled[-1][0][1] )

    ##################

    if debug:
        for pt in range(0, len(control_points), 2):

            curr = (control_points[pt], control_points[pt + 1])
            next_pt = (control_points[pt] - off_sets_next[pt], control_points[pt + 1] - off_sets_next[pt + 1])

            cv2.arrowedLine(canvas, curr, next_pt, 1, 1)
        combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
        cv2.imshow('t',combined_show)
        cv2.waitKey(0)

    if debug:
        for pt in range(0, len(control_points), 2):

            curr = (control_points[pt], control_points[pt + 1])
            next_pt = (control_points[pt] - off_sets_prev[pt], control_points[pt + 1] - off_sets_prev[pt + 1])

            cv2.arrowedLine(canvas, curr, next_pt, 1, 1)
        combined_show = draw_mask(true_image, canvas, [0., 255., 0.])
        cv2.imshow('t',combined_show)
        cv2.waitKey(0)


    return end_points , control_points, off_sets_prev, off_sets_next

def trans_anno(instance_root, binary_root, org_image_folder, output_root, target_file, reverse_seq = False):
	
    train_anno = os.path.join(output_root, target_file)
    reverse_seq = reverse_seq
    idx = 0

    # step_size = 20
    # step_size = 60
    step_size = 15
    binary_images_files = os.listdir(binary_root)
    org_image_files = os.listdir(org_image_folder)
    crop_edge = 0
    train_data = {}
    category_infos = []
    category_info= {}
    category_info['supercategory'] = 'w'
    category_info['id'] = 1
    category_info['name'] = 'worms'
    category_infos.append(category_info)
    train_data['categories'] = category_infos

    print('transforming annotations...')
    num_bad_images = 0
    num_good_images = 0

    anno_id = 0
    images = []
    annotations = []
        # for img_id in tqdm(coco_ids):
    img_id = 0
    img_id_map = {}

    new_image = False
    for filename in os.listdir(instance_root):
        end_points = []
        control_points = []
        f = os.path.join(instance_root, filename)
        if os.path.isfile(f):
            print(f)
        file_name = f.split('/')[-1]
        image_id = file_name.split('_')[0]

        if image_id in img_id_map:
            current_id = img_id_map[image_id]
        else:
            new_image = True
            img_id_map[image_id] = img_id
            img_id += 1
            current_id = img_id_map[image_id]


        for name in binary_images_files:
            if image_id in name:
                binary_image_name = name

        for name in org_image_files:
            if image_id in name:
                if image_id + '_w1' in name:
                    w1 = name
                elif image_id+'_w2' in name:
                    w2 = name


        try:
            instance_id = file_name.split('_')[1]
        except:
            print(file_name)
            continue

        file_path = os.path.join(instance_root, file_name)
        semantic_binary_path = os.path.join(binary_root, binary_image_name)

        w2_path = os.path.join(org_image_folder, w2)

        this_isntance_mask = cv2.imread(file_path)
        this_isntance_mask = cv2.cvtColor(this_isntance_mask, cv2.COLOR_BGR2GRAY);
        this_isntance_mask = (255. * (this_isntance_mask>0)).astype('uint8')


        polygon_this_instance = mask_to_pologons(this_isntance_mask) 

        semantic_binary = cv2.imread(semantic_binary_path)
        semantic_binary = semantic_binary.astype('uint8')
        semantic_binary = cv2.cvtColor(semantic_binary, cv2.COLOR_BGR2GRAY);
        semantic_binary = (255. * (semantic_binary>0)).astype('uint8')        
        
        w2_show = cv2.imread(w2_path, cv2.IMREAD_COLOR)
        w2_show = cv2.normalize(w2_show, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_shape = semantic_binary.shape

        width, height = img_shape[0], img_shape[1]


        unit = {}
        polygon_semantic = mask_to_pologons(semantic_binary)
        polygon_semantic_test = pologons_to_mask(polygon_semantic, semantic_binary.shape)



        _, contours,_ = cv2.findContours(np.uint8(this_isntance_mask),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            px = contours[0].squeeze()[:,0]
        except:
            print('invalid')
            continue
        py = contours[0].squeeze()[:,1]

        bbox = [int(np.min(px)),int(np.min(py)),int(np.max(px) - np.min(px)), int(np.max(py) - np.min(py))]

        end_point, control_point, off_sets_prev, off_sets_next= get_keypoints(this_isntance_mask>0, w2_show, step = step_size, crop_edge = crop_edge, debug = False)
        
        if len(control_point) < 1:
            print('no good')
            continue
        control_points_label = np.reshape(np.asarray(control_point), (-1, 2))
        end_points_label = np.reshape(np.asarray(end_point), (-1, 2))
        

        sequence_x = control_points_label[:,0]
        sequence_y = control_points_label[:,1] 
        unit['control_points'] = control_points_label.tolist()
        unit['endpoints'] = end_points_label.tolist()
        unit['seq_x_col'] = control_points_label[:,0].tolist()
        unit['seq_y_row'] = control_points_label[:,1].tolist()                
        

        sequence_x_offset = np.zeros(len(sequence_x))
        sequence_y_offset = np.zeros(len(sequence_y))

        sequence_x_offset[:-1] = sequence_x[1:] - sequence_x[:-1] 
        sequence_y_offset[:-1] = sequence_y[1:] - sequence_y[:-1] 

        sequence_offset = np.stack((sequence_x_offset, sequence_y_offset)).transpose(0,1)
        control_points_of_current_curve= [unit['seq_x_col'],unit['seq_y_row']]
        
        # offset_map = compute_offsets((width, height),sequence_offset,control_points_of_current_curve,\
        #              disc_radius = 25)
        offset_map = 0

        ###########
        #visualize
        ###########
        # test_offset_map = np.load(offset_matrix_normal_dir + '/' + str(anno_id) + '.npy')
        # # test_offset_map = offset_map

        # image = cv2.rectangle(this_image, (int(bbox[0]),int(bbox[1])), (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), (0,0,255), 2)
        # cv2.imshow('this_image', this_image)
        
        # canvas = np.zeros((width, height))
        # canvas = visualize_offset(canvas, test_offset_map[0,:,:], test_offset_map[1,:,:])
        # # combined_show = draw_mask_color(np.tile(this_image,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])
        # combined_show = draw_mask(this_image, canvas, [0., 255., 0.])
        # cv2.imshow('short', combined_show)
        # canvas = np.zeros((width, height))
        # canvas = visualize_offset(canvas, test_offset_map[2,:,:], test_offset_map[3,:,:])
        # # combined_show = draw_mask_color(np.tile(this_image,(3,1,1)).transpose(1,2,0), canvas, [0., 255., 0.])
        # combined_show = draw_mask(this_image, canvas, [0., 255., 0.])

        # cv2.imshow('mid_offsets_map', combined_show)
        # cv2.waitKey(0)

        ###############
        #             #
        ###############
        ###########################
        unit['id'] = int(anno_id)
        unit['image_id'] = int(current_id)
        unit['end_points'] = end_point
        unit['control_point'] = control_point
        unit['segmentation'] = polygon_this_instance
        # unit['segmentation'] = mask_cropped

        unit['reverse'] = int(0)
        
        # unit['control_points'] = control_point
        # unit['off_sets_prevs'] = off_sets_prev
        # unit['off_sets_nexts'] = off_sets_next

        unit['bbox'] =  bbox
        unit['area'] =  int(bbox[2] * bbox[3])
        unit['iscrowd'] =  0
        unit['category_id'] =  1

        annotations.append(unit)
        anno_id += 1  

        ###########
        #visualize
        ###########
        # from copy import deepcopy
        # canvas = deepcopy(w2_show)

        # # cv2.imshow('canvas', canvas)
        # # cv2.waitKey(0)
        # # import pdb; pdb.set_trace()
        # canvas = canvas[:,:]
        # canvas = canvas[crop_edge:canvas.shape[0] - crop_edge, crop_edge : canvas.shape[1] - crop_edge, :]
        
        # sequence_x[:len(unit['seq_x_col'])] = (np.asarray(unit['seq_x_col']))

        # sequence_y[:len(unit['seq_y_row'])] = (np.asarray(unit['seq_y_row']))
        # sequence_x_offset = np.zeros(len(sequence_x))
        # sequence_y_offset = np.zeros(len(sequence_y))

        # sequence_x_offset[:-1] = sequence_x[1:] - sequence_x[:-1] 
        # sequence_y_offset[:-1] = sequence_y[1:] - sequence_y[:-1] 

        # for pp in tqdm(range(len(sequence_x))):
        #     # import pdb; pdb.set_trace()
        #     if pp == 0:
        #         pre = (int(sequence_x[pp]), int(sequence_y[pp]))
        #         canvas = cv2.circle(canvas, (int(sequence_x[pp]), int(sequence_y[pp])), 5, (200,0,0), 1)
        #         continue

        #     cc = 0
        #     if sequence_x[pp] == 9999:
        #         break

        #     canvas = cv2.circle(canvas, (pre[0] + int(sequence_x_offset[pp-1]), pre[1] + int(sequence_y_offset[pp-1])), 5, (200,0,0), 1)
        #     # canvas = cv2.circle(canvas, (int(sequence_x[pp]), int(sequence_y[pp])), 5, (200,0,0), 1)
        #     pre = (pre[0] + int(sequence_x_offset[pp-1]), pre[1] + int(sequence_y_offset[pp-1]))
        #     cv2.imshow('test', canvas)
        #     cv2.waitKey()
        ###############
        #             #
        ###############

        if reverse_seq:
            unit = {}
            unit['control_points'] = control_points_label[::-1].tolist()
            unit['endpoints'] = end_points_label[::-1].tolist()
            unit['seq_x_col'] = control_points_label[::-1,0].tolist()
            unit['seq_y_row'] = control_points_label[::-1,1].tolist()      

            unit['id'] = int(anno_id)
            unit['image_id'] = int(current_id)
            unit['end_points'] = end_point
            unit['control_point'] = control_point
            unit['segmentation'] = polygon_this_instance
            # unit['segmentation'] = mask_cropped

            unit['reverse'] = int(1)
            

            unit['bbox'] =  bbox
            unit['area'] =  int(bbox[2] * bbox[3])
            unit['iscrowd'] =  0
            unit['category_id'] =  1
                                    
            offset_map = 0


            ###########
            #visualize
            ###########
            # from copy import deepcopy
            # canvas = deepcopy(this_image)
            # canvas = canvas[:,:,:3]
            # canvas = canvas[crop_edge:canvas.shape[0] - crop_edge, crop_edge : canvas.shape[1] - crop_edge, :]
            
            # sequence_x[:len(unit['seq_x_col'])] = (np.asarray(unit['seq_x_col']))

            # sequence_y[:len(unit['seq_y_row'])] = (np.asarray(unit['seq_y_row']))
            # sequence_x_offset = np.zeros(len(sequence_x))
            # sequence_y_offset = np.zeros(len(sequence_y))

            # sequence_x_offset[:-1] = sequence_x[1:] - sequence_x[:-1] 
            # sequence_y_offset[:-1] = sequence_y[1:] - sequence_y[:-1] 

            # for pp in tqdm(range(len(sequence_x))):
            #     # import pdb; pdb.set_trace()
            #     if pp == 0:
            #         pre = (int(sequence_x[pp]), int(sequence_y[pp]))
            #         canvas = cv2.circle(canvas, (int(sequence_x[pp]), int(sequence_y[pp])), 5, (200,0,0), 1)
            #         continue

            #     cc = 0
            #     if sequence_x[pp] == 9999:
            #         break

            #     canvas = cv2.circle(canvas, (pre[0] + int(sequence_x_offset[pp-1]), pre[1] + int(sequence_y_offset[pp-1])), 5, (200,0,0), 1)
            #     # canvas = cv2.circle(canvas, (int(sequence_x[pp]), int(sequence_y[pp])), 5, (200,0,0), 1)
            #     pre = (pre[0] + int(sequence_x_offset[pp-1]), pre[1] + int(sequence_y_offset[pp-1]))
            #     cv2.imshow('test', canvas)
            #     cv2.waitKey()                    
            ###############
            annotations.append(unit)
            anno_id += 1  

        for i in range(len(end_point)):
            end_point[i] = int(end_point[i])
            # start_points_offsets[i] = int(start_points_offsets[i])

        for i in range(len(control_point)):
            control_point[i] = int(control_point[i])
            # off_sets_prev[i] = int(off_sets_prev[i])
            # off_sets_next[i] = int(off_sets_next[i])     

        end_points += end_point
        control_points += control_point

                # off_sets_prevs += off_sets_prev
                # off_sets_nexts += off_sets_next




        if new_image:
            img_info ={}


            img_info['id'] = int(current_id)
            img_info['w1'] = w1
            img_info['w2'] = w2
            img_info['binary'] = binary_image_name
            img_info['file_path'] = file_path
            img_info['width'] = int(width)
            img_info['height'] = int(height)
            img_info['flickr_url'] = ''
            img_info['coco_url'] = ''
            img_info['date_captured'] = 'none'
            img_info['cropped_edge'] = int(crop_edge)
            img_info['segmentaion'] = polygon_semantic


            images.append(img_info)

    print('saving transformed annotation...')
    train_data['images'] = images
    train_data['annotations'] = annotations        
    with open(train_anno,'w') as wf:
        json.dump(train_data, wf)
        # json.dumps(coco_fiber.anns, wf)
        # json.dumps(coco_fiber.cats, wf)
        # json.dumps(coco_fiber.imgs, wf)
    print('done')



if __name__ == '__main__':
    # cnt_overlap_number()
    # root = '/opt/FTE/users/chrgao/datasets/Textile/data_for_zfb/new_data/mian_zhuma_hard_data/selected_img/'
    # root = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/instance_labeled_file'
#################################################################################################
    # folders = [x[0] for x in os.walk('//opt/intern/users/yiliu/instance_labeled_file/')]
    # for i in range(2, len(folders),3):
    #     img_root = folders[i]
    #     anno_root = folders[i + 1] + '/'
    #     output_root = anno_root

    #     ori_file = 'ann_common.json'
    #     target_file = 'ann_keypoint_double_offset_test.json'
    #     # import pdb; pdb.set_trace()

    #     trans_anno(img_root, anno_root, output_root, ori_file, target_file)

# #######################################################
    # root = '/home/****/work/colab/work/fiberPJ/data/fiber_labeled_data/instance_labeled_file'
    # # coco_anno_file = '/opt/FTE/users/chrgao/datasets/Textile/data_annotation/common_instance_json_format/fb_coco_style_fifth.json'
    # # coco_anno_file = '/home/****/work/colab/work/fiberPJ/data/fiber_labeled_data/fb_coco_style_fifth.json'
    # # coco_anno_file = '/home/****/work/fiberPJ/data/fiber_labeled_data/skel_5_coco_style_fifth.json'
    # # show_anno(root, coco_anno_file)
    # anno_root = '/home/****/work/colab/work/fiberPJ/data/fiber_labeled_data'
    # output_root = anno_root
    # img_root = root
    # ori_file = 'fb_coco_style_fifth.json'
    # # target_file = 'skel_1_non_coco_fifth.json'
    # offset_matrix_normal_dir = 'ann_keypoint_sequence_offset'
    # target_file = 'ann_keypoint_sequence.json'
# #######################################################
    # img_root = '/home/****/work/colab/work/fiberPJ/data/worm_data/BBBC010_v1_foreground_eachworm'
    img_root = '/home/****/work/colab/work/fiberPJ/data/worm_data/train'
    img_root = '/home/****/work/colab/work/fiberPJ/data/worm_data/test'
    binary_root = '/home/****/work/colab/work/fiberPJ/data/worm_data/BBBC010_v1_foreground'
    rog_image_root = '/home/****/work/colab/work/fiberPJ/data/worm_data/BBBC010_v2_images'


    # target_file = 'skel_1_non_coco_fifth.json'
    output_root = '/home/****/work/colab/work/fiberPJ/data/worm_data/'
    # target_file = 'worm_ann_keypoint_sequence.json'
    target_file = 'worm_ann_keypoint_sequence_train.json'
    target_file = 'worm_ann_keypoint_sequence_test_one_way.json'
    target_file = 'worm_ann_keypoint_sequence_test_step_60.json'
    target_file = 'worm_ann_keypoint_sequence_test_step_30.json'
    # target_file = 'worm_ann_keypoint_sequence_test_step_15.json'

    trans_anno(img_root, binary_root, rog_image_root, \
         output_root, target_file, reverse_seq = True)
