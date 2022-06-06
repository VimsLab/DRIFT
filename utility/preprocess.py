import numpy as np

import cv2

from utility.color_map import GenColorMap

from numba import jit, float32, int32
import numpy as np    
    

@jit(float32[:, :](float32[:, :], float32[:, :], int32[:, :], int32[:, :], float32), nopython=True, fastmath=True)
def apply_gaussian(accumulate_confid_map, centers, xx, yy, sigma):
    for i in range(len(centers)):
        center = centers[i]
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        cofid_map = np.exp(-exponent)
        cofid_map = np.multiply(mask, cofid_map)
        accumulate_confid_map += cofid_map
    return accumulate_confid_map

def gen_gaussian_map(centers, shape, sigma):

    centers = np.float32(centers)
    sigma = np.float32(sigma)
    accumulate_confid_map = np.zeros(shape, dtype=np.float32)
    y_range = np.arange(accumulate_confid_map.shape[0], dtype=np.int32)
    x_range = np.arange(accumulate_confid_map.shape[1], dtype=np.int32)
    xx, yy = np.meshgrid(x_range, y_range)

    accumulate_confid_map = apply_gaussian(accumulate_confid_map, centers, xx, yy, sigma)
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    
    return accumulate_confid_map

def draw_mask(im, mask, color):
    import copy
    temp = copy.deepcopy(im)
    im = cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    mask = mask>0
    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.uint8)

def draw_mask_color(im, mask, color):
    mask = mask>0.02

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32)*0.5
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.uint8)

def visualize_label_map(im, label_map):
    CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)


    ids = np.unique(label_map)

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)
    # import pdb; pdb.set_trace()

    for idx in ids:
        if idx == 0:
            continue
        color = CLASS_2_COLOR[int(idx) + 1]

        r[label_map==idx] = color[0]
        g[label_map==idx] = color[1]
        b[label_map==idx] = color[2]

    combined = cv2.merge([r, g, b]) * 0.75+ im.astype(np.float32) * 0.25
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.uint8)

def visualize_skel(input_image, skels):
    CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)
    canvas = input_image

    for i in range(len(skels)):
        one_canvas = np.zeros(input_image.shape[:2])
        color = CLASS_2_COLOR[i * 2 + 1]
        for ii in range(len(skels[i])):

            x = skels[i][ii]['xy'][0]
            y = skels[i][ii]['xy'][1]
            curr = (x, y)

            one_canvas = cv2.circle(one_canvas, curr, 5, 1, 3)

        canvas = draw_mask_color(canvas, one_canvas, color)

    return canvas

def visualize_skel_by_offset_and_tag(input_image, skels, keypoints, endpoints):
    CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)
    canvas_skel = input_image
    canvas_instance = input_image
    num = 0
    for i in range(len(skels)):

        canvas_instance = input_image
        print(num)
        if skels[i]['activate'] == True:
            one_canvas = np.zeros(input_image.shape[:2])
            color = CLASS_2_COLOR[num * 2 + 1]
            x = skels[i]['start_point']['xy'][0]
            y = skels[i]['start_point']['xy'][1]
            curr = (x, y)
            # one_canvas = cv2.circle(one_canvas, curr, 5, 1, 3)
            # canvas = draw_mask_color(canvas, one_canvas, color)
            # if skels[i]['one_point'] == False:
                # print(skels[i])
            print(skels[i]['this_skel_indexes'])
            print('--------------------------------------')
            for ii in range(len(skels[i]['this_skel_indexes'])):
                # import pdb;pdb.set_trace()
                ind = skels[i]['this_skel_indexes'][ii]
                one_canvas[keypoints[ind]['area']] = i + 1
                x = keypoints[ind]['xy'][0]
                y = keypoints[ind]['xy'][1]
                curr = (x, y)
                # print(curr)
                # # one_canvas = cv2.circle(canvas, curr, 5, 1, 3)
                # canvas = draw_mask_color(canvas, one_canvas, color)
                canvas_skel = cv2.circle(canvas_skel, curr, 5, color, 3)

            # draw start point
            ind_start = skels[i]['start_point_ind']
            x = endpoints[ind_start]['xy'][0]
            y = endpoints[ind_start]['xy'][1]

            canvas_skel = cv2.circle(canvas_skel, (x,y), 30, color, 1)
            one_canvas[endpoints[ind_start]['area']] = i + 1
            # draw end point

            ind = skels[i]['end_point_ind']
            if ind != -1:
                x = endpoints[ind]['xy'][0]
                y = endpoints[ind]['xy'][1]
                canvas_skel = cv2.circle(canvas_skel, (x,y), 10, color, 3)
                one_canvas[endpoints[ind]['area']] = i + 1

            canvas_instance = draw_mask_color(canvas_instance, one_canvas, color)
            cv2.imshow('t', canvas_skel)
            cv2.imshow('tt',canvas_instance)
            cv2.waitKey(0)
            num = num + 1
    return canvas_skel, canvas_instance


def visualize_offset(canvas, offset_h, offset_v):
    for x in range(0, canvas.shape[1], 5):
        for y in range(0, canvas.shape[0], 5):
            curr = (x, y)
            offset_x = offset_h[y, x]
            offset_y = offset_v[y, x]

            next_pt = (int(x + offset_x), int(y + offset_y))
            cv2.arrowedLine(canvas, curr, next_pt, 1, 1)

    return canvas
#
def normalize_include_neg_val(tag):

    # Normalised [0,255]
    normalised = 1.*(tag - np.min(tag))/np.ptp(tag).astype(np.float32)

    return normalised

def visualize_points(canvas, points_map):

    canvas[np.where(points_map>0)] = 1
    return canvas

def visualize_keypoint(canvas, keypoints):
    for i in range(len(keypoints)):
        curr = (keypoints[i]['xy'][0],keypoints[i]['xy'][1])

        cv2.circle(canvas, curr, 3, 1, -1)
    return canvas

def create_position_index(height, width):
    """
    create width x height x 2  pixel position indexes
    each position represents (x,y)
    """
    position_indexes = np.rollaxis(np.indices(dimensions=(width, height)), 0, 3).transpose((1,0,2))
    return position_indexes


def get_keypoint_discs_offset(all_keypoints, offset_map_point, img_shape, radius):

    #WHY NOT JUST USE IMDILATE
    #TO DO: USE discs, then use the offsets map(single point), find the value. then Map back to discs.
    map_shape = (img_shape[0], img_shape[1])
    offset_map_circle = np.zeros(map_shape)
    offset_map_circle_debug = np.zeros(map_shape)
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    discs = [[] for _ in range(len(all_keypoints))]
    # centers is the same with all keypoints.
    # Will change later.
    centers = all_keypoints
    dists = np.zeros(map_shape+(len(centers),))
    for k, center in enumerate(centers):
        dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1)) #Return the distance map to the point.
    # import pdb; pdb.set_trace()
    if len(centers) > 0:
        try:
            inst_id = dists.argmin(axis=-1)   #To which points its the closest
        except:
            print ('argmin fail')
            import pdb; pdb.set_trace()
    count = 0
    for i in range(len(all_keypoints)):

        discs[i].append(np.logical_and(inst_id == count, dists[:,:,count]<= radius))
        # offset_map_circle_debug[discs[i][0]] = 1.0
        offset_map_circle[discs[i][0]] = offset_map_point[dists[:,:,count] == 0]
        count +=1
    # tmp = np.asarray(offset_map_circle_debug * 255.0)
    # cv2.imshow('t', tmp)
    # cv2.waitKey(0)
    return discs, offset_map_circle

def get_keypoint_discs(all_keypoints, img_shape, radius):

    map_shape = (img_shape[0], img_shape[1])
    offset_map_circle = np.zeros(map_shape)
    offset_map_circle_debug = np.zeros(map_shape)
    # import pdb; pdb.set_trace()
    idx = create_position_index(height = map_shape[0], width = map_shape[1])
    # idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    # discs = [[] for _ in range(len(all_keypoints))]
    discs = []
    # centers is the same with all keypoints.
    # Will change later.
    centers = all_keypoints
    dists = np.zeros(map_shape+(len(centers),))
    for k, center in enumerate(centers):
        dists[:,:,k] = np.sqrt(np.square(center-idx).sum(axis=-1)) #Return the distance map to the point.
    if len(centers) > 0:
        inst_id = dists.argmin(axis=-1)   #To which points its the closest
    count = 0
    for i in range(len(all_keypoints)):

        discs.append(np.logical_and(inst_id == count, dists[:,:,count]<= radius))

        count +=1

    return discs


def compute_short_offsets(all_keypoints, discs, map_shape, radius):

    r = radius
    x = np.tile(np.arange(r, -r - 1, -1), [2 * r + 1, 1])
    y = x.transpose()
    m = np.sqrt(x*x + y*y) <= r
    kp_circle = np.stack([x, y], axis=-1) * np.expand_dims(m, axis=-1)

    def copy_with_border_check(map, center, disc):
        from_top = max(r-center[1], 0)
        from_left = max(r-center[0], 0)
        from_bottom = max(r-(map_shape[0]-center[1])+1, 0)
        from_right =  max(r-(map_shape[1]-center[0])+1, 0)
        # import pdb;pdb.set_trace()
        cropped_disc = disc[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right]
        map[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right, :][cropped_disc,:] = kp_circle[from_top:2*r+1-from_bottom, from_left:2*r+1-from_right, :][cropped_disc,:]

    offsets = np.zeros(map_shape+(2,)) #x offeset, y offset
    # import pdb;pdb.set_trace()
    for i in range(len(all_keypoints)):
        copy_with_border_check(offsets[:,:,0:2], (all_keypoints[i,0], all_keypoints[i,1]), discs[i])
                                                 # x col               # y, row
    canvas = np.zeros_like(offsets[:,:,0])

    canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])

    return offsets, canvas

def compute_short_offsets_as_patches(all_keypoints, max_len, discs, map_shape, radius):
    '''
    The first point will not be included
    beacuse the gournd truth of first point
    will the the short offset of the second point
    '''
    r = radius
    x = np.tile(np.arange(r, -r - 1, -1), [2 * r + 1, 1])
    y = x.transpose()
    m = np.sqrt(x*x + y*y) <= r
    kp_circle = np.stack([x, y], axis=-1) * np.expand_dims(m, axis=-1)

    def copy_with_border_check(map, center, disc):
        from_top = max(r-center[1], 0)
        from_left = max(r-center[0], 0)
        from_bottom = max(r-(map_shape[0]-center[1])+1, 0)
        from_right =  max(r-(map_shape[1]-center[0])+1, 0)
        # import pdb;pdb.set_trace()
        cropped_disc = disc[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right]
        map[center[1]-r+from_top:center[1]+r+1-from_bottom, center[0]-r+from_left:center[0]+r+1-from_right, :][cropped_disc,:] = kp_circle[from_top:2*r+1-from_bottom, from_left:2*r+1-from_right, :][cropped_disc,:]

    offsets = np.zeros((max_len, )+map_shape+(2,)) #x offeset, y offset

    for i in range(1, len(all_keypoints)):
        copy_with_border_check(offsets[i-1,:,:,0:2], (all_keypoints[i,0], all_keypoints[i,1]), discs[i])
                                                 # x col               # y, row
    return offsets

def compute_mid_offsets(all_keypoints, offset_map_h, offset_map_v, map_shape, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes

    offsets = np.zeros(map_shape+(2,))
    canvas = np.zeros_like(offsets[:,:,0])
    for k, center in enumerate(all_keypoints):
        # import pdb;pdb.set_trace()
        next_point_h = center[0] + offset_map_h[(center[1],center[0])]
        next_point_v = center[1] + offset_map_v[(center[1],center[0])]

        next_point_center = (int(next_point_h), int(next_point_v))
        curr = (int(center[0]), int(center[1]))
        #####
        # debug
        # cv2.arrowedLine(canvas, curr, next_point_center, 1, 1)
        #####
        # m = discs[i]

        # import pdb;pdb.set_trace()
        dists = next_point_center - idx
        offsets[discs[k],0] = dists[discs[k],0]
        offsets[discs[k],1] = dists[discs[k],1]
        ######################
        # debug
        # canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
        # cv2.imshow('t', canvas)
        # cv2.waitKey(0)
        ######################
        # import pdb;pdb.set_trace()
    return offsets, canvas

def compute_closest_control_point_offset(all_keypoints, seg_mask, map_shape):
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    direct_dists = np.zeros(map_shape + (len(all_keypoints),))

    offsets = np.zeros(map_shape+(2,))
    offsets_tmp = np.zeros(map_shape+(2,))
    offsets_h = np.zeros(map_shape + (len(all_keypoints),))
    offsets_v = np.zeros(map_shape + (len(all_keypoints),))

    seg_mask_index = seg_mask > 0
    canvas = np.zeros_like(offsets[:,:,0])

    for k, center in enumerate(all_keypoints):
        curr = (int(center[0]), int(center[1]))

        dists = curr - idx
        offsets_tmp[seg_mask_index,0] = dists[seg_mask_index,0]
        offsets_tmp[seg_mask_index,1] = dists[seg_mask_index,1]

        offsets_h[seg_mask_index,k] = dists[seg_mask_index,0]
        offsets_v[seg_mask_index,k] = dists[seg_mask_index,1]

        # canvas = visualize_offset(canvas, offsets_h[:,:,k], offsets_v[:,:,k])
        # cv2.imshow('t', canvas)
        # cv2.waitKey(0)

        direct_dists[:,:,k] = np.sqrt(np.sum(np.square(offsets_tmp), axis = 2)) # obtain the shortest dist to the control point


    try:
       closest_keypoints = np.argmin(direct_dists, axis=2)
       # import pdb;pdb.set_trace()
    except:
        import pdb; pdb.set_trace()
        print (direct_dists.shape)
        print ('argmin fail')



    closest_keypoints = closest_keypoints.flatten() # flatten (indices of the last dimension)

    ind = (np.arange(len(closest_keypoints)), closest_keypoints) # create indexes

    offsets_flatten_h = np.reshape(offsets_h, (-1, len(all_keypoints)))

    offsets_h_final = offsets_flatten_h[ind]

    offsets_flatten_v = np.reshape(offsets_v, (-1, len(all_keypoints)))

    offsets_v_final = offsets_flatten_v[ind]

    offsets_h_final = np.reshape(offsets_h_final, map_shape)
    offsets_v_final = np.reshape(offsets_v_final, map_shape)
    # import pdb;pdb.set_trace()

    offsets[:,:,0] = offsets_h_final
    offsets[:,:,1] = offsets_v_final
    # import pdb;pdb.set_trace()
    # canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
    # cv2.imshow('t', canvas)
    # cv2.waitKey(0)

    return offsets, canvas

def compute_mid_long_offsets(all_keypoints, offset_map_h, offset_map_v, map_shape, discs):
    # map_shape = (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes
    # import pdb;pdb.set_trace()

    dists = np.zeros(map_shape+(len(all_keypoints),))
    offsets = np.zeros(map_shape+(2,))
    canvas = np.zeros_like(offsets[:,:,0])
    for k, center in enumerate(all_keypoints):
        # import pdb;pdb.set_trace()
        next_point_h = center[0] - offset_map_h[(center[1],center[0])]
        next_point_v = center[1] - offset_map_v[(center[1],center[0])]
        next_next_point_h = next_point_h - offset_map_h[(int(next_point_v),int(next_point_h))]
        next_next_point_v = next_point_v - offset_map_v[(int(next_point_v),int(next_point_h))]


        next_next_point_center = (int(next_next_point_h), int(next_next_point_v))
        curr = (int(center[0]), int(center[1]))
        cv2.arrowedLine(canvas, curr, next_next_point_center, 1, 1)
        # m = discs[i]

        # import pdb;pdb.set_trace()
        dists = next_next_point_center - idx

        offsets[discs[k],0] = dists[discs[k],0]
        offsets[discs[k],1] = dists[discs[k],1]

    canvas = visualize_offset(canvas, offsets[:,:,0], offsets[:,:,1])
    # cv2.imshow('t', canvas)
    # cv2.waitKey(0)
        # import pdb;pdb.set_trace()
    return offsets, canvas



def sequence_patches(image, max_len, sequence_offset, segmentation, control_points, \
                    patch_size, disc_radius):
    # shape: image 3; [segmentaiton:1, contro_points_label: 1 ; short_offset:2 ; next_point:2] 
    channel = 6
    image_shape = (image.shape[0],image.shape[1])
    patches_seqeunces =np.zeros((max_len, channel, patch_size, patch_size))

    image_patch_sequence = np.zeros((max_len, patch_size, patch_size, 3))

    patches_input =np.zeros((max_len, channel, image.shape[0], image.shape[1]))
    
    canvas = np.zeros((channel, image.shape[0], image.shape[1]))

    control_points = np.transpose(control_points)
    control_points_label = np.reshape(np.asarray(control_points), (-1, 2))

    control_points_discs = get_keypoint_discs(control_points_label, image_shape, radius = disc_radius)
    
    short_offset_patches = compute_short_offsets_as_patches(control_points_label, max_len, control_points_discs, image_shape, disc_radius)
    control_point_disc_patches = np.zeros((max_len,) + image_shape) # [max_len, H,W]

    for idx in range(1, len(control_points_discs)):
        control_point_disc_patches[idx - 1,:,:] = control_points_discs[idx]

    control_points_nexts_offset = np.zeros((max_len,) + image_shape + (2,), dtype = np.float32)
    for idx, points_coor in enumerate(control_points):
        off_sets_nexts_map_h = np.zeros(image_shape, dtype = np.float32)
        off_sets_nexts_map_v = np.zeros(image_shape, dtype = np.float32)
        if idx == len(control_points) - 1:
            off_sets_nexts_map_h[points_coor[1], points_coor[0]] = 0
            off_sets_nexts_map_v[points_coor[1], points_coor[0]] = 0       
            control_points_nexts_offset[idx, :,:,:], _ = \
                compute_mid_offsets([control_points[idx]], off_sets_nexts_map_h, off_sets_nexts_map_v, image_shape, [control_points_discs[idx]])     
        else:
            off_sets_nexts_map_h[points_coor[1], points_coor[0]] = sequence_offset[idx][0]
            off_sets_nexts_map_v[points_coor[1], points_coor[0]] = sequence_offset[idx][1]      
            control_points_nexts_offset[idx, :,:,:], _ = \
                compute_mid_offsets([control_points[idx]], off_sets_nexts_map_h, off_sets_nexts_map_v, image_shape, [control_points_discs[idx]])     

    padding_size = patch_size//2 + patch_size // 4
    # input_batch = np.pad(input=input_batch, pad=(padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
    control_points_nexts_offset = np.pad(control_points_nexts_offset, [(0,0),(padding_size, padding_size) , (padding_size, padding_size), (0,0)], mode='constant', constant_values=0)
    short_offset_patches = np.pad(short_offset_patches, [(0,0),(padding_size, padding_size) , (padding_size, padding_size), (0,0)], mode='constant', constant_values=0)
    control_point_disc_patches = np.pad(control_point_disc_patches, [(0,0),(padding_size, padding_size) , (padding_size, padding_size)], mode='constant', constant_values=0)
    segmentation_padded = np.pad(segmentation, [(padding_size, padding_size) , (padding_size, padding_size)], mode='constant', constant_values=0)
    
    image_padded = np.pad(image, [(padding_size, padding_size) , (padding_size, padding_size), (0,0)], mode='constant', constant_values=0)
    for i in range(len(control_points)):

        x_coor_st = int(control_points[i][0].item() - patch_size//2 + padding_size)
        x_coor_ed = int(control_points[i][0].item() + patch_size//2 + padding_size)
        y_coor_st = int(control_points[i][1].item()- patch_size//2 + padding_size)
        y_coor_ed = int(control_points[i][1].item() + patch_size//2 + padding_size)

        #seg
        patches_seqeunces[i, 0,:, :] = segmentation_padded[y_coor_st :y_coor_ed, x_coor_st :x_coor_ed]
        #disc
        # patches_seqeunces[i, 1,:, :] = control_point_disc_patches[i, y_coor_st :y_coor_ed, x_coor_st :x_coor_ed]
        #short
        patches_seqeunces[i, 1:3,:, :] = np.transpose(short_offset_patches[i, y_coor_st :y_coor_ed, x_coor_st :x_coor_ed,:]\
                                                    ,(2,0,1))
        #mid
        patches_seqeunces[i, 3:5,:, :] = np.transpose(control_points_nexts_offset[i, y_coor_st :y_coor_ed, x_coor_st :x_coor_ed,:]\
                                                    ,(2,0,1))
        image_patch_sequence[i,:,:,:] = image_padded[y_coor_st :y_coor_ed, x_coor_st :x_coor_ed,:]
       
        ##########################################
        # cv2.imshow('control_points_map', control_point_disc_patches[i,:,:] * 255.)

        # canvas = np.zeros(image_shape)
        # canvas = visualize_offset(canvas, short_offset_patches[i,:,:,0], short_offset_patches[i,:,:,1])
        # canvas = visualize_offset(canvas, control_points_nexts_offset[i,:,:,0], control_points_nexts_offset[i,:,:,1])
        # combined_show = draw_mask_color(image, canvas, [0., 255., 0.])
        # cv2.imshow('short_and_mid_offsets_map', combined_show)
        # cv2.waitKey(0)
        ##################################################
    # cv2.imshow('seg', segmentation * 255.)

    # for i in range(len(control_points)):
    #     ##########################################
    #     # cv2.imshow('control_points_map', patches_seqeunces[i, 0,:, :] * 255.)
    #     cv2.imshow('segmentation', patches_seqeunces[i, 0,:, :] * 255.)
    #     cv2.imshow('image_patch_sequence', image_patch_sequence[i,:,:,:])

    #     # out_folder = '/home/****/work/colab/work/fiberPJ/a-PyTorch-Tutorial-to-Image-Captioning/demo_output/lstmdemo/'

    #     # cv2.imwrite(out_folder+str(i) + '_image_patch_sequence_worm.png', image_patch_sequence[i,:,:,:])
        
    #     canvas = np.zeros((patch_size, patch_size))
    #     canvas = visualize_offset(canvas, patches_seqeunces[i, 1,:, :], patches_seqeunces[i, 2,:, :])
    #     combined_show = draw_mask_color(image_patch_sequence[i,:,:,:] , canvas, [0., 255., 0.])
    #     combined_show = cv2.circle(combined_show, (patch_size // 2, patch_size//2), 3, [0.,255., 0], 2)
    #     cv2.imshow('short_offsets_map', combined_show)
    #     # cv2.imwrite(out_folder+str(i) + '_short_offsets_map_worm.png', combined_show)
        
    #     canvas = np.zeros((patch_size, patch_size))
    #     canvas = visualize_offset(canvas, patches_seqeunces[i, 3,:, :], patches_seqeunces[i, 4,:, :])
    #     combined_show = draw_mask_color(image_patch_sequence[i,:,:,:] , canvas, [0., 255., 0.])
    #     combined_show = cv2.circle(combined_show, (patch_size // 2, patch_size//2), 3, [0.,255., 0], 2)
    #     cv2.imshow('mid_offsets_map', combined_show)
    #     # cv2.imwrite(out_folder+str(i) + '_mid_offsets_map_worm.png', combined_show)

    #     segmentation_show = draw_mask_color(image_patch_sequence[i,:,:,:] ,  patches_seqeunces[i, 0,:, :] * 255., [0., 0., 255.])
    #     # cv2.imshow('short_and_mid_offsets_map', combined_show)
    #     segmentation_show = cv2.circle(segmentation_show, (patch_size // 2, patch_size//2), 3, [0.,255., 0], 2)
    #     cv2.imshow('segmentation_show', segmentation_show)
        
    #     # cv2.imwrite(out_folder+str(i) + '_segmentation_show_worm.png', segmentation_show)
    #     cv2.waitKey(0)
    return patches_seqeunces



def sequence_segmentation_patches(image, max_len, sequence_offset, segmentation, control_points, \
                    patch_size, disc_radius):
    # shape: image 3; [segmentaiton:1, contro_points_label: 1 ; short_offset:2 ; next_point:2] 
    channel = 1
    image_shape = (image.shape[0],image.shape[1])
    patches_seqeunces =np.zeros((max_len, channel, patch_size, patch_size))

    image_patch_sequence = np.zeros((max_len, patch_size, patch_size, 3))

    patches_input =np.zeros((max_len, channel, image.shape[0], image.shape[1]))
    
    canvas = np.zeros((channel, image.shape[0], image.shape[1]))

    control_points = np.transpose(control_points)
 
    padding_size = patch_size//2 + patch_size // 4
    # input_batch = np.pad(input=input_batch, pad=(padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
    segmentation_padded = np.pad(segmentation, [(padding_size, padding_size) , (padding_size, padding_size)], mode='constant', constant_values=0)
    
    image_padded = np.pad(image, [(padding_size, padding_size) , (padding_size, padding_size), (0,0)], mode='constant', constant_values=0)
    for i in range(len(control_points)):

        x_coor_st = int(control_points[i][0].item() - patch_size//2 + padding_size)
        x_coor_ed = int(control_points[i][0].item() + patch_size//2 + padding_size)
        y_coor_st = int(control_points[i][1].item()- patch_size//2 + padding_size)
        y_coor_ed = int(control_points[i][1].item() + patch_size//2 + padding_size)

        #seg
        patches_seqeunces[i, 0,:, :] = segmentation_padded[y_coor_st :y_coor_ed, x_coor_st :x_coor_ed]
        #disc
        image_patch_sequence[i,:,:,:] = image_padded[ y_coor_st :y_coor_ed, x_coor_st :x_coor_ed,:]
       
        ##########################################
        # cv2.imshow('control_points_map', control_point_disc_patches[i,:,:] * 255.)

        # canvas = np.zeros(image_shape)
        # canvas = visualize_offset(canvas, short_offset_patches[i,:,:,0], short_offset_patches[i,:,:,1])
        # canvas = visualize_offset(canvas, control_points_nexts_offset[i,:,:,0], control_points_nexts_offset[i,:,:,1])
        # combined_show = draw_mask_color(image, canvas, [0., 255., 0.])
        # cv2.imshow('short_and_mid_offsets_map', combined_show)
        # cv2.waitKey(0)
        ##################################################
    # cv2.imshow('seg', segmentation * 255.)

    # for i in range(len(control_points)):
    #     ##########################################
    #     cv2.imshow('control_points_map', patches_seqeunces[i, 0,:, :] * 255.)
    #     cv2.imshow('segmentation', patches_seqeunces[i, 1,:, :] * 255.)

    #     canvas = np.zeros((patch_size, patch_size))
    #     canvas = visualize_offset(canvas, patches_seqeunces[i, 2,:, :], patches_seqeunces[i, 3,:, :])
    #     canvas = visualize_offset(canvas, patches_seqeunces[i, 4,:, :], patches_seqeunces[i, 5,:, :])
    #     combined_show = draw_mask_color(image_patch_sequence[i,:,:,:] , canvas, [0., 255., 0.])
    #     cv2.imshow('short_and_mid_offsets_map', combined_show)
    #     cv2.waitKey(0)
    return patches_seqeunces

def sequence_patches_with_offset_matrix(image, max_len, offset_matrix, segmentation, control_points, \
                    patch_size, disc_radius):
    # shape: image 3; [segmentaiton:1, contro_points_label: 1 ; short_offset:2 ; next_point:2] 

    channel = 5
    image_shape = (image.shape[0],image.shape[1])
    patches_seqeunces =np.zeros((max_len, channel, patch_size, patch_size))

    image_patch_sequence = np.zeros((max_len, patch_size, patch_size, 3))

    patches_input =np.zeros((max_len, channel, image.shape[0], image.shape[1]))
    
    canvas = np.zeros((channel, image.shape[0], image.shape[1]))

    control_points = np.transpose(control_points)
    # import pdb; pdb.set_trace()
    padding_size = patch_size//2 + patch_size // 4
    # input_batch = np.pad(input=input_batch, pad=(padding_size, padding_size, padding_size, padding_size), mode='constant', value=0)
    segmentation_padded = np.pad(segmentation, [(padding_size, padding_size) , (padding_size, padding_size)], mode='constant', constant_values=0)
    offset_matrix_padded = np.pad(offset_matrix, [(0,0), (padding_size, padding_size), (padding_size, padding_size)], mode='constant', constant_values=0)
    
    image_padded = np.pad(image, [(padding_size, padding_size) , (padding_size, padding_size), (0,0)], mode='constant', constant_values=0)
    
    center_mask = np.zeros((patch_size,patch_size))
    center_mask = cv2.circle(center_mask, (patch_size // 2, patch_size//2), disc_radius, 1, -1)

    for i in range(len(control_points)):

        x_coor_st = int(control_points[i][0].item() - patch_size//2 + padding_size)
        x_coor_ed = int(control_points[i][0].item() + patch_size//2 + padding_size)
        y_coor_st = int(control_points[i][1].item()- patch_size//2 + padding_size)
        y_coor_ed = int(control_points[i][1].item() + patch_size//2 + padding_size)

        #seg
        patches_seqeunces[i, 0,:, :] = segmentation_padded[y_coor_st :y_coor_ed, x_coor_st :x_coor_ed]

        #mid_offset
        patches_seqeunces[i, 3:5,:, :] = offset_matrix_padded[2:4, y_coor_st :y_coor_ed, x_coor_st :x_coor_ed] * center_mask

        #short_offset
        next_y = patches_seqeunces[i, 3,:, :][int(patch_size // 2)][int(patch_size // 2)]
        next_x = patches_seqeunces[i, 4,:, :][int(patch_size // 2)][int(patch_size // 2)]
        next_point_center_mask = np.zeros((patch_size,patch_size))
        next_point_center_mask = cv2.circle(next_point_center_mask, (patch_size // 2 + int(next_y), patch_size//2 + int(next_x)), disc_radius, 1, -1)
        
        patches_seqeunces[i, 1:3,:, :] = offset_matrix_padded[0:2, y_coor_st :y_coor_ed, x_coor_st :x_coor_ed] * next_point_center_mask
        #disc
        image_patch_sequence[i,:,:,:] = image_padded[y_coor_st :y_coor_ed, x_coor_st :x_coor_ed,:]
       
        ##########################################
        # cv2.imshow('control_points_map', control_point_disc_patches[i,:,:] * 255.)

        # canvas = np.zeros(image_shape)
        # canvas = visualize_offset(canvas, short_offset_patches[i,:,:,0], short_offset_patches[i,:,:,1])
        # canvas = visualize_offset(canvas, control_points_nexts_offset[i,:,:,0], control_points_nexts_offset[i,:,:,1])
        # combined_show = draw_mask_color(image, canvas, [0., 255., 0.])
        # cv2.imshow('short_and_mid_offsets_map', combined_show)
        # cv2.waitKey(0)
        ##################################################
    # cv2.imshow('seg', segmentation * 255.)

    # for i in range(len(control_points)):
    #     ##########################################
    #     cv2.imshow('image_padded',image_padded)
    #     cv2.imshow('segmentation', patches_seqeunces[i, 0,:, :] * 255.)
    #     # cv2.imshow('segmentation', patches_seqeunces[i, 1,:, :] * 255.)

    #     next_y = patches_seqeunces[i, 3,:, :][int(patch_size // 2)][int(patch_size // 2)]
    #     next_x = patches_seqeunces[i, 4,:, :][int(patch_size // 2)][int(patch_size // 2)]
    #     next_point_center_mask = np.zeros((patch_size,patch_size))
    #     next_point_center_mask = cv2.circle(next_point_center_mask, (patch_size // 2 + int(next_y), patch_size//2 + int(next_x)), disc_radius, 1, -1)

    #     cv2.imshow('short_m', next_point_center_mask)

    #     canvas = np.zeros((patch_size, patch_size))
    #     combined_show = visualize_offset(canvas, patches_seqeunces[i, 1,:, :], patches_seqeunces[i, 2,:, :])
    #     combined_show = draw_mask_color(image_patch_sequence[i,:,:,:] , combined_show, [0., 0., 255.])
    #     cv2.imshow('short', combined_show)

    #     canvas = np.zeros((patch_size, patch_size))
    #     canvas = visualize_offset(canvas, patches_seqeunces[i, 3,:, :], patches_seqeunces[i, 4,:, :])
    #     cv2.imshow('patch', image_patch_sequence[i,:,:,:].astype(np.uint8))
    #     combined_show = draw_mask_color(image_patch_sequence[i,:,:,:] , canvas, [0., 0., 255.])
    #     # combined_show = draw_mask(image_patch_sequence[i,:,:,:] , canvas, [0., 255., 0.])
    #     cv2.imshow('mid', combined_show)
    #     cv2.waitKey(0)
    # print(len(control_points))
    return patches_seqeunces