import numpy as np
import cv2

# import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import coo_matrix
# from utility.cv2_util import mask_to_pologons
# from cocoapi.PythonAPI.pycocotools.mask import toBbox

# from test_config import cfg
from utility.bilinear import bilinear_sampler
from utility.preprocess import get_keypoint_discs
from utility.preprocess import visualize_keypoint
from utility.preprocess import normalize_include_neg_val

def draw_mask_color(im, mask, color):
    mask = mask>0.02

    r = im[:, :, 0].astype(np.float32)
    g = im[:, :, 1].astype(np.float32)
    b = im[:, :, 2].astype(np.float32)

    # import pdb; pdb.set_trace()

    r[mask] = color[0]
    g[mask] = color[1]
    b[mask] = color[2]

    combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32)*0.5
    # combined = cv2.merge([r, g, b])

    return combined.astype(np.float32)

def refine(base,offsets,num_steps=1):
    for i in range(num_steps):

        base = base + bilinear_sampler(offsets,base)
    return base

def split_and_refine_mid_offsets(mid_offsets, short_offsets, num_steps = 2):
    mid_offsets_refine = mid_offsets
    # for mid_idx, edge in enumerate(config.EDGES+[edge[::-1] for edge in config.EDGES]):
    #     to_keypoint = edge[1]
    #     kp_short_offsets = short_offsets[:,:,:,2*to_keypoint:2*to_keypoint+2]
    #     kp_mid_offsets = mid_offsets[:,:,:,2*mid_idx:2*mid_idx+2]
    kp_mid_offsets = refine(mid_offsets_refine, short_offsets, num_steps)
    # output_mid_offsets.append(kp_mid_offsets)
    return kp_mid_offsets

def get_angle(x, y):
    Lx=np.sqrt(x.dot(x))
    Ly=np.sqrt(y.dot(y))
    cos_angle=x.dot(y)/(Lx*Ly)
    angle_radius=np.arccos(cos_angle)
    angle_degree=angle_radius*360/2/np.pi
    return angle_degree

def accumulate_votes(votes, shape):
    xs = votes[:,0]
    ys = votes[:,1]
    ps = votes[:,2]
    tl = [np.floor(ys).astype('int32'), np.floor(xs).astype('int32')]
    tr = [np.floor(ys).astype('int32'), np.ceil(xs).astype('int32')]
    bl = [np.ceil(ys).astype('int32'), np.floor(xs).astype('int32')]
    br = [np.ceil(ys).astype('int32'), np.ceil(xs).astype('int32')]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps*(1.-dx)*(1.-dy)
    tr_vals = ps*dx*(1.-dy)
    bl_vals = ps*dy*(1.-dx)
    br_vals = ps*dy*dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    heatmap = np.asarray(coo_matrix( (data[good_inds], (I[good_inds],J[good_inds])), shape=shape ).todense())
    return heatmap

def iterative_bfs(graph, start, path=[]):
    '''iterative breadth first search from start'''
    q=[(None,start)]
    visited = []
    while q:
        v=q.pop(0)
        if not v[1] in visited:
            visited.append(v[1])
            path=path+[v]
            q=q+[(v[1], w) for w in graph[v[1]]]
    return path

def compute_heatmaps(kp_maps, short_offsets, disc_radius = 15):

    map_shape = kp_maps.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))

    this_kp_map = kp_maps
    votes = idx + short_offsets.transpose((1,2,0))
    votes = np.reshape(np.concatenate([votes, this_kp_map[:,:,np.newaxis]], axis=-1), (-1, 3))
    heatmap = accumulate_votes(votes, shape=map_shape) / (np.pi*disc_radius**2)

    return heatmap
    
def compute_heatmaps_with_momentum(kp_maps, short_offsets, moemntum_map):

    map_shape = kp_maps.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))

    this_kp_map = kp_maps
    votes = idx + short_offsets.transpose((1,2,0))
    votes = np.reshape(np.concatenate([votes, this_kp_map[:,:,np.newaxis]], axis=-1), (-1, 3))
    heatmap = accumulate_votes(votes, shape=map_shape) / (np.pi*cfg.disc_radius**2)

    return heatmap

def compute_end_point_heatmaps(kp_maps, offsets_maps):

    map_shape = kp_maps.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2))


    this_kp_map = kp_maps
    # import pdb;pdb.set_trace()
    for i in range(len(offsets_maps)):
        # import pdb;pdb.set_trace()
        votes = idx + offsets_maps[i].transpose((1,2,0))
    votes = np.reshape(np.concatenate([votes, this_kp_map[:,:,np.newaxis]], axis=-1), (-1, 3))
    heatmap = accumulate_votes(votes, shape=map_shape) / (np.pi*cfg.disc_radius**2)

    return heatmap

def get_keypoints(heatmaps, tag_maps, threshold):
    # import pdb;pdb.set_trace()
    keypoints = []
    # peaks = maximum_filter(heatmaps, footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmaps
    peaks = maximum_filter(heatmaps, size = 3) == heatmaps
    # import pdb;pdb.set_trace()
    peaks = zip(*np.nonzero(peaks))
    ind = 0
    for peak in peaks:
        if heatmaps[peak[0], peak[1]] > threshold:
            keypoints.append({'ind' : ind, 'xy': np.array(peak[::-1]), 'conf': heatmaps[peak[0], peak[1]], \
                                'tag_val': tag_maps[peak[0], peak[1]], 'flag': False})
            ind = ind + 1

    # keypoints.extend([{'xy': np.array(peak[::-1]), 'conf': heatmaps[peak[0], peak[1]], 'tag_val': tag_maps[peak[0], peak[1]]} for peak in peaks])
    # keypoints = [kp for kp in keypoints if kp['conf'] > threshold]

    return keypoints

def get_next_point(this_point, offset_map, vis_each_step_idx= 1): # keypoints for debug

    visual = False
    keypoints = []
    x = this_point[1]
    y = this_point[0]
    temp_heatmap = np.zeros(offset_map.shape[1:])

    img_dilation = cv2.circle(temp_heatmap, (x ,y), 10, 1, -1)

    # img_prevouse_dilation = cv2.line(temp_heatmap, (x - int(displacement[0] * 1),y - int(displacement[1] * 1)), (x + int(displacement[0] * 0), int(y + displacement[1] * 0)), 1,15)
    # offset_map_momentum = np.roll(offset_map, int(displacement[1]), axis = 1)
    # offset_map_momentum = np.roll(offset_map_momentum, int(displacement[0]), axis = 2)
    # heatmap_momentum = compute_heatmaps(temp_heatmap, img_dilation)


    temp_heatmap[int(y),int(x)] = 1
    temp_heatmap = img_dilation

    if visual:
        out_folder = '/home/****/work/colab/work/fiberPJ/a-PyTorch-Tutorial-to-Image-Captioning/demo_output/worm_demo/'
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(img_dilation)
        cv2.imshow('rgba_img',rgba_img)

        cv2.imwrite(out_folder+str(vis_each_step_idx) + '_circle.png', rgba_img*255)

        cv2.waitKey(0)

    # print ('dilation')
    # print (np.max(img_dilation))
    # print (np.min(img_dilation))
    # print ('offset')
    # print (np.max(offset_map))
    # print (np.min(offset_map))
    # print (np.max(temp_heatmap))

    heatmap_next = compute_heatmaps(img_dilation, offset_map)
    if visual:
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(heatmap_next * 10)

    peaks = maximum_filter(heatmap_next, footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmap_next 

    # import pdb;pdb.set_trace()
    peaks = zip(*np.nonzero(peaks))
    ind = 0
    max_conf = 0
    for peak in peaks:
        if  heatmap_next[peak[0], peak[1]] > max_conf:
            max_conf = heatmap_next[peak[0], peak[1]]
            next_point =  np.array(peak[::-1])
            keypoints = next_point
        # if heatmap_next[peak[0], peak[1]] > 0.003:
        #     keypoints.append({'ind' : ind, 'xy': np.array(peak[::-1]), 'conf': heatmap_next[peak[0], peak[1]]})
        #     ind = ind + 1
    if visual:
        # cv2.circle(rgba_img,  (keypoints[0] ,keypoints[1]), 5, (0,255,255), 1)
        cv2.imshow('rgba_img',rgba_img)
        cv2.imwrite(out_folder+str(vis_each_step_idx) + '_heatmap.png', rgba_img*255)
        cv2.circle(rgba_img,  (keypoints[0] ,keypoints[1]), 5, (0,255,255), 1)

        cv2.imwrite(out_folder+str(vis_each_step_idx) + '_heatmap_circle.png', rgba_img*255)
        cv2.imshow('rgba_img_draw_circl',rgba_img)
        cv2.waitKey(0)
    # keypoints.extend([{'xy': np.array(peak[::-1]), 'conf': heatmaps[peak[0], peak[1]], 'tag_val': tag_maps[peak[0], peak[1]]} for peak in peaks])
    # keypoints = [kp for kp in keypoints if kp['conf'] > threshold]
    offset = np.asarray([keypoints[0] - offset_map.shape[1]//2, keypoints[1] - offset_map.shape[1]//2])
    # import pdb; pdb.set_trace()
    return offset

#     # heatmap_next = gaussian_filter(heatmap_next, sigma=5)
#     heatmap_next = heatmap_next
#     heatmap_next = gaussian_filter(heatmap_next, sigma=10)
#     if np.amax(heatmap_next) == 0:
#         return 0, 0
#     # print (np.max(heatmap_next))
#     # print (np.min(heatmap_next))

#     # control_point_tags_masked = control_point_tags_numpy * control_point_mask

#     normalised_tags = normalize_include_neg_val(heatmap_next)
#     next_point = normalised_tags * ((location_map > 0) * 1.0) # get valid values




#     which_point_coor = np.where(next_point == np.amax(next_point))

#     if np.amax(end_end_point) == 0:
#         which_end_point = -1
#     else:
#         which_end_point_coor = np.where(end_end_point == np.amax(end_end_point))
#         dists_L2 = np.sqrt((which_point_coor[0][0] - which_end_point_coor[0][0]) ** 2 + \
#                 (which_point_coor[1][0] - which_end_point_coor[1][0]) ** 2)

#         if dists_L2 < 10:
#             # print (dists_L2)
#             # print('emmm')

#             which_end_point = int(end_points_map[which_end_point_coor] - 1)

#         else:
#             which_end_point = -1

#     which_point = int(location_map[(which_point_coor[0][0], which_point_coor[1][0])] - 1) # index is from 0





#     # print (which_point)

#     # print (which_end_point)
# #################### debug #############################
#     def draw_mask(im, mask):
#         # import pdb; pdb.set_trace()
#         mask = mask > 0.02

#         r = im[:, :, 0].astype(np.float32)
#         g = im[:, :, 1].astype(np.float32)
#         b = im[:, :, 2].astype(np.float32)

#         r[mask] = 1.0
#         g[mask] = 1.0
#         b[mask] = 0.0

#         combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
#         return combined.astype(np.float32)

#     canvas = np.zeros(input_image.shape[0:2])

#     ##############debug
#     # cmap = plt.get_cmap('jet')
#     # rgba_img = cmap(normalised_tags)
#     # cv2.imshow('rgba_img',rgba_img)
#     # cv2.waitKey(0)

#     # cv2.circle(canvas, (which_point_coor[1][0],which_point_coor[0][0]), 2, (0,255,255), -1)
#     # cv2.circle(canvas, (x,y), 8, (0,0,255), -1)
#     # cv2.circle(canvas, (keypoints[which_point]['xy'][0],keypoints[which_point]['xy'][1]), 5, (255,155,255), -1)
#     # ##############debug
#     # combined = draw_mask(input_image, canvas)
#     # cv2.imshow('input_rgb',combined)
#     # cv2.waitKey(0)
#     return which_point, which_end_point


def group_skel_by_offsets_and_tags(endpoints, keypoints, next_refine, prev_refine, short_offset, input_image):

    skeletons = []
    end_points_map = np.zeros(next_refine.shape[1:])
    location_map = np.zeros(next_refine.shape[1:])
    endpoints_connection_map = np.ones(len(endpoints)) * -1
    # get location map
    for i in range(len(keypoints)):
        x = keypoints[i]['xy'][0]
        y = keypoints[i]['xy'][1]
        location_map[int(y), int(x)] = keypoints[i]['ind'] + 1

    # get end_points_map
    for i in range(len(endpoints)):
        x = endpoints[i]['xy'][0]
        y = endpoints[i]['xy'][1]
        # print(endpoints[i]['ind'])
        end_points_map[int(y), int(x)] = endpoints[i]['ind'] + 1


    for i in range(len(endpoints)):
        this_skel_cp = [] # cp for control points
        this_skel_indexes = []
        this_skel = {'start_point': endpoints[i], 'start_point_ind': i, 'one_point': True}

        this_point = endpoints[i]
        prev_point_corr = this_point['xy']

        # get start point
        # start_point_ind, _ = get_next_point(this_point, short_offset, location_map, end_points_map, prev_point_corr, keypoints)
        # this_skel_cp.append(keypoints[start_point_ind])

        # use next refine
        next_point_ind, which_end_point = get_next_point(this_point, next_refine, location_map, end_points_map, prev_point_corr, keypoints, input_image)

        dists_L2 = np.sqrt((keypoints[next_point_ind]['xy'][0] - this_point['xy'][0]) ** 2 \
                    + (keypoints[next_point_ind]['xy'][1] - this_point['xy'][1]) ** 2)
        #first checkt next point, if next point not is it self, check prev point.

        if dists_L2 < 10:

            prev_point_ind, which_end_point = get_next_point(this_point, prev_refine, location_map, end_points_map, prev_point_corr, keypoints, input_image)

            # dists_L2_prev = np.sqrt((keypoints[prev_point_ind]['xy'][0] - this_point['xy'][0]) ** 2 \
            #         + (keypoints[prev_point_ind]['xy'][1] - this_point['xy'][1]) ** 2)

            if which_end_point == i:
                # if previous point is still it self.
                # append the keypoint.
                # this is one point skel.
                this_skel_cp.append(keypoints[prev_point_ind])
                this_skel_indexes.append(prev_point_ind)
                keypoints[prev_point_ind]['flag'] = True


            else:
                this_point_ind = 99999
                prev_point_corr = this_point['xy']

                while(this_point_ind != prev_point_ind and prev_point_ind != 0):

                    this_point_ind = prev_point_ind

                    this_skel_cp.append(keypoints[this_point_ind])
                    this_skel_indexes.append(this_point_ind)
                    keypoints[this_point_ind]['flag'] = True

                    prev_point_ind, which_end_point = get_next_point(keypoints[this_point_ind], prev_refine, location_map, end_points_map, prev_point_corr, keypoints, input_image)
                    prev_point_corr = keypoints[this_point_ind]['xy']
                    if which_end_point > -1:
                        this_skel_cp.append(keypoints[prev_point_ind])
                        keypoints[prev_point_ind]['flag'] = True
                        this_skel_indexes.append(prev_point_ind)
                        break
        else:
            this_point_ind = 9999
            prev_point_corr = this_point['xy']

            while(this_point_ind != next_point_ind and next_point_ind != 0 ):

                this_point_ind = next_point_ind

                this_skel_cp.append(keypoints[this_point_ind])
                keypoints[this_point_ind]['flag'] = True
                this_skel_indexes.append(this_point_ind)
                next_point_ind, which_end_point = get_next_point(keypoints[this_point_ind], next_refine, location_map, end_points_map, prev_point_corr, keypoints, input_image)
                prev_point_corr = keypoints[this_point_ind]['xy']
                if which_end_point > -1:
                    this_skel_cp.append(keypoints[next_point_ind])
                    keypoints[next_point_ind]['flag'] = True
                    this_skel_indexes.append(next_point_ind)
                    break

        if which_end_point > -1:
            endpoints_connection_map[i] = which_end_point
            this_skel['end_point_ind'] = which_end_point
            this_skel['end_point'] = endpoints[which_end_point]
        else:
            this_skel['end_point_ind'] = -1
            this_skel['end_point'] = None

        this_skel['one_point'] = False
        this_skel['this_skel_cp'] = this_skel_cp
        this_skel['this_skel_indexes'] = this_skel_indexes
        this_skel['activate'] = True
        skeletons.append(this_skel)


    visited = []

    for i in range(len(endpoints)):
        curr_end = i
        # import pdb;pdb.set_trace()
        if not (curr_end in visited):
            # import pdb;pdb.set_trace()
            connected_end = int(endpoints_connection_map[i])

            if connected_end != -1:
                end_from_end_travel_back = int(endpoints_connection_map[int(connected_end)])
                if (end_from_end_travel_back == curr_end):
                    skeletons[connected_end]['activate'] = False
                    visited.append(connected_end)

                    one_skel = skeletons[curr_end]
                    the_other_skel = skeletons[connected_end]

                    one_skel['this_skel_cp'].extend(the_other_skel['this_skel_cp'])
                    one_skel['this_skel_indexes'].extend(the_other_skel['this_skel_indexes'])
                    skeletons[curr_end] = one_skel

    # import pdb;pdb.set_trace()
    # visited = []
    # for i in range(len(connected_ind[0])):

    #     if not (connected_ind[0][i] in visited):
    #         visited.append(connected_ind[0][i])
    #         if not (connected_ind[1][i] in visited):

    #             visited.append(connected_ind[1][i])
    #             one_skel = skeletons[connected_ind[0][i]]
    #             the_other_skel = skeletons[connected_ind[1][i]]
    #             print('---')
    #             print(len(one_skel['this_skel_cp']))
    #             one_skel['this_skel_cp'].extend(the_other_skel['this_skel_cp'])
    #             one_skel['this_skel_indexes'].extend(the_other_skel['this_skel_indexes'])
    #             the_other_skel ['activate'] = False
    #             skeletons[connected_ind[0][i]] = one_skel
    #             skeletons[connected_ind[1][i]] = the_other_skel
    #             print('--------------------------------------------------------------------------------------')
    #             print(len(the_other_skel['this_skel_cp']))
    #             print(len(one_skel['this_skel_cp']))

            # mylist = list(dict.fromkeys(one_skel['this_skel_cp']))
            # print(len(mylist))

    # print(endpoints_connection_map)
    # connected_ind = np.asarray(connected_ind)
    # connected_ind_copy = np.copy(connected_ind)
    # print(connected_ind_copy)
    # connected_ind[[0,1],:] = connected_ind[[1,0],:]
    # print(connected_ind)
    # print('-------------------------')
    # print(connected_ind_copy)
    # share_same_end_points = connected_ind_copy[np.where(connected_ind_copy == connected_ind)]
    # print (share_same_end_points)

    return skeletons, keypoints


def compute_key_points_belongs(all_keypoints, end_points, seg_mask, offsets_map_short):

    map_shape = seg_mask.shape[:2]
    idx = np.rollaxis(np.indices(map_shape[::-1]), 0, 3).transpose((1,0,2)) #get indexes

    offsets_map_short = idx + offsets_map_short.transpose((1,2,0))

    #add one last layer to deal with area with no mask
    direct_dists = np.zeros(map_shape + (len(all_keypoints)+ len(end_points) + 1,))

    offsets = np.zeros(map_shape+(2,))
    offsets_tmp = np.ones(map_shape+(2,))
    offsets_h = np.zeros(map_shape + (len(all_keypoints)+ len(end_points),))
    offsets_v = np.zeros(map_shape + (len(all_keypoints)+ len(end_points),))

    seg_mask_index = seg_mask > 0
    canvas = np.zeros_like(offsets[:,:,0])

    for k in range (len(all_keypoints)):
        # if k == 15:
        #     # import pdb;pdb.set_trace()
        if (all_keypoints[k]['flag'] == False):
            seg_mask_index_reverse = seg_mask == 0
            direct_dists[:,:,k] = 999999

            # direct_dists[seg_mask_index_reverse, k] = 999999
            # direct_dists[seg_mask_index, k] = 999999
        else:

    # for k, center in enumerate(all_keypoints):
            center = (all_keypoints[k]['xy'][1], all_keypoints[k]['xy'][0])
            curr = (int(center[1]), int(center[0]))
            dists = curr - offsets_map_short
            offsets_tmp[seg_mask_index,0] = dists[seg_mask_index,0]
            offsets_tmp[seg_mask_index,1] = dists[seg_mask_index,1]

            # offsets_h[seg_mask_index,k] = dists[seg_mask_index,0]
            # offsets_v[seg_mask_index,k] = dists[seg_mask_index,1]
            # canvas = visualize_offset(canvas, offsets_h[:,:,k], offsets_v[:,:,k])
            # cv2.imshow('t', canvas)
            # cv2.waitKey(0)
            # obtain the shortest dist to the control point
            # direct_dists[:,:,k] = np.sqrt(np.sum(np.square(offsets_tmp), axis = 2))

            #--------------------------------------------------------------

            map_shape = seg_mask.shape
            mask = ~(seg_mask>0)


            phi = 999999.0 * np.ones(map_shape)
            phi  = np.ma.MaskedArray(phi, mask)

            corr_xy = all_keypoints[k]['xy']

            if (corr_xy[1] == 0):
                corr_xy[1] = corr_xy[1] + 1
            elif(corr_xy[1] == (seg_mask.shape[0] - 1)):
                corr_xy[1] = corr_xy[1] - 1


            if (corr_xy[0] == 0):
                corr_xy[0] = corr_xy[0] + 1
            elif(corr_xy[0] == (seg_mask.shape[1] - 1)):
                corr_xy[0] = corr_xy[0] - 1


            if phi.mask[corr_xy[1], corr_xy[0]] == True:
                direct_dists[:,:,k] = 999999
                all_keypoints[k]['flag'] == False
                continue
            phi[corr_xy[1], corr_xy[0]] = -1

    #############debug ###################################
            # print ('k:')

            # print (k)
            # normalized_distance_map = normalize_include_neg_val(skfmm.distance(phi, dx=1).data) # skfmm distances are negative value

            # import matplotlib.pyplot as plt
            # cmap = plt.get_cmap('jet')
            # rgba_img = cmap(normalized_distance_map)
            # print(np.max(normalized_distance_map))
            # print(np.min(normalized_distance_map))
            # cv2.imshow('rgba_img',rgba_img)
            # cv2.imshow('t', normalized_distance_map)
            # cv2.waitKey(0)
    ##############debug end ####################
            normalized_distance_map = (skfmm.distance(phi, dx=1).data)

            # direct_dists[:,:,k] = (skfmm.distance(phi, dx=1).data)

            # import pdb;pdb.set_trace()
            normalized_distance_map[np.where(normalized_distance_map==0)] = 999999
            # print(np.max(normalized_distance_map))
            # print(np.min(normalized_distance_map))

            direct_dists[:,:,k] = normalized_distance_map

    for k_end in range(len(end_points)):
        center = (end_points[k_end]['xy'][1], end_points[k_end]['xy'][0])
        curr = (int(center[1]), int(center[0]))
        # dists = curr - offsets_map_short
        # offsets_tmp[seg_mask_index,0] = dists[seg_mask_index,0]
        # offsets_tmp[seg_mask_index,1] = dists[seg_mask_index,1]
        # direct_dists[:,:, k_end + len(all_keypoints)] = np.sqrt(np.sum(np.square(offsets_tmp), axis = 2)) # obtain the shortest dist to the control point

            #--------------------------------------------------------------

        map_shape = seg_mask.shape
        mask = ~(seg_mask>0)


        phi = 999999.0 * np.ones(map_shape)
        phi  = np.ma.MaskedArray(phi, mask)

        corr_xy = end_points[k_end]['xy']

        if (corr_xy[1] == 0):
            corr_xy[1] = corr_xy[1] + 1
        elif(corr_xy[1] == (seg_mask.shape[0] - 1)):
            corr_xy[1] = corr_xy[1] - 1


        if (corr_xy[0] == 0):
            corr_xy[0] = corr_xy[0] + 1
        elif(corr_xy[0] == (seg_mask.shape[1] - 1)):
            corr_xy[0] = corr_xy[0] - 1

        if phi.mask[corr_xy[1], corr_xy[0]] == True:
            direct_dists[:,:,k_end + len(all_keypoints)] = 999999
            continue

        phi[corr_xy[1], corr_xy[0]] = -1

    #############debug ###################################

        # normalized_distance_map = normalize_include_neg_val(skfmm.distance(phi, dx=1).data) # skfmm distances are negative value
        # import matplotlib.pyplot as plt
        # cmap = plt.get_cmap('jet')
        # rgba_img = cmap(normalized_distance_map)
        # cv2.imshow('rgba_img',rgba_img)
        # cv2.imshow('t', normalized_distance_map)
        # cv2.waitKey(0)
    ##############debug end ####################

        normalized_distance_map = (skfmm.distance(phi, dx=1).data)

        # normalized_distance_map[np.where(normalized_distance_map==0)] = 999999
        normalized_distance_map[:,:] = 999999
        direct_dists[:,:, k_end + len(all_keypoints)] = normalized_distance_map
    try:

        #last layer is -1, when there is no mask (to aovid 0 when there is no mask.)
        seg_mask_index_reverse = seg_mask == 0
        direct_dists[seg_mask_index_reverse, len(all_keypoints) + len(end_points)] = -2
        direct_dists[seg_mask_index, len(all_keypoints) + len(end_points)] = 9999999

        closest_keypoints = np.argmin(direct_dists, axis=2)
        for k in range(len(all_keypoints)):
            all_keypoints[k]['area'] = np.where(closest_keypoints == k)

        direct_dists[:,:, len(all_keypoints):] = direct_dists[:,:, len(all_keypoints):] - 0.1
        closest_keypoints = np.argmin(direct_dists, axis=2)
        for k_end in range(len(end_points)):
            end_points[k_end]['area'] = np.where(closest_keypoints == (len(all_keypoints) + k_end))

    except:
        import pdb;pdb.set_trace()

    # debug_canvas =  np.zeros(map_shape)
    # for k in range(len(all_keypoints)):
    #     debug_canvas[all_keypoints[k]['area']] = 1
    #     cv2.circle(debug_canvas, (all_keypoints[k]['xy'][0],all_keypoints[k]['xy'][1]), 5, 5, 1)
    #     cv2.imshow('t', debug_canvas)
    #     cv2.waitKey(0)

    return all_keypoints, end_points


def get_end_point(keypoints, offset_map_next, offset_map_pre):

    for ind in range(len(keypoints)):
        end_points_candidate = []
        x = keypoints[ind]['xy'][0]
        y = keypoints[ind]['xy'][1]
        temp_heatmap = np.zeros(offset_map_next.shape[1:])

        temp_heatmap[int(y),int(x)] = 1
        kernel = np.ones((10,10), np.float32)
        img_dilation = cv2.dilate(temp_heatmap, kernel, iterations=1)

        # print (np.max(heatmap_pre))
        # h_show = heatmap_pre
        # imgplot = plt.imshow(h_show)
        # plt.show()

        # import pdb;pdb.set_trace()
        heatmap_next = compute_heatmaps(img_dilation, offset_map_next)
        heatmap_next = gaussian_filter(heatmap_next, sigma=5)
        peaks_next = maximum_filter(heatmap_next, footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmap_next

        peaks_next = zip(*np.nonzero(peaks_next))
        end_points_candidate.extend([{'xy': np.array(peak[::-1]), 'conf': heatmap_next[peak[0], peak[1]]} for peak in peaks_next])

        heatmap_pre = compute_heatmaps(img_dilation, offset_map_pre)

        peaks_pre = maximum_filter(heatmap_pre, footprint=[[0,1,0],[1,1,1],[0,1,0]]) == heatmap_pre
        peaks_pre = zip(*np.nonzero(peaks_pre))

        end_points_candidate.extend([{'xy': np.array(peak[::-1]), 'conf': heatmap_pre[peak[0], peak[1]]} for peak in peaks_pre])

        end_points_candidate = [kp for kp in end_points_candidate if kp['conf'] > 0.001]

        is_end_point = False
        for candidate in end_points_candidate:
            if np.sqrt((candidate['xy'][0] - x) ** 2 + (candidate['xy'][1] - y) ** 2 ) < 11 :
                is_end_point = True

        keypoints[ind]['end_point'] = is_end_point
    return keypoints

def resize_back_output_shape(input_map, output_shape):

    input_map_shape = input_map.shape
    output_map = cv2.resize(input_map, (int(output_shape[1]), int(output_shape[0])),interpolation = cv2.INTER_NEAREST )
    return output_map

def refine_next(keypoints, short_offsets, mid_offsets, num_steps):

    x = keypoints[0]
    y = keypoints[1]

    y_v = short_offsets.shape[1] # height
    x_h = short_offsets.shape[2] # width

    mid_offsets_h = mid_offsets[0]
    mid_offsets_v = mid_offsets[1]

    short_offsets_h = short_offsets[0]
    short_offsets_v = short_offsets[1]

    y = min(y_v - 1, int(y))
    if int(y) < 0:
        y = 0
    x = min(x_h - 1, int(x))

    if int(x) < 0:
        x = 0
    for i in range(num_steps):
        curr = (x, y)



        offset_x = mid_offsets_h[y, x]
        offset_y = mid_offsets_v[y, x]

        tmp_y = min(y_v - 1, int(y + offset_y))
        if int(y + offset_y) < 0:

            tmp_y = 0
        tmp_x = min(x_h - 1, int(x + offset_x))
        if int(x + offset_x) < 0:
            tmp_x = 0

        offset_x_n = offset_x + short_offsets_h[tmp_y, tmp_x]
        offset_y_n = offset_y + short_offsets_v[tmp_y, tmp_x]

        new_x = int(x + offset_x_n)
        new_y = int(y + offset_y_n)

        x = new_x
        y = new_y


        y = min(y_v - 1, int(y))
        if int(y) < 0:
            y = 0
        x = min(x_h - 1, int(x))
        if int(x) < 0:
            x = 0

    return (x, y)

def group_skels_by_tag(keypoints, ae_threshold = 1.0):

    tag_values = np.zeros(len(keypoints))
    for i in range(len(keypoints)):
        tag_values[i] = keypoints[i]['tag_val']

    tag_values_a = np.expand_dims(tag_values, axis = 0)
    tag_values_b = np.expand_dims(tag_values, axis = 1)
    # import pdb;pdb.set_trace()
    dists = np.absolute(tag_values_a - tag_values_b)
    dists = dists < ae_threshold
    skels = []
    for i in range(len(keypoints)):
        # import pdb;pdb.set_trace()
        indexes = np.where(dists[i,:] == True)
        new_skel = []
        for indx in indexes[0]:
            new_skel.append(keypoints[indx])
            dists[indx,:] = False
            dists[:, indx] = False
        if len(new_skel) != 0:
            skels.append(new_skel)

    return skels

def group_one_skel_by_angle(st, endpoints, keypoints, short_offsets, mid_offsets_next):

    x = st[0]
    y = st[1]
    skel = []
    curr = (x,y)
    canvas = np.zeros(short_offsets[0].shape)
    cv2.circle(canvas, curr, 5, 1, 3)

    skel.append(curr)
    continue_flag = True
    count = 1
    count_2 = 1
    double_mid_offsets_next = mid_offsets_next * 2
    mid_offsets_next_to_use = mid_offsets_next
    length = 0
    # import pdb; pdb.set_trace()
    while (continue_flag):


        proposal_next = refine_next(curr, short_offsets, mid_offsets_next_to_use, 1)
        proposal_next_next = refine_next(proposal_next, short_offsets, mid_offsets_next, 1) # always mid offsets.
        curr_dir = [proposal_next[1] - curr[1], proposal_next[0] - curr[0]]
        next_dir = [proposal_next_next[1] - proposal_next[1], proposal_next_next[0] -  proposal_next[0]]
        curr_dir = np.asarray(curr_dir)
        next_dir = np.asarray(next_dir)
        angle = get_angle(curr_dir, next_dir)
        print (np.linalg.norm(curr_dir))
        print (np.linalg.norm(next_dir))
        print (angle)

        #   1 check if it is end point
        for i in range(len(endpoints)):
            if np.linalg.norm(np.asarray(proposal_next)-np.asarray(endpoints[i]['xy'])) <= 12:
                print ('endpoints_added')
                skel.append(proposal_next)
                cv2.circle(canvas, proposal_next, 5, 1, 3)
                return skel, canvas

        if np.linalg.norm(np.asarray(proposal_next)-np.asarray(proposal_next_next)) <= 10:
            print (np.linalg.norm(np.asarray(proposal_next)-np.asarray(proposal_next_next)))
            if count_2 == 0:
                skel.append(proposal_next)
                return skel, canvas
            else:
                skel.append(proposal_next)
                curr = proposal_next
                count_2 = 0

        count_2 = 1
        if angle < 45:
            tmp_ind = 0
            dist_min = np.inf
            for i in range(len(keypoints)):
                dist = np.linalg.norm(np.asarray(proposal_next)-np.asarray(keypoints[i]['xy']))
                if dist <= dist_min:
                    dist_min = dist
                    tmp_ind = i
            if dist_min < 12:
                proposal_next = (keypoints[tmp_ind]['xy'][0],keypoints[tmp_ind]['xy'][1])


            cv2.circle(canvas, proposal_next, 5, 1, 1)
            skel.append(proposal_next)
            curr = proposal_next

            mid_offsets_next_to_use = mid_offsets_next
            count = 1
            length += 1
            print(length)
            if length > 50:
                return skel, canvas
        else:
            print ('bigger than 45')
            if count == 1:
                mid_offsets_next_to_use = double_mid_offsets_next
                count = 0
            else:
                return skel, canvas



def convert_to_coco(input_image, skels, keypoints, endpoints, meta):
    index = meta['index']
    imgID = meta['imgID']
    img_path = meta['img_path']
    det_scores = meta['det_scores']
    input_shape = meta['input_shape']
    out_shape = meta['out_shape']

    canvas_skel = input_image
    canvas_instance = input_image
    num = 0
    final_result = []
    for i in range(len(skels)):
        single_result_dict = {}
        canvas_instance = input_image
        # print(num)
        if skels[i]['activate'] == True:
            one_canvas = np.zeros(input_image.shape[:2])

            for ii in range(len(skels[i]['this_skel_indexes'])):
                ind = skels[i]['this_skel_indexes'][ii]
                one_canvas[keypoints[ind]['area']] = 1

            ind_start = skels[i]['start_point_ind']
            one_canvas[endpoints[ind_start]['area']] = 1
            ind_end = skels[i]['end_point_ind']
            one_canvas[endpoints[ind_end]['area']] = 1

            # import pdb;pdb.set_trace()
            resize_back_output_shape = cv2.resize(one_canvas, (input_shape[1], input_shape[0]), interpolation = cv2.INTER_NEAREST)
            mask_pologons = mask_to_pologons(resize_back_output_shape>0)
            bbox = extract_bboxes(resize_back_output_shape)
            # import pdb;pdb.set_trace()

            def draw_mask(im, mask):
            # import pdb; pdb.set_trace()
                mask = mask > 0.02

                r = im[:, :, 0].astype(np.float32)
                g = im[:, :, 1].astype(np.float32)
                b = im[:, :, 2].astype(np.float32)

                r[mask] = 1.0
                g[mask] = 1.0
                b[mask] = 0.0

                combined = cv2.merge([r, g, b]) * 0.5 + im.astype(np.float32) * 0.5
                return combined.astype(np.float32)
####################debug
            input_image_debug = cv2.resize(input_image, (input_shape[1],input_shape[0]), interpolation = cv2.INTER_NEAREST)
            test = draw_mask(input_image_debug, resize_back_output_shape)
            cv2.imshow('tt', test)
            cv2.waitKey(0)
            test = cv2.rectangle(test, (bbox[0],bbox[1]), (bbox[0] + bbox[2],bbox[1] + bbox[3]), (255, 0, 0), 1)

            cv2.imshow('t', test)
            cv2.waitKey(0)
####################

            # import pdb;pdb.set_trace()
            single_result_dict['image_id'] = int(imgID.cpu().numpy())
            single_result_dict['img_path'] = img_path[0]
            single_result_dict['id'] = int(index.cpu().numpy())
            single_result_dict['category_id'] = 1
            single_result_dict['iscrowd'] = 0
            single_result_dict['segmentation'] = list(mask_pologons)
            single_result_dict['bbox'] = bbox
            single_result_dict['annIds'] = num
            single_result_dict['score'] = float(99.99)
            num = num + 1

            final_result.append(single_result_dict)



    return final_result, int(imgID.cpu().numpy())

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([4], dtype=np.int32)

    m = mask[:, :]
    # Bounding box.
    horizontal_indicies = np.where(np.any(m, axis=0))[0]
    # print('horizontal_indicies',horizontal_indicies)
    # print('horizontal_indicies_shape',horizontal_indicies.shape)
    vertical_indicies = np.where(np.any(m, axis=1))[0]
    if horizontal_indicies.shape[0]:
        x1, x2 = horizontal_indicies[[0, -1]]
        y1, y2 = vertical_indicies[[0, -1]]
        # x2 and y2 should not be part of the box. Increment by 1.
        # x2 += 1
        # y2 += 1
        y2 = y2 - y1
        x2 = x2 - x1
    else:
        # No mask for this instance. Might happen due to
        # resizing or cropping. Set bbox to zeros
        x1, x2, y1, y2 = 0, 0, 0, 0
    # boxes = np.array([y1, x1, y2, x2])
    # boxes = (int(y1), int(x1), int(y2), int(x2))
    boxes = (int(x1), int(y1), int(x2), int(y2))
    return boxes
