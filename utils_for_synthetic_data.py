import sys
import numpy as np
sys.path.append('./')
from utility.preprocess import get_keypoint_discs
from utility.preprocess import compute_short_offsets, compute_mid_offsets

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