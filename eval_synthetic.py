import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
# from scipy.misc import imread, imresize
import imageio
from PIL import Image
from fiberDataset_COCO import *
from realFiberDataset import *
from utility.postprocess import split_and_refine_mid_offsets, get_next_point
from utils import LMFPeakFinder, mask_to_pologons
from copy import deepcopy
import cocoapi.PythonAPI.pycocotools.mask as coco_mask_util
from tqdm import tqdm
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from skimage.morphology import skeletonize
from utility.color_map import GenColorMap
from utility.get_intersection_and_endpoint import get_skeleton_intersection, get_skeleton_intersection_and_endpoint
from config_Synthetic import cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from skimage.measure import label  
import time
from utility.color_map import LabelToColor

def getLargestCC(segmentation, points):
    labels = label(segmentation)
    largest_vote = labels[np.asarray(points[1]).astype('uint8'),np.asarray(points[0]).astype('uint8')]
    
    assert( labels.max() != 0 ) # assume at least 1 CC
    # largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    try:
        largestCC = labels == np.argmax(np.bincount(largest_vote)[1:])+1
    except:
        largestCC = labels == 1
    return largestCC

def draw_mask(im, mask, color):

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

def pologons_to_mask(polygons, size):
    height, width = size
    # formatting for COCO PythonAPI

    rles = coco_mask_util.frPyObjects(polygons, height, width)


    rle = coco_mask_util.merge(rles)
    mask = coco_mask_util.decode(rle)
    return mask

if __name__ == '__main__':
    max_len = 50
    visual = cfg.visual

    result_file = cfg.save_weights_name + '.json'
    parser = argparse.ArgumentParser(description='evalsyntehtic')

    parser.add_argument('--model', '-m', help='path to model')

    args = parser.parse_args()
    data_folder_val = cfg.data_folder_test
    image_folder_val =cfg.image_folder_test
    offset_folder_val = cfg.offset_folder_test
    jason_file_val = cfg.jason_file_test
    gt_file = data_folder_val + '/' + jason_file_val

    patch_size = cfg.patch_size
    batch_size = 1
    workers = 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_folder = None

    val_loader_p = torch.utils.data.DataLoader(
            fiberDataset_COCO(data_folder_val, jason_file_val, image_folder_val, offset_folder_val, transforms.Compose([transforms.ToTensor(), normalize]), False),
            batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Load model

    checkpoint = torch.load(args.model, map_location=str(device))
    multiGpu = False
    if multiGpu:
        decoder = checkpoint['decoder']
        decoder = (decoder.module).to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder =  (encoder.module).to(device)
        encoder.eval()
        decoderMulti = checkpoint['decoderMulti']
        decoderMulti = (decoderMulti.module).to(device)
        decoderMulti.eval()    
    else:
        # import pdb; pdb.set_trace()
        decoder = checkpoint['decoder']
        decoder = decoder.to(device)
        decoder.eval()
        encoder = checkpoint['encoder']
        encoder = encoder.to(device)
        encoder.eval()
        decoderMulti = checkpoint['decoderMulti']
        decoderMulti = decoderMulti.to(device)
        decoderMulti.eval()

    results = []

    for i,  (imgs, meta) in enumerate(val_loader_p):
        # if i >0:
        #     break
        start = time.time()
        print(meta['image_id'])

        img_path = meta['img_path']
        org_image = imageio.imread(img_path[0])

  
        org_image = np.asarray(org_image).astype('uint8')
        
        ###############################

        #########################3

        if len(org_image.shape) > 2:
            if org_image.shape[2] > 3:
                org_image = org_image[:,:,:3]
        # org_image = cv2.resize(org_image, (256,256))

        real_org_image = deepcopy(org_image)
 
        imgs = imgs.to(device, dtype=torch.float)[0]



        imgs = imgs.unsqueeze(0)

        [encoder_out,x4,x3,x2,x1] = encoder(imgs)


        control_area_pred, end_area_pred, _ = decoderMulti([x4,x3,x2,x1])
        #######################################
        encoder_dim = encoder_out.size(-1)
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        #######################################
        
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(control_area_pred.squeeze().detach().cpu().numpy() + org_image*0.0)
        rgba_img2 = cmap(end_area_pred.squeeze().detach().cpu().numpy()+ org_image*0.0)
    
        peak_finder = LMFPeakFinder(min_th=0.8)
        coord, score = peak_finder.detect(control_area_pred.squeeze().detach().cpu().numpy())
        coord_end, score = peak_finder.detect(end_area_pred.squeeze().detach().cpu().numpy())

         # ############
        # ############
        # ############

        if visual:
            from copy import deepcopy
            draw_circle = deepcopy(org_image)
            
            draw_circle_end = np.zeros(org_image.shape)
            for (x,y) in coord:
                cv2.circle(draw_circle, (int(y), int(x)), 8, (200,0,255), 1)
            cv2.imshow('draw_circle',draw_circle)
            
            for (x,y) in coord_end:
                if y < 56 and y > 40 and x < 135 and x > 115:
                    continue
                if y < 110 and y > 90 and x < 110 and x > 90:
                    continue
                cv2.circle(draw_circle_end, (int(y), int(x)), 8, (255,0,255), 2)

            draw_circle_end = draw_mask(np.tile(org_image,(3,1,1)).transpose(1,2,0), draw_circle_end, (0,0,255))

            cv2.imwrite('demo_output/'+'end_point.png', draw_circle_end)
            cv2.imwrite('demo_output/'+'org.png', org_image)
            cv2.imshow('draw_circle_end',draw_circle_end)
            cv2.waitKey(0)
            # import pdb; pdb.set_trace()

            cv2.imshow('org_image',org_image)
            cv2.imshow('rgba_img_end',rgba_img2)
            cv2.imshow('rgba_img_control',rgba_img)
            cv2.waitKey(0)

        curr_image_NMS = []

        map_NMS = np.zeros(org_image.shape[:2]) 
        max_label = 1
        # import pdb; pdb.set_trace()
        
        for ed_id in tqdm(range(len(coord_end))): 
            if visual:
                CLASS_2_COLOR, COLOR_2_CLASS = GenColorMap(200)
                color = CLASS_2_COLOR[int(ed_id) + 1]

            (sp_row_y, sp_col_x) = coord_end[ed_id]
            start_point = torch.FloatTensor([[sp_col_x,sp_row_y]])
            start_point = start_point.unsqueeze(0).to(device)
            embeddings = decoder.patchEmbedding(imgs, start_point)
            embeddings = embeddings.squeeze(2)
            

            h, c = decoder.init_hidden_state(embeddings[:batch_size,0,:])

            predictions = torch.zeros(batch_size, max_len, 2).to(device)
            predictionsStop = torch.zeros(batch_size, max_len, 1).to(device)
            alphas = torch.zeros(batch_size,max_len, num_pixels).to(device)

        
            offset_prediction = torch.zeros(batch_size, max_len, 2, patch_size, patch_size).to(device)    
        
            padding_size = patch_size // 2 + patch_size // 4
            
            
            image_padded = np.zeros_like(org_image)
            real_org_image_for_instance = deepcopy(real_org_image)
            org_image =  deepcopy(real_org_image)

            one_instance = {}

            predicted_sequence_instance = []
            end_point_instance = []
            segmentation_instance = []
            contro_point_instance = [[],[]]
            non_visited = coord[:, [1,0]]
            non_visited_end_points = coord_end[:, [1,0]]
    

            for t in range(max_len):
                
                # batch_size_t = sum([l > t for l in decode_lengths])
                batch_size_t = 1
                # if t <= 0:
                #     preds = sequence[:batch_size_t, t, :]
                
                attention_weighted_encoding, alpha = decoder.attention(encoder_out[:batch_size_t],
                                                                    h[:batch_size_t])
                gate = decoder.sigmoid(decoder.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

                if t <= 0:
                    h, c = decoder.decode_step(
                        embeddings[:batch_size_t,t,:],
                        # torch.cat([torch.floor(preds.data)], dim=1),
                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
                    pre = start_point[:batch_size_t, t, :]

                    end_point_instance.append(pre.detach().cpu().numpy())
                    distance_to_end = np.sqrt(np.sum(np.square(non_visited_end_points - pre.detach().cpu().numpy()), axis = 1))
                    indx_closest = np.argmin(distance_to_end)
                    non_visited_end_points = np.delete(non_visited_end_points, indx_closest, 0)      

                    # distance_to_control = np.sqrt(np.sum(np.square(non_visited - pre.detach().cpu().numpy()), axis = 1))
                    # indx_closest = np.argmin(distance_to_control)
                    # non_visited = np.delete(non_visited, indx_closest, 0)
                    
                else:
                    contro_point_instance[0].append(pre.detach().cpu().numpy()[0][0])
                    contro_point_instance[1].append(pre.detach().cpu().numpy()[0][1])
                    offset_dist = (offset_next[0] ** 2 + offset_next[1] ** 2) ** 0.5

                    # distance_to_end = np.sqrt(np.sum(np.square(non_visited_end_points - pre.detach().cpu().numpy()), axis = 1))
                    # indx_closest = np.argmin(distance_to_end)
                    # if distance_to_end[indx_closest] < 5:
                    #     print('reached')
                    #     break
                        
                    if offset_dist < 6:
                        distance_to_end = np.sqrt(np.sum(np.square(non_visited_end_points - pre.detach().cpu().numpy()), axis = 1))
                        indx_closest = np.argmin(distance_to_end)  

                        turn_distance_to_end_off = True
                        if distance_to_end[indx_closest] < 10 or turn_distance_to_end_off:
                            end_point_instance.append(pre.detach().cpu().numpy())
                            image_padded = cv2.dilate(image_padded, np.ones((3,3)))
                            image_padded = cv2.erode(image_padded, np.ones((3,3)))
                            # image_padded = skeletonize(1.0 * (image_padded>0))
                            # image_padded = cv2.dilate(image_padded.astype('uint8'), np.ones((3,3)))
                            image_padded = image_padded * (1.0 * (real_org_image_for_instance>0))
                            
                            if np.sum(image_padded) < 5:
                                break      
                            image_padded = getLargestCC(image_padded, contro_point_instance)        
                            image_padded = ((image_padded > 0) * 1.0).astype('uint8')
                            # image_padded_ = draw_mask_color(org_image, image_padded, [0,0,255])

                            if visual:
                                cv2.imshow('instance', image_padded)
                                cv2.waitKey(0)
                                print('reached_confirm')                           
                            
                            repeat_flag = False
                            
                            #######################################
                            # for i in range(len(curr_image_NMS)):
                            #     intersection = np.sum((curr_image_NMS[i]>0) * (image_padded > 0))
                            #     union = np.sum(curr_image_NMS[i] > 0) + np.sum(image_padded> 0)
                            #     iou = 2 * intersection/union
                            #     if iou > 0.8:
                            #         print('repeated')
                            #         print(iou)
                            #         repeat_flag = True
                            #         break
                            # if repeat_flag:
                            #     break
                            ###########################################

                            intersection = np.sum((map_NMS>0) * (image_padded > 0))
                            
                            curr_pixel = np.sum((image_padded>0))
                            if curr_pixel < 10:
                                print('break current:',curr_pixel)
                                break
                            overlap_pixels_labels = np.unique(map_NMS[image_padded>0])
                            curr_pixels = np.where(image_padded>0)
                            overlap_pixels = map_NMS[image_padded>0]
                            myself_pixels = np.sum(image_padded)

                            overlap_flag = False
                            asigned_label = max_label

                            for label_id in overlap_pixels_labels:
                                if label_id == 0:
                                    continue
                                curr_overlapped_pixels = np.sum(map_NMS == label_id)
                                intersection_pixels = np.sum(overlap_pixels == label_id)
                                #1. I over that:
                                if intersection_pixels / myself_pixels > 0.75 or intersection_pixels / curr_overlapped_pixels > 0.75:
                                    overlap_flag = True
                                    print('overlap')
                                    print(intersection_pixels / myself_pixels)
                                    print(intersection_pixels / curr_overlapped_pixels)
                                    asigned_label = label_id
                                    break

                            if not overlap_flag:
                                max_label += 1
                                
                            map_NMS[curr_pixels]=asigned_label

                            # curr_image_NMS.append(image_padded)

                            # seg_poly = coco_mask_util.encode(np.asfortranarray(image_padded))
                            # seg_poly = mask_to_pologons(image_padded>0)
                            
                            # mask_ind = np.where(image_padded>0)
                            # bbox = [int(min(end_point_instance[0][0][0],end_point_instance[1][0][0])),\
                            #     int(min(end_point_instance[0][0][1],end_point_instance[1][0][1])),\
                            #         int(abs(end_point_instance[0][0][0]- end_point_instance[1][0][0])),\
                            #             int(abs(end_point_instance[0][0][1] - end_point_instance[1][0][1]))]


                            # bbox = [int(min(mask_ind[1])),\
                            #     int(min(mask_ind[0])),\
                            #         int(abs(max(mask_ind[1])- min(mask_ind[1]))),\
                            #             int(abs(max(mask_ind[0])- min(mask_ind[0])))]

                            # one_instance['image_id'] = int(meta['image_id'].detach().cpu()) 
                            # one_instance['bbox'] = bbox
                            # one_instance['category_id'] = 1        
                            # one_instance['segmentation'] = seg_poly       
                            # one_instance['score'] = 99 
                            # results.append(one_instance)


                        # print('end_point')
                        break
                    # import pdb; pdb.set_trace()
                    curr = pre + torch.Tensor(offset_next).unsqueeze(0).to(device)
                

                    curr_coor_y = curr[0][1].detach().cpu().numpy()
                    coor_coor_x = curr[0][0].detach().cpu().numpy()

                    validate_image = np.zeros(org_image.shape[:2])

                    validate_image = cv2.circle(validate_image, (int(max(0, min(coor_coor_x, org_image.shape[1]))), int(max(0, min(curr_coor_y, org_image.shape[0])))), 5, (200,0,0), 1)
                    validate_image = (validate_image * (1.0 * (real_org_image_for_instance>0)) > 0)

                    
                    if  curr[0][0] < 0 or curr[0][1] < 0 or curr[0][0] >= imgs.shape[-1] or curr[0][1] >= imgs.shape[-2]:
                        distance_to_end = np.sqrt(np.sum(np.square(non_visited_end_points - pre.detach().cpu().numpy()), axis = 1))
                        # distance_to_end_curr = np.sqrt(np.sum(np.square(non_visited_end_points - curr.detach().cpu().numpy()), axis = 1))
                        indx_closest = np.argmin(distance_to_end)   
                        # print('break_over')
                        # print(distance_to_end[indx_closest]) 
                        # if distance_to_end[indx_closest] < 25:
                        if True:
                            end_point_instance.append(pre.detach().cpu().numpy())
                            image_padded = cv2.dilate(image_padded, np.ones((3,3)))
                            image_padded = cv2.erode(image_padded, np.ones((3,3)))
                            # image_padded = skeletonize(1.0 * (image_padded>0))
                            # image_padded = cv2.dilate(image_padded.astype('uint8'), np.ones((3,3)))

                            image_padded = image_padded * (1.0 * (real_org_image_for_instance>0))
                            
                            if np.sum(image_padded) < 5:
                                break    

                            image_padded = getLargestCC(image_padded, contro_point_instance)        
                            image_padded = ((image_padded > 0) * 1.0).astype('uint8')


                            if visual:
                                cv2.imshow('instance', image_padded)
                                cv2.waitKey(0)
                                print('reached_confirm')   

                            repeat_flag = False

                            ######################################33
                            # for i in range(len(curr_image_NMS)):
                            #     intersection = np.sum((curr_image_NMS[i]>0) * (image_padded > 0))
                            #     union = np.sum(curr_image_NMS[i] > 0) + np.sum(image_padded> 0)
                            #     iou = 2 * intersection/union
                            #     if iou > 0.8:
                            #         print('repeated')
                            #         print(iou)
                            #         repeat_flag = True
                            #         break
                                
                            # if repeat_flag:
                            #     break         
                            ###############################                  

                            # curr_image_NMS.append(image_padded)
                            # # seg_poly = coco_mask_util.encode(np.asfortranarray(image_padded))
                            # seg_poly = mask_to_pologons(image_padded>0)
                            
                            # mask_ind = np.where(image_padded>0)
                            # # bbox = [int(min(end_point_instance[0][0][0],end_point_instance[1][0][0])),\
                            # #     int(min(end_point_instance[0][0][1],end_point_instance[1][0][1])),\
                            # #         int(abs(end_point_instance[0][0][0]- end_point_instance[1][0][0])),\
                            # #             int(abs(end_point_instance[0][0][1] - end_point_instance[1][0][1]))]
                            # bbox = [int(min(mask_ind[1])),\
                            #     int(min(mask_ind[0])),\
                            #         int(abs(max(mask_ind[1])- min(mask_ind[1]))),\
                            #             int(abs(max(mask_ind[0])- min(mask_ind[0])))]
                            # one_instance['image_id'] = int(meta['image_id'].detach().cpu()) 
                            # one_instance['category_id'] = 1   
                            # one_instance['bbox'] = bbox     
                            # one_instance['segmentation'] = seg_poly       
                            # one_instance['score'] = 99  
                            # results.append(one_instance)

                            ###################################


                            curr_pixel = np.sum((image_padded>0))
                            if curr_pixel < 50:
                                print('break current:',curr_pixel)
                                break

                            overlap_pixels_labels = np.unique(map_NMS[image_padded>0])
                            curr_pixels = np.where(image_padded>0)
                            overlap_pixels = map_NMS[image_padded>0]
                            myself_pixels = np.sum(image_padded)

                            overlap_flag = False
                            asigned_label = max_label

                            for label_id in overlap_pixels_labels:
                                if label_id == 0:
                                    continue
                                curr_overlapped_pixels = np.sum(map_NMS == label_id)
                                intersection_pixels = np.sum(overlap_pixels == label_id)
                                #1. I over that:
                                if intersection_pixels / myself_pixels > 0.75 or intersection_pixels / curr_overlapped_pixels > 0.75:
                                    overlap_flag = True
                                    print('overlap')
                                    print(intersection_pixels / myself_pixels)
                                    print(intersection_pixels / curr_overlapped_pixels)
                                    asigned_label = label_id                                    
                                    break
                                
                            if not overlap_flag:
                                max_label += 1
                                
                            map_NMS[curr_pixels]=asigned_label       
                                
                            if visual:
                                cv2.imshow('instance', image_padded)
                                cv2.waitKey(0)
                                print('reached_confirm')

                        # print('end_point')
                        break    


                    # distance_to_control = np.sqrt(np.sum(np.square(non_visited - curr.detach().cpu().numpy()), axis = 1))
                    # indx_closest = np.argmin(distance_to_control)
                    # curr = torch.Tensor(non_visited[indx_closest]).unsqueeze(0).to(device)
                    # non_visited = np.delete(non_visited, indx_closest, 0)
                    ####################################
                    # import pdb; pdb.set_trace()
                    # curr = sequence[:batch_size_t, t-1, :] + torch.Tensor(offset_next).unsqueeze(0).to(device)
                    # curr = pre + sequence_offset[t-1, :]    
                    pre = curr
                    # curr = sequence[:batch_size_t, t, :]
                    # import pdb; pdb.set_trace()
                    # short_offset = ground_truth_multi[:,t-1,2:4,int(32 + preds[0,0]),int(32 + preds[0,1])]
                    # short_offset = test[:,0:2,int(32 + preds[0,0]),int(32 + preds[0,1])]
                    # curr = curr + short_offset
                    # pre = curr
                    curr = curr.unsqueeze(0)
                    
                    # import pdb; pdb.set_trace()
                    #############################


                    embeddings = decoder.patchEmbedding(imgs, curr) #

                    embeddings_2 = embeddings.squeeze(2)
                    embeddings_2 = embeddings_2[:batch_size_t,0,:]
                    # embeddings_2 = embeddings[:batch_size,t,:]
                    ######################################
                    # import pdb; pdb.set_trace()
                    h, c = decoder.decode_step(
                        embeddings_2,
                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
               ####################################################################################
                #######################################################################
                patch_Seg = decoder.sigmoid(decoder.patchSegmentationDecoder(decoder.dropout(h)))        

                if t == 0:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                else:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                center = [coor_x+patch_size, coor_y+patch_size]
                # shift_x = random.randint(0,self.patch_size // 4)
                # shift_y = random.randint(0,self.patch_size // 4)
                shift_x = 0
                shift_y = 0
                center[0] = center[0] + shift_x
                center[1] = center[1] + shift_y

                x_coor_st = int(center[0].item() - patch_size//2 + patch_size//2)
                x_coor_ed = int(center[0].item() + patch_size//2 + patch_size//2)
                y_coor_st = int(center[1].item() - patch_size//2 + patch_size//2)
                y_coor_ed = int(center[1].item() + patch_size//2 + patch_size//2)
                indx = np.where(patch_Seg.detach().cpu().numpy().squeeze() > 0.8)

                
                indx = [(indx[0]+int(coor_y)-patch_size//2),(indx[1]+int(coor_x))-patch_size//2]
                
                # import pdb; pdb.set_trace()
                indx[0][indx[0]>= org_image.shape[0]] = org_image.shape[0]-1
                indx[0][indx[0]<0] = 0
                indx[1][indx[1]>= org_image.shape[1]] = org_image.shape[1]-1
                indx[1][indx[1]<0] = 0
                # org_image[indx] = 255.//2
                # image_padded[x_coor_st :x_coor_ed,  y_coor_st :y_coor_ed] = (patch_Seg.detach().cpu().numpy().squeeze() > 0.5) * 255. // 2
                image_padded[tuple(indx)] = 255
                
                if visual:
                    cv2.imshow('test2', (patch_Seg.detach().cpu().numpy().squeeze() > 0.5) * 255.)
                    show_mask = draw_mask_color(np.tile(real_org_image,(3,1,1)).transpose(1,2,0), image_padded, color)
                    show_rectangle  = cv2.rectangle(show_mask, (int(coor_x)-32,int(coor_y)-32),(int(coor_x)+32,int(coor_y)+32), color, 2) 
                    cv2.imshow('test3', image_padded)
                    cv2.imshow('show_mask', show_mask)
                    cv2.imshow('show_rectangle', show_rectangle)
                    cv2.waitKey()

                test = decoder.patchDecoder(decoder.dropout(h))      
                predsStop = decoder.fcStop(decoder.dropout(h))  # (batch_size_t, vocab_size)
                predsStop = decoder.fcStop_act(predsStop)
                # if predsStop > 0.8:
                #         # import pdb; pdb.set_trace()
                #         print('break_stop')
                #         print(predsStop)
                #         break                
                # predictions[:batch_size_t, t, :] = preds
                predictionsStop[:batch_size_t, t, :] = predsStop
                alphas[:batch_size_t, t, :] = alpha
                # print(torch.round(preds.data))
                # print(predsStop)

                # import pdb; pdb.set_trace()
                # coor_y = int(preds[0][1].detach().cpu().numpy())
                # coor_x = int(preds[0][0].detach().cpu().numpy())
                # import pdb; pdb.set_trace()
                if t == 0:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                else:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                
                # import pdb; pdb.set_trace()

                org_image = cv2.circle(org_image, (int(max(0, min(coor_x, org_image.shape[1]))), int(max(0, min(coor_y, org_image.shape[0])))), 5, (200,0,0), 1)
                # # import pdb; pdb.set_trace()
                if visual:
                    cv2.imshow('test', org_image)
                    cv2.waitKey()
                canvas = np.zeros((patch_size, patch_size))
                center_mask = np.zeros((patch_size,patch_size))
                center_mask = cv2.circle(center_mask, (patch_size // 2, patch_size//2), 10, 1, -1)
                
                
                
                test[0, 0,:, :] = test[0, 0,:, :] * (1 - torch.Tensor(center_mask).to(device))
                test[0, 1,:, :] = test[0, 1,:, :] * (1 - torch.Tensor(center_mask).to(device))

                canvas = visualize_offset(canvas, test[0, 0,:, :], test[0, 1,:, :])
                # combined_show = draw_mask_color(canvas, [0., 255., 0.])
                if visual:
                    cv2.imshow('short_offsets_map', canvas)
                canvas = np.zeros((patch_size, patch_size))
                canvas = visualize_offset(canvas, test[0, 2,:, :], test[0, 3,:, :])
                # combined_show = draw_mask_color(canvas, [0., 255., 0.])
                if visual:
                    cv2.imshow('mid', canvas)

                kp_mid_offsets = split_and_refine_mid_offsets(test[0,2:4,:,:].cpu().detach().numpy(), test[0,0:2,:,:].cpu().detach().numpy())
                canvas = np.zeros((patch_size, patch_size))

                canvas = visualize_offset(canvas, kp_mid_offsets[0, :, :], kp_mid_offsets[1, :, :])
                if visual:
                    cv2.imshow('mid_refine', canvas)

                # canvas = np.zeros((patch_size, patch_size))
                # canvas = visualize_offset(canvas, ground_truth_multi[0, t,3,:, :], ground_truth_multi[0, t, 4,:, :])
                # combined_show = draw_mask_color(canvas, [0., 255., 0.])
                # cv2.imshow('gt', canvas)
                # cv2.waitKey(0)
                # import pdb; pdb.set_trace()
                offset_next = get_next_point([patch_size//2,patch_size//2], kp_mid_offsets)
            
            show_NMS = LabelToColor(map_NMS.astype('uint8'))
            ##########################
            # convert to white?
            #################
            # bg = np.where(map_NMS==0)
            # show_NMS[bg[0], bg[1], :] = 255
            ##########################
            show_NMS = show_NMS.astype('uint8')

        # cv2.imshow('show_NMS', show_NMS)
        save_folder = './demo_output/' + cfg.save_weights_name
        os.makedirs(save_folder, exist_ok=True)

        cv2.imwrite(save_folder + '/' + meta['file_name'][0], show_NMS)


    valid_labels = np.unique(map_NMS)

    # import pdb; pdb.set_trace()
    
    for label_id in valid_labels:
        one_instance = {}
        if label_id == 0:
            continue
        # instance_map = (map_NMS == label_id)

        seg_poly = mask_to_pologons(map_NMS == label_id)
        if np.sum(map_NMS == label_id) < 20:
            continue
        mask_ind = np.where(map_NMS == label_id)

        bbox = [int(min(mask_ind[1])),\
            int(min(mask_ind[0])),\
                int(abs(max(mask_ind[1])- min(mask_ind[1]))),\
                    int(abs(max(mask_ind[0])- min(mask_ind[0])))]    

        one_instance['image_id'] = int(meta['image_id'].detach().cpu()) 
        one_instance['category_id'] = 1   
        one_instance['bbox'] = bbox     
        one_instance['segmentation'] = seg_poly       
        one_instance['score'] = 99  
        results.append(one_instance)
    print(time.time() - start)

    with open(result_file,'w') as wf:
        json.dump(results, wf)
        print('done')  

    # evaluate on COCO
    eval_gt = COCO(gt_file)

    eval_dt = eval_gt.loadRes(result_file)

 
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()      

    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()      