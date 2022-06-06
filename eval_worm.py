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
import cv2
from PIL import Image

from fiberDataset_COCO import *


from utility.postprocess import split_and_refine_mid_offsets, get_next_point
from utils import LMFPeakFinder, mask_to_pologons
from copy import deepcopy
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from worm_COCO import *
from skimage.measure import label   
from utility.color_map import LabelToColor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getLargestCC(segmentation, points):
    labels = label(segmentation)
    largest_vote = labels[np.asarray(points[1]).astype('uint8'),np.asarray(points[0]).astype('uint8')]
    
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    # try:
    #     largestCC = labels == np.argmax(np.bincount(largest_vote)[1:])+1
    # except:
    #     import pdb; pdb.set_trace()
        # largestCC = labels == 1
    return largestCC

if __name__ == '__main__':

    result_file =  'result_worm.json'

    visual = cfg.visual
    max_len = 20
    parser = argparse.ArgumentParser(description='eval_worm')

    parser.add_argument('--model', '-m', help='path to model')

    args = parser.parse_args()

    data_folder_val = cfg.data_folder_val  # base name shared by data files

    jason_file_val = cfg.jason_file_test  # base name shared by data files

    image_folder_val = cfg.image_folder_val
    binary_folder_val = cfg.binary_folder

    output_size = (256,256)
    patch_size = cfg.patch_size
    batch_size = 1
    workers = 0
    gt_file = data_folder_val + '/' + jason_file_val
    eval_gt = COCO(gt_file)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image_folder = None

    val_loader_p = torch.utils.data.DataLoader(
            worm_COCO(data_folder_val, jason_file_val, image_folder_val, binary_folder_val, transforms.Compose([transforms.ToTensor(), normalize]), False),
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
    
    for i,  (imgs,img_org, meta) in enumerate(val_loader_p):
        print(len(val_loader_p))
        print(i)

        img_path = meta['w2_file_path']

        org_image = cv2.imread(img_path[0], cv2.COLOR_BGR2GRAY)
        org_image = np.asarray(org_image)

        crop_height = org_image.shape[0] % 32
        crop_width = org_image.shape[1]  % 32

        # org_image = org_image[int(crop_height / 2) : int(org_image.shape[0] - crop_height /2),int(crop_width / 2) : int(org_image.shape[1] - crop_width /2)]
        org_image = np.tile(org_image,(3,1,1)).transpose(1,2,0)
        org_image = cv2.normalize(org_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        org_image = (org_image * 255).astype('uint8')


        w2_show = cv2.normalize(org_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # im_show = cv2.imshow('test_show_org',org_image)
        # cv2.waitKey(0)
        if visual:
            im_show = cv2.imshow('test_show_org',org_image)
            cv2.waitKey(0)
        if len(org_image.shape) > 2:
            if org_image.shape[2] > 3:
                org_image = org_image[:,:,:3]

        # org_image_shape = org_image.shape[:3]
        # crop_edge =25
        crop_height = org_image.shape[0] % 32
        crop_width = org_image.shape[1]  % 32

        image_crop = w2_show[int(crop_height / 2) : int(org_image.shape[0] - crop_height /2),int(crop_width / 2) : int(org_image.shape[1] - crop_width /2)]
        
        # replace segmentation with gt for now, predited mask has different image size.


        imgs = imgs.to(device, dtype=torch.float)[0]

        imgs = imgs.unsqueeze(0)

        [encoder_out,x4,x3,x2,x1] = encoder(imgs)
        control_area_pred, end_area_pred, mask_pred = decoderMulti([x4,x3,x2,x1])
        if visual:
            cv2.imshow('control_area_pred', ((control_area_pred.squeeze().detach().cpu().numpy()>0.5) * 255).astype('uint8'))
            cv2.waitKey(0)
        # for mask 
        segmentation = 1 * (mask_pred.squeeze().detach().cpu().numpy() > 0.8)

        binary_file_path = meta['binary_file_path']

        binary_image = cv2.imread(binary_file_path[0],cv2.COLOR_BGR2GRAY)
        binary_image = (1. * (binary_image>0)).astype('uint8')
        binary_image = binary_image[:,:,0]     
        binary_image = np.asarray(binary_image).astype('uint8')

        # real_org_image = deepcopy(segmentation)
        real_org_image = deepcopy(binary_image)

        #######################################
        encoder_dim = encoder_out.size(-1)
        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        #######################################
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(control_area_pred.squeeze().detach().cpu().numpy()) # + org_image*0.0)
        rgba_img2 = cmap(end_area_pred.squeeze().detach().cpu().numpy()) #+ org_image*0.0)
        rgba_img_mask_pred = cmap(mask_pred.squeeze().detach().cpu().numpy()) #+ org_image*0.0)

        peak_finder = LMFPeakFinder(min_th=0.4)
        coord, score = peak_finder.detect(control_area_pred.squeeze().detach().cpu().numpy())
        coord_end, score = peak_finder.detect(end_area_pred.squeeze().detach().cpu().numpy())
        coord_end[:,1] += int(crop_width/2)
        coord_end[:,0] += int(crop_height/2)

        if visual:
            from copy import deepcopy

            draw_circle_end = np.zeros(org_image.shape[:2])
            draw_circle = np.zeros(org_image.shape[:2])

            for (x,y) in coord:
                draw_circle = cv2.circle(draw_circle, (int(y), int(x)), 8, (200,0,255), 1)
            
            draw_circle_show = draw_mask_color(org_image, draw_circle*255, [0,0,255])
            cv2.imshow('draw_circle',draw_circle_show)

            for (x,y) in coord_end:
                draw_circle_end= cv2.circle(draw_circle_end, (int(y), int(x)), 8, (255), 1)
            draw_circle_end = draw_mask_color(org_image, draw_circle_end*255, [0,0,255])
 
            cv2.imshow('draw_circle_end',draw_circle_end)
            cv2.waitKey(0)

            cv2.imshow('org_image',org_image)
            cv2.imshow('rgba_img_end',rgba_img2)
            cv2.imshow('rgba_img_control',rgba_img)
            cv2.imshow('rgba_img_mask_pred',rgba_img_mask_pred)
            cv2.imshow('segmentation',segmentation.astype('uint8')* 255.)
            cv2.waitKey(0)

        curr_image_NMS = []
        vis_each_step_idx = 0

        map_NMS = np.zeros(org_image.shape[:2]) 
        max_label = 1
        for (sp_row_y, sp_col_x) in coord_end: 

            start_point = torch.FloatTensor([[sp_col_x,sp_row_y]])
            start_point = start_point.unsqueeze(0).to(device)

            embeddings = decoder.patchEmbedding(img_org, start_point)
            embeddings = embeddings.squeeze(2)
            
            # h, c = decoder.init_hidden_state(encoder_out, embeddings[:batch_size,0,:])  # (batch_size, decoder_dim)
            h, c = decoder.init_hidden_state(embeddings[:batch_size,0,:])

            predictions = torch.zeros(batch_size, max_len, 2).to(device)
            predictionsStop = torch.zeros(batch_size, max_len, 1).to(device)
            alphas = torch.zeros(batch_size,max_len, num_pixels).to(device)

        
            offset_prediction = torch.zeros(batch_size, max_len, 2, patch_size, patch_size).to(device)    
        
            padding_size = patch_size // 2 + patch_size // 4
            
            
            image_padded = np.zeros(org_image.shape[:2])
            real_org_image_for_instance = deepcopy(real_org_image)

            org_image =  deepcopy(org_image)

            # org_image =  np.tile((real_org_image>0)*1.0,(3,1,1)).transpose(1,2,0)
            # org_image =  org_image.astype('uint8')
            

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
                
                attention_weighted_encoding, alpha = decoder.attention(encoder_out,
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
                    
                else:
                    contro_point_instance[0].append(pre.detach().cpu().numpy()[0][0])
                    contro_point_instance[1].append(pre.detach().cpu().numpy()[0][1])
                    offset_dist = (offset_next[0] ** 2 + offset_next[1] ** 2) ** 0.5

                    # print(offset_dist)
                    if offset_dist < 6 : #or predsStop > 0.8:
                        distance_to_end = np.sqrt(np.sum(np.square(non_visited_end_points - pre.detach().cpu().numpy()), axis = 1))
                        indx_closest = np.argmin(distance_to_end)  

                        turn_distance_to_end_off = True
                        if distance_to_end[indx_closest] < 10 or turn_distance_to_end_off:
                            end_point_instance.append(pre.detach().cpu().numpy())
                            image_padded = cv2.dilate(image_padded, np.ones((5,5)))
                            image_padded = cv2.erode(image_padded, np.ones((3,3)))
                            # image_padded = skeletonize(1.0 * (image_padded>0))
                            # image_padded = cv2.dilate(image_padded.astype('uint8'), np.ones((3,3)))
                            image_padded = image_padded * (1.0 * (real_org_image_for_instance>0))

                            # image_padded = draw_mask_color(255.0 * (real_org_image_for_instance>0), image_padded, [0,0,255])
                            # image_padded_ = draw_mask_color(np.tile(255.0 * (real_org_image_for_instance>0),(3,1,1)).transpose(1,2,0), image_padded, [0,0,255])
                            if np.sum(image_padded) < 5:
                                break                            
                            image_padded = getLargestCC(image_padded, contro_point_instance)        
                            image_padded = ((image_padded > 0) * 1.0).astype('uint8')
                            image_padded_ = draw_mask_color(org_image, image_padded, [0,0,255])

                            
                            if visual:
                                # import pdb; pdb.set_trace()
                                    print('break_stop')
                                    cv2.imshow('instance', image_padded_)
                                    cv2.waitKey(0)
                                    print('reached_confirm')
                       
                            
                            repeat_flag = False
                            ####################
                            # for i in range(len(curr_image_NMS)):
                            #     intersection = np.sum((curr_image_NMS[i]>0) * (image_padded > 0))
                            #     union = np.sum(curr_image_NMS[i] > 0) + np.sum(image_padded> 0)
                            #     iou = 2 * intersection/union
                            #     if iou > 0.70:
                            #         print('repeated')
                            #         print(iou)
                            #         repeat_flag = True
                            #         break
                            #####################
                            intersection = np.sum((map_NMS>0) * (image_padded > 0))
                            
                            curr_pixel = np.sum((image_padded>0))
                            if curr_pixel < 10:
                                # print('break current:',curr_pixel)
                                break
                            # overlap_rate = intersection/curr_pixel
                            # print('overlap_rate:', overlap_rate)
                            # if overlap_rate > 0.8:
                            #     repeat_flag = True
                            #     print('overlap_rate:', overlap_rate)
                            #     break
                            # if repeat_flag:
                            #     break
                            
                            # get over lap 

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
                                    # print('overlap')
                                    # print(intersection_pixels / myself_pixels)
                                    # print(intersection_pixels / curr_overlapped_pixels)
                                    asigned_label = label_id
                                    break

                            if not overlap_flag:
                                max_label += 1
                                
                            map_NMS[curr_pixels]=asigned_label
                            
                            curr_image_NMS.append(image_padded)

                            # back_to_original_size_image_padded = cv2.resize(image_padded, (org_image_shape[1] - crop_edge * 2,org_image_shape[0] - crop_edge * 2),interpolation = cv2.INTER_AREA)
                            # back_to_original_size_image_padded = np.pad(back_to_original_size_image_padded, [(crop_edge, crop_edge) , (crop_edge, crop_edge)], mode='constant', constant_values=0)
                            
                            # seg_poly = coco_mask_util.encode(np.asfortranarray(image_padded))
                            # image_padded = back_to_original_size_image_padded

                            seg_poly = mask_to_pologons(image_padded>0)
                            
                            mask_ind = np.where(image_padded>0)
                            if np.sum(image_padded) < 5:
                                
                                break

                            try:
                                bbox = [int(min(mask_ind[1])),\
                                    int(min(mask_ind[0])),\
                                        int(abs(max(mask_ind[1])- min(mask_ind[1]))),\
                                            int(abs(max(mask_ind[0])- min(mask_ind[0])))]
                            except:
                                import pdb; pdb.set_trace()

                        break
                    # import pdb; pdb.set_trace()
                    curr = pre + torch.Tensor(offset_next).unsqueeze(0).to(device)

                    if  curr[0][0] < 0 or curr[0][1] < 0 or curr[0][0] >= img_org.shape[-1] or curr[0][1] >= img_org.shape[-2]:
                        # print('out_of_bound')
                        # import pdb; pdb.set_trace()
                        distance_to_end = np.sqrt(np.sum(np.square(non_visited_end_points - pre.detach().cpu().numpy()), axis = 1))
                        # distance_to_end_curr = np.sqrt(np.sum(np.square(non_visited_end_points - curr.detach().cpu().numpy()), axis = 1))
                        indx_closest = np.argmin(distance_to_end)   
                        # print('break_over')
                        # print(distance_to_end[indx_closest]) 
                        # if distance_to_end[indx_closest] < 25:
                        if True:
                            end_point_instance.append(pre.detach().cpu().numpy())
                            image_padded = cv2.dilate(image_padded, np.ones((5,5)))
                            image_padded = cv2.erode(image_padded, np.ones((3,3)))
                            # image_padded = skeletonize(1.0 * (image_padded>0))
                            # image_padded = cv2.dilate(image_padded.astype('uint8'), np.ones((3,3)))

                            image_padded = image_padded * (1.0 * (real_org_image_for_instance>0))

                            if np.sum(image_padded) < 5:
                                break    
                            image_padded = getLargestCC(image_padded, contro_point_instance)        
                            image_padded = ((image_padded > 0) * 1.0).astype('uint8')

                            # image_padded_ = draw_mask_color(np.tile(255.0 * (real_org_image_for_instance>0),(3,1,1)).transpose(1,2,0), image_padded, [0,0,255])
                            image_padded_ = draw_mask_color(org_image, image_padded, [0,0,255])


                            # image_padded = image_padded * (1.0 * (real_org_image_for_instance>0))

                            if visual:
                                cv2.imshow('instance', image_padded_)
                                cv2.waitKey(0)
                                print('reached_confirm')   

                            repeat_flag = False
                            ####################
                            # for i in range(len(curr_image_NMS)):
                            #     intersection = np.sum((curr_image_NMS[i]>0) * (image_padded > 0))
                            #     union = np.sum(curr_image_NMS[i] > 0) + np.sum(image_padded> 0)
                            #     iou = 2 * intersection/union
                            #     if iou > 0.70:
                            #         print('repeated')
                            #         print(iou)
                            #         repeat_flag = True
                            #         break
                            #####################
                            # intersection = np.sum((map_NMS>0) * (image_padded > 0))
                            curr_pixel = np.sum((image_padded>0))
                            if curr_pixel < 10:
                                print('break current:',curr_pixel)
                                break

                            if repeat_flag:
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
            
                            curr_image_NMS.append(image_padded)
                            # seg_poly = coco_mask_util.encode(np.asfortranarray(image_padded))
                            

                    
                            seg_poly = mask_to_pologons(image_padded>0)
                            
                            mask_ind = np.where(image_padded>0)

                            if np.sum(image_padded) < 5:        
                                break
                            bbox = [int(min(mask_ind[1])),\
                                int(min(mask_ind[0])),\
                                    int(abs(max(mask_ind[1])- min(mask_ind[1]))),\
                                        int(abs(max(mask_ind[0])- min(mask_ind[0])))]

                            # one_instance['image_id'] = int(meta['image_id'].detach().cpu()) 
                            # one_instance['category_id'] = 1   
                            # one_instance['bbox'] = bbox     
                            # one_instance['segmentation'] = seg_poly       
                            # one_instance['score'] = 99  
                            # results.append(one_instance)

                            if visual:
                                if predsStop > 0.8:
                                # import pdb; pdb.set_trace()
                                    print('break_stop')
                                    cv2.imshow('instance', image_padded_)
                                    cv2.waitKey(0)
                                    print('reached_confirm')

                        # print('end_point')
                        break                    
                    # distance_to_control = np.sqrt(np.sum(np.square(non_visited - curr.detach().cpu().numpy()), axis = 1))

                    pre = curr

                    curr = curr.unsqueeze(0)
                    
                    # import pdb; pdb.set_trace()
                    #############################


                    embeddings = decoder.patchEmbedding(img_org, curr) #

                    embeddings_2 = embeddings.squeeze(2)
                    embeddings_2 = embeddings_2[:batch_size_t,0,:]
                    # embeddings_2 = embeddings[:batch_size,t,:]
                    ######################################
                    # import pdb; pdb.set_trace()
                    h, c = decoder.decode_step(
                        embeddings_2,
                        (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
       
                patch_Seg = decoder.sigmoid(decoder.patchSegmentationDecoder(decoder.dropout(h)))        

                if t == 0:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                else:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                center = [coor_x, coor_y]
                # shift_x = random.randint(0,self.patch_size // 4)
                # shift_y = random.randint(0,self.patch_size // 4)
                shift_x = 0
                shift_y = 0
                center[0] = center[0] + shift_x
                center[1] = center[1] + shift_y

                x_coor_st = int(center[0].item() - patch_size//2 + patch_size)
                x_coor_ed = int(center[0].item() + patch_size//2 + patch_size)
                y_coor_st = int(center[1].item() - patch_size//2 + patch_size)
                y_coor_ed = int(center[1].item() + patch_size//2 + patch_size)
                indx = np.where(patch_Seg.detach().cpu().numpy().squeeze() > 0.5)

                padding_size = patch_size//2 + patch_size // 4
                indx = [(indx[0]+int(coor_y)-patch_size//2),(indx[1]+int(coor_x))-patch_size//2]
                # indx = [(indx[0]+int(coor_y)-padding_size//2),(indx[1]+int(coor_x))-padding_size//2]
                
                # import pdb; pdb.set_trace()
                indx[0][indx[0]>= org_image.shape[0]] = org_image.shape[0]-1
                indx[0][indx[0]<0] = 0
                indx[1][indx[1]>= org_image.shape[1]] = org_image.shape[1]-1
                indx[1][indx[1]<0] = 0
                # org_image[indx] = 255.//2
                # image_padded[x_coor_st :x_coor_ed,  y_coor_st :y_coor_ed] = (patch_Seg.detach().cpu().numpy().squeeze() > 0.5) * 255. // 2
                image_padded[tuple(indx)] = 255
                image_padded = cv2.circle(image_padded, (int(center[0]), int(center[1])), 5, 255, -1)
                if visual:

                    padding_size = patch_size//2 + patch_size // 4
                    org_image_visual = np.pad(org_image, [(padding_size, padding_size) , (padding_size, padding_size), (0,0)], mode='constant', constant_values=0)
                    
                    x_coor_st = int(center[0].item() - patch_size//2 + padding_size)
                    x_coor_ed = int(center[0].item() + patch_size//2 + padding_size)
                    y_coor_st = int(center[1].item() - patch_size//2 + padding_size)
                    y_coor_ed = int(center[1].item() + patch_size//2 + padding_size)
                    org_image_visual = org_image_visual[y_coor_st :y_coor_ed, x_coor_st :x_coor_ed]
                    cv2.imshow('test2', (patch_Seg.detach().cpu().numpy().squeeze() > 0.5) * 255.)
                    org_image_visual = draw_mask_color(org_image_visual, (patch_Seg.detach().cpu().numpy().squeeze() > 0.3) * 255., [0,55,255])
                    cv2.circle(org_image_visual, (patch_size//2, patch_size//2), 8, (0,255,255), 1)
                    cv2.imshow('test3', image_padded)
                    cv2.imshow('org_image_visual', org_image_visual)

                    cv2.imwrite('demo_output/worm_demo/'+str(vis_each_step_idx) + 'org_image_visual.png', org_image_visual)
                    cv2.waitKey()

                test = decoder.patchDecoder(decoder.dropout(h))      
                predsStop = decoder.fcStop(decoder.dropout(h))  # (batch_size_t, vocab_size)
                predsStop = decoder.fcStop_act(predsStop)
                if predsStop > 0.8:
                        # import pdb; pdb.set_trace()
                        print('break_stop')
                        # print(predsStop)
                #         break                
                # predictions[:batch_size_t, t, :] = preds
                predictionsStop[:batch_size_t, t, :] = predsStop
                alphas[:batch_size_t, t, :] = alpha

                if t == 0:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()
                else:
                    coor_y = pre[0][1].detach().cpu().numpy()
                    coor_x = pre[0][0].detach().cpu().numpy()



                if visual:
                    org_image = cv2.circle(org_image.astype('uint8'), (int(max(0, min(coor_x, org_image.shape[1]))), int(max(0, min(coor_y, org_image.shape[0])))), 5, (200,0,0), 1)
                
                # # import pdb; pdb.set_trace()
                    cv2.imshow('test', org_image)
                    cv2.waitKey()
                canvas = np.zeros((patch_size, patch_size))
                center_mask = np.zeros((patch_size,patch_size))
                center_mask = cv2.circle(center_mask, (patch_size // 2, patch_size//2), 5, 1, -1)
                test[0, 0,:, :] = test[0, 0,:, :] * (1 - torch.Tensor(center_mask).to(device))
                test[0, 1,:, :] = test[0, 1,:, :] * (1 - torch.Tensor(center_mask).to(device))


                if visual:
                    canvas = visualize_offset(canvas, test[0, 0,:, :], test[0, 1,:, :])
                    # combined_show = draw_mask_color(canvas, [0., 255., 0.])
                    canvas = draw_mask_color(org_image_visual, canvas, [0., 255., 0.])
 
                    cv2.imshow('short_offsets_map', canvas)
                    cv2.imwrite('demo_output/worm_demo/'+str(vis_each_step_idx) + 'short_offsets_map.png', canvas)
               
                if visual:
                    canvas = np.zeros((patch_size, patch_size))
                    canvas = visualize_offset(canvas, test[0, 2,:, :], test[0, 3,:, :])
                    # combined_show = draw_mask_color(canvas, [0., 255., 0.])
                    canvas = draw_mask_color(org_image_visual, canvas, [0., 255., 0.])
    
                    cv2.imshow('mid', canvas)
                    cv2.imwrite('demo_output/worm_demo/'+str(vis_each_step_idx) + 'mid.png', canvas)
                
                kp_mid_offsets = split_and_refine_mid_offsets(test[0,2:4,:,:].cpu().detach().numpy(),\
                     test[0,0:2,:,:].cpu().detach().numpy(),num_steps = 1)
                
                if visual:
                    canvas = np.zeros((patch_size, patch_size))

                    canvas = visualize_offset(canvas, kp_mid_offsets[0, :, :], kp_mid_offsets[1, :, :])
                    canvas = draw_mask_color(org_image_visual, canvas, [0., 255., 0.])

                    cv2.imshow('mid_refine', canvas)
                    cv2.imwrite('demo_output/worm_demo/'+str(vis_each_step_idx) + 'mid_refine.png', canvas)
                
                # canvas = np.zeros((patch_size, patch_size))
                # canvas = visualize_offset(canvas, ground_truth_multi[0, t,3,:, :], ground_truth_multi[0, t, 4,:, :])
                # combined_show = draw_mask_color(canvas, [0., 255., 0.])
                # cv2.imshow('gt', canvas)
                # cv2.waitKey(0)
                # import pdb; pdb.set_trace()
                
                offset_next = get_next_point([patch_size//2,patch_size//2], kp_mid_offsets, vis_each_step_idx)
                vis_each_step_idx += 1
            
            show_NMS = LabelToColor(map_NMS.astype('uint8'))
            bg = np.where(map_NMS==0)
            show_NMS[bg[0], bg[1], :] = 255
            show_NMS = show_NMS.astype('uint8')

            # cv2.imshow('show_NMS', show_NMS)
            
            cv2.imwrite('./demo_output/demo_worms/demo_'+meta['binary_file_path'][0].split('/')[-1], show_NMS)
            # print(meta['binary_file_path'][0])
            # cv2.waitKey(0)

       

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

    with open(result_file,'w') as wf:
        json.dump(results, wf)
        print('done')  

    # evaluate on COCO
    eval_gt = COCO(gt_file)

    eval_dt = eval_gt.loadRes(result_file)

    ##############################################
    #visualize
    # import pdb; pdb.set_trace()
    # for idx in range(len(eval_dt.anns)+1):
    #     if idx == 0: 
    #         continue
    #     ann = eval_dt.anns[idx]


    #     file_path = eval_gt.loadImgs(ann['image_id'])[0]['w2']
    #     file_path = os.path.join(image_folder_val, file_path)


    #     image_w2 = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)
    #     image_w2 = np.asarray(image_w2)
    #     im = np.tile(image_w2,(3,1,1)).transpose(1,2,0)
    #     im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #     im = (im*255).astype('uint8') 
    #     org_img_shape = im.shape[:-1]
    #     # crop_edge = 25
    #     # im = im[crop_edge:im.shape[0] - crop_edge, crop_edge : im.shape[1] - crop_edge, :]
    #     # im = cv2.resize(im, (256,256), interpolation = cv2.INTER_AREA)
    #     im_h, im_w, _ = im.shape
    
    #     canvas = np.zeros(im.shape,dtype = np.float32) 
    #     bbox = ann['bbox']
    #     x1, y1, w, h = bbox
    #     x2 = x1 + w
    #     y2 = y1 + h
    #     category_id = ann['category_id']
    #     mask = pologons_to_mask(ann['segmentation'],im.shape[:-1])
    #     canvas = draw_mask_color(im, mask, [0,255,155])
    #     canvas = cv2.rectangle(canvas, (x1,y1), (x2,y2), [100,255,155], 2) 
    #     cv2.imshow('tttt', canvas)
    #     cv2.waitKey(0)

    #     ann_gt_ids = eval_gt.getAnnIds(imgIds=ann['image_id'])
    #     anns = eval_gt.loadAnns(ann_gt_ids)
    #     canvas = np.zeros(im.shape,dtype = np.float32) 
    #     canvas_demo = np.zeros(im.shape,dtype = np.float32) 
    #     id_for_demo = 1
    #     canvas = im
    #     print(len(anns))
    #     for ann_gt in anns:
            
    #         mask = pologons_to_mask(ann_gt['segmentation'],org_img_shape)
    #         canvas_demo[mask>0] = id_for_demo
    #         id_for_demo+=1

    #         # mask = mask[crop_edge:mask.shape[0] - crop_edge, crop_edge : mask.shape[1] - crop_edge]
    #         # mask = cv2.resize(mask, (256,256), interpolation = cv2.INTER_AREA)
    #         bbox = ann_gt['bbox']
    #         x1, y1, w, h = bbox
    #         x2 = x1 + w
    #         y2 = y1 + h            
    #         canvas = draw_mask_color(canvas, mask, [0,100,155])
    #         mask = pologons_to_mask(ann['segmentation'],im.shape[:-1])
            
    #         # canvas = cv2.rectangle(canvas, (x1,y1), (x2,y2), [100,255,155], 2) 
    #     canvas = draw_mask_color(canvas, mask, [0,0,255])
    #     cv2.imshow('tt', canvas)
    #     cv2.waitKey(0)       

    #     canvas_demo = LabelToColor(canvas_demo.astype('uint8'))
    #     bg = np.where(map_NMS==0)
    #     show_NMS[bg[0], bg[1], :] = 255
    #     show_NMS = show_NMS.astype('uint8')

    #     cv2.imshow('show_NMS', show_NMS)
        
    #     cv2.imwrite('./demo_output/demo_worms/demo_'+meta['binary_file_path'][0].split('/')[-1], show_NMS)
    #     print(meta['binary_file_path'][0])
    #     cv2.waitKey(0)


    ###################################################

    # import pdb; pdb.set_trace()
    # ann_ids = eval_gt.getAnnIds(imgIds=[0,1,2])
    # import pdb; pdb.set_trace()
    # anns = eval_gt.loadAnns(ann_ids)
    
    # anns = eval_gt.loadAnns(ann_ids)
    # eval_dt = eval_gt.loadRes(anns)
    
    ##################################################################################
    cocoEval = COCOeval(eval_gt, eval_dt, iouType='segm')


    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()      

    cocoEval = COCOeval(eval_gt, eval_dt, iouType='bbox')

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()      

  