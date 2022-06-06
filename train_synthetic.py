import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention

from fiberDataset_COCO import *

from utils import *

from networks import network
from utility.loss import get_losses, short_offset_loss, mid_offset_loss, mask_loss
from config_Synthetic import cfg

# Data parameters
save_weights_name = cfg.save_weights_name
data_folder_p = cfg.data_folder_p  # base name shared by data files
jason_file_p = cfg.jason_file_p 

image_folder = cfg.image_folder 
offset_folder = cfg.offset_folder  
data_folder_val = cfg.data_folder_val 

jason_file_val  = cfg.jason_file_val 
image_folder_val = cfg.image_folder_val 
offset_folder_val = cfg.offset_folder_val

# Model parameters

emb_dim = 2 
attention_dim = 512 
decoder_dim = 512 
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
print("Let's use", torch.cuda.device_count(), "GPUs!")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = cfg.epochs  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = cfg.batch_size
workers = cfg.workers # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
decoderMulti_lr = 5e-4

decoderMulti_lr_weight_decay = 1e-5
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
best = 30000. 
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder
checkpoint = None  # path to checkpoint, None if none

multiGpu = cfg.multi_gpu
numofGpu = torch.cuda.device_count()
numofGpu = 3
output_size = (256,256)
def main():
    """
    Training and validation.
    """

    global best, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder

    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       # vocab_size=len(word_map),
                                       vocab_size=2, # X, Y coordinates and use it for regression
                                       dropout=dropout)
        
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

        decoderMulti = network.__dict__['MultiTask'](output_size)
        decoderMulti_optimizer = torch.optim.Adam(decoderMulti.parameters(),
                                 lr=decoderMulti_lr,
                                 weight_decay=decoderMulti_lr_weight_decay)

        encoder = Encoder()
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best = checkpoint['b4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']

        decoderMulti = checkpoint['decoderMulti']
        decoderMulti_optimizer = checkpoint['decoderMulti_optimizer']

        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available

    if multiGpu:

        decoder = torch.nn.DataParallel(decoder).to(device)
        encoder = torch.nn.DataParallel(encoder).to(device)
        decoderMulti = torch.nn.DataParallel(decoderMulti).to(device) 

    else:

        decoder = decoder.to(device)
        encoder = encoder.to(device)
        decoderMulti = decoderMulti.to(device)

    # Loss function
    criterionBinary = nn.BCELoss().to(device)
    criterionMse = nn.MSELoss().to(device)
    
    criterion = [criterionBinary, criterionMse]


    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader_p = torch.utils.data.DataLoader(
        fiberDataset_COCO(data_folder_p, jason_file_p, image_folder, offset_folder, transforms.Compose([transforms.ToTensor(), normalize]), True),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

    

    val_loader_p = torch.utils.data.DataLoader(
        fiberDataset_COCO(data_folder_val, jason_file_val, image_folder_val, offset_folder_val, transforms.Compose([transforms.ToTensor(), normalize]), True),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        train(train_loader=train_loader_p,
              encoder=encoder,
              decoder=decoder,
              decoderMulti = decoderMulti,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              decoderMulti_optimizer=decoderMulti_optimizer,
              epoch=epoch)

        recent = validate(val_loader=val_loader_p,
                                encoder=encoder,
                                decoder=decoder,
                                decoderMulti=decoderMulti,
                                criterion=criterion)
        #
        # # Check if there was an improvement
        is_best = recent < best
        best = min(recent, best)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            epochs_since_improvement = 0

        # Save checkpoint
            save_checkpoint(save_weights_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, decoderMulti, decoderMulti_optimizer, best, is_best)


def train(train_loader, encoder, decoder, decoderMulti, criterion, encoder_optimizer, decoder_optimizer, decoderMulti_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()
    decoderMulti.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading timew
    losses = AverageMeter()  # loss (per word decoded)
    loss_keypointses  = AverageMeter()
    losso_segmentationes = AverageMeter()  
    losso_mid_q = AverageMeter()  
    loss_shorts_q = AverageMeter()  
    # top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    # Batches
    for i, (imgs, sequence, sequence_offset, sequence_len, stop_sequence, ground_truth_multi, image_level_gt, meta) in enumerate(train_loader):
        '''
        input -> imgs
              -> patch_sequece
        Stage 1: 
            imgs -> embedding
        Stage 2 (can be pretrained): 
            embedding -> keypoints (end points + controlpoints)
                            -> disc heatmap + short offset map
        
        Stage 3:
            embedding + sequence patch
                -> segmentation + short offsets + next points
        #
        '''
        data_time.update(time.time() - start)
        # Move to GPU, if available
        imgs = imgs.to(device, dtype=torch.float)
        '''
        sequence = sequence_offset
        '''
        sequence_offset = sequence_offset.to(device)
        image_level_gt = image_level_gt.to(device)
        # ===================================
        sequence = sequence.to(device)
        sequence_len = sequence_len.to(device)
        stop_sequence = stop_sequence.to(device)

        short_offset_patches = ground_truth_multi[:,:,1:3,:,:]
        short_offset_patches = short_offset_patches.to(device)
        mid_offset_patches = ground_truth_multi[:,:,3:5,:,:]
        mid_offset_patches = mid_offset_patches.to(device)
        segmentation_patches = ground_truth_multi[:,:,0:1,:,:]
        segmentation_patches = segmentation_patches.to(device)
        # Forward prop.
        [imgs_encoded, x4,x3,x2,x1] = encoder(imgs)
        
        control_area_pred, end_area_pred, _ = decoderMulti([x4,x3,x2,x1])
        # import pdb; pdb.set_trace()
        loss_control_area = criterion[0](control_area_pred, image_level_gt[:,1:2,:,:])
        loss_end_area = criterion[0](end_area_pred, image_level_gt[:,0:1,:,:])
        loss_keypoints = (loss_control_area+ loss_end_area)

        sequence_length_or_decode_lengths, sort_ind = sequence_len.squeeze(1).sort(dim=0, descending=True)

        scores, predictionsStop, caps_sorted, decode_lengths, offset_prediction, segmentation_prediction, alphas, sort_ind = decoder(imgs_encoded, imgs, sequence, sequence_offset, sequence_len)

        decode_lengths = sequence_length_or_decode_lengths

        # import pdb; pdb.set_trace()
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:,:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        stop_sequence = stop_sequence[sort_ind]
        stop_sequence  = stop_sequence[:,:]

        short_offset_patches = short_offset_patches[sort_ind]
        mid_offset_patches = mid_offset_patches[sort_ind]
        segmentation_patches = segmentation_patches[sort_ind]

        predictionsStop = pack_padded_sequence(predictionsStop, decode_lengths, batch_first=True).data
        stopSequenceTarget = pack_padded_sequence(stop_sequence, decode_lengths, batch_first=True).data
        short_offset_patches_target = pack_padded_sequence(short_offset_patches, decode_lengths, batch_first=True).data
        mid_offset_patches_target = pack_padded_sequence(mid_offset_patches, decode_lengths, batch_first=True).data
        
        offset_prediction = pack_padded_sequence(offset_prediction, decode_lengths, batch_first=True).data
        sequenceTarget = pack_padded_sequence(sequence, decode_lengths, batch_first=True).data

        segmentation_patches_target = pack_padded_sequence(segmentation_patches, decode_lengths, batch_first=True).data
        segmentation_prediction = pack_padded_sequence(segmentation_prediction, decode_lengths, batch_first=True).data

        lossoffset = short_offset_loss(short_offset_patches_target, offset_prediction[:,0:2,:,:])
        lossoffset = lossoffset.to(device)

        lossoffset_mid = mid_offset_loss(mid_offset_patches_target, offset_prediction[:,2:4,:,:])
        lossoffset_mid = lossoffset_mid.to(device)

        losso_segmentation = mask_loss(segmentation_patches_target, segmentation_prediction)
        losso_segmentation = losso_segmentation.to(device)

        # import pdb; pdb.set_trace() 
        lossbi = criterion[0](predictionsStop, stopSequenceTarget.float().view(-1,1))
        # loss = 1.5 * lossoffset_mid + 2*lossbi +  losso_segmentation
        loss = lossbi +  1*losso_segmentation + 1*loss_keypoints + lossoffset_mid + lossoffset
        
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        decoderMulti_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        decoderMulti_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        
        # Keep track of metrics
        # top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        loss_keypointses.update(loss_keypoints.item(), batch_size)
        losso_segmentationes.update(losso_segmentation.item(), sum(decode_lengths))
        loss_shorts_q.update(lossoffset.item(), sum(decode_lengths))
        losso_mid_q.update(lossoffset_mid.item(), sum(decode_lengths))
            
        # top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:

            print('Epoch: [{0}][{1}/{2}]\t'
                    'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'loss_keypointses {loss_keypointses.val:.4f} ({loss_keypointses.avg:.4f})\t'
                    'losso_segmentationes {losso_segmentationes.val:.4f} ({losso_segmentationes.avg:.4f})\t'
                    'loss_shorts {loss_shorts.val:.4f} ({loss_shorts.avg:.4f})\t'
                    'losso_mid {losso_mid.val:.4f} ({losso_mid.avg:.4f})\t'
                    .format(epoch, i, len(train_loader), batch_time=batch_time,
                                            loss=losses,
                                            loss_keypointses=loss_keypointses,
                                            losso_segmentationes= losso_segmentationes,
                                            loss_shorts= loss_shorts_q,
                                            losso_mid= losso_mid_q))

def validate(val_loader, encoder, decoder, decoderMulti, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_keypointses  = AverageMeter()
    losso_segmentationes = AverageMeter()
    losso_mid_q = AverageMeter()  
    loss_shorts_q = AverageMeter()  

    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)


    with torch.no_grad():
        # Batches
        for i, (imgs, sequence, sequence_offset, sequence_len, stop_sequence, ground_truth_multi, image_level_gt, meta) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            sequence_offset = sequence_offset.to(device)
            image_level_gt = image_level_gt.to(device)
            # ===================================
            sequence = sequence.to(device)
            sequence_len = sequence_len.to(device)
            stop_sequence = stop_sequence.to(device)

            short_offset_patches = ground_truth_multi[:,:,1:3,:,:]
            short_offset_patches = short_offset_patches.to(device)
            mid_offset_patches = ground_truth_multi[:,:,3:5,:,:]
            mid_offset_patches = mid_offset_patches.to(device)
            segmentation_patches = ground_truth_multi[:,:,0:1,:,:]
            segmentation_patches = segmentation_patches.to(device)
            # Forward prop.
            [imgs_encoded, x4,x3,x2,x1] = encoder(imgs)
            # for global keypoint detection
            # 
            control_area_pred, end_area_pred, _ = decoderMulti([x4,x3,x2,x1])
            
            loss_control_area = criterion[0](control_area_pred, image_level_gt[:,1:2,:,:])
            loss_end_area = criterion[0](end_area_pred, image_level_gt[:,0:1,:,:])
            loss_keypoints = (loss_control_area+ loss_end_area)
           # Forward prop.

            # Forward prop.
            sequence_length_or_decode_lengths, sort_ind = sequence_len.squeeze(1).sort(dim=0, descending=True)

            scores, predictionsStop, caps_sorted, decode_lengths, offset_prediction, segmentation_prediction, alphas, sort_ind  = decoder(imgs_encoded, imgs, sequence, sequence_offset, sequence_len)
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:,:]

            short_offset_patches = short_offset_patches[sort_ind]
            mid_offset_patches = mid_offset_patches[sort_ind]
            segmentation_patches = segmentation_patches[sort_ind]
            decode_lengths = sequence_length_or_decode_lengths
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            short_offset_patches = pack_padded_sequence(short_offset_patches, decode_lengths, batch_first=True).data
            mid_offset_patches = pack_padded_sequence(mid_offset_patches, decode_lengths, batch_first=True).data

            predictionsStop = pack_padded_sequence(predictionsStop, decode_lengths, batch_first=True).data
            stopSequenceTarget = pack_padded_sequence(stop_sequence[sort_ind], decode_lengths, batch_first=True).data
            offset_prediction = pack_padded_sequence(offset_prediction, decode_lengths, batch_first=True).data

            segmentation_patches_target = pack_padded_sequence(segmentation_patches[sort_ind], decode_lengths, batch_first=True).data
            segmentation_prediction = pack_padded_sequence(segmentation_prediction, decode_lengths, batch_first=True).data


            # Calculate loss
            # lossmse = criterion[1](scores, targets)
            # import pdb; pdb.set_trace()
            lossoffset = short_offset_loss(short_offset_patches, offset_prediction[:,0:2,:,:])
            lossoffset_mid = mid_offset_loss(mid_offset_patches, offset_prediction[:,2:4,:,:])

            losso_segmentation = mask_loss(segmentation_patches_target, segmentation_prediction)
            losso_segmentation = losso_segmentation.to(device)
        
            lossbi = criterion[0](predictionsStop, stopSequenceTarget.float().view(-1,1))

            loss = lossbi +  losso_segmentation + loss_keypoints + 2 * lossoffset_mid + lossoffset

            # import pdb; pdb.set_trace()
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            loss_keypointses.update(loss_keypoints.item(), batch_size)
            losso_segmentationes.update(losso_segmentation.item(), sum(decode_lengths))

            batch_time.update(time.time() - start)

            start = time.time()
        
            losses.update(loss.item(), sum(decode_lengths))
            loss_keypointses.update(loss_keypoints.item(), batch_size)
            losso_segmentationes.update(losso_segmentation.item(), sum(decode_lengths))
            loss_shorts_q.update(lossoffset.item(), sum(decode_lengths))
            losso_mid_q.update(lossoffset_mid.item(), sum(decode_lengths))

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'loss_keypointses {loss_keypointses.val:.4f} ({loss_keypointses.avg:.4f})\t'
                      'losso_segmentationes {losso_segmentationes.val:.4f} ({losso_segmentationes.avg:.4f})\t'
                      'loss_shorts {loss_shorts.val:.4f} ({loss_shorts.avg:.4f})\t'
                      'loss_mid {loss_mid.val:.4f} ({loss_mid.avg:.4f})\t'
                      .format(i, len(val_loader), batch_time=batch_time,
                                                loss=losses,
                                                loss_keypointses=loss_keypointses,
                                                losso_segmentationes= losso_segmentationes,
                                                loss_shorts= loss_shorts_q,
                                                loss_mid= losso_mid_q))
    return loss.item()



if __name__ == '__main__':
    main()
