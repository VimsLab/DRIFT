import os
import numpy as np
import h5py
import json
import torch
import cv2

from torch import nn
from imageio import imread
from skimage.transform import resize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.cluster.hierarchy import fclusterdata

def mask_to_pologons(mask):
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    # import pdb;pdb.set_trace()
    contours,_= cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        contour = contour+0.5
        contour = contour.flatten().tolist()
        if len(contour) >= 6:
            polygons.append(contour)
    return polygons


class LMFPeakFinder(object):
    """
    borrow from https://stackoverflow.com/a/3689710
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    def __init__(self, min_dist=5., min_th=0.3):
        self.min_dist = min_dist
        self.min_th = min_th

    def detect(self, image):
        # define an 8-connected neighborhood
        neighborhood = generate_binary_structure(2, 2)

        # apply the local maximum filter; all pixel of maximal value
        # in their neighborhood are set to 1
        local_max = maximum_filter(image, footprint=neighborhood) == image
        # local_max is a mask that contains the peaks we are
        # looking for, but also the background.
        # In order to isolate the peaks we must remove the background from the mask.

        # we create the mask of the background
        background = (image < self.min_th)

        # a little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will
        # appear along the background border (artifact of the local maximum filter)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

        # we obtain the final mask, containing only peaks,
        # by removing the background from the local_max mask (xor operation)
        detected_peaks = local_max ^ eroded_background

        detected_peaks[image < self.min_th] = False
        peaks = np.array(np.nonzero(detected_peaks)).T

        if len(peaks) == 0:
            return peaks, np.array([])

        # nms
        if len(peaks) == 1:
            clusters = [0]
        else:
            clusters = fclusterdata(peaks, self.min_dist, criterion="distance")
        peak_groups = {}
        for ind_junc, ind_group in enumerate(clusters):
            if ind_group not in peak_groups.keys():
                peak_groups[ind_group] = []
                peak_groups[ind_group].append(peaks[ind_junc])
        peaks_nms = []
        peaks_score = []
        for peak_group in peak_groups.values():
            values = [image[y, x] for y, x in peak_group]
            ind_max = np.argmax(values)
            peaks_nms.append(peak_group[int(ind_max)])
            peaks_score.append(values[int(ind_max)])

        return np.float32(np.array(peaks_nms)), np.float32(np.array(peaks_score))



class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
 

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    decoderMulti, decoderMulti_optimizer, bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoderMulti': decoderMulti,
             'decoderMulti_optimizer': decoderMulti_optimizer,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
