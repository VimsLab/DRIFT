import sys
import numpy as np
import cv2


def bitget(bitset, pos):
    offset = pos
    return bitset >> offset & 1


def bitshift(bitset, offset):
    if offset > 0:
        return bitset << offset
    else:
        return bitset >> (-offset)


def bitor(bitset_l, bitset_r):
    return bitset_l | bitset_r


def bitwise_get(bitset, pos):
    return np.bitwise_and(np.right_shift(bitset, pos), 1)


def GenColorMap(num_classes):
    color_map = []
    for i in range(num_classes):
        id = i
        r = 0
        g = 0
        b = 0
        for j in range(8):
            r = bitor(r, bitshift(bitget(id, 0), 7 - j))
            g = bitor(g, bitshift(bitget(id, 1), 7 - j))
            b = bitor(b, bitshift(bitget(id, 2), 7 - j))
            id = bitshift(id, -3)
        color_map.append((b, g, r))
    class_2_color = color_map
    color_2_class = dict([(item[1], item[0]) for item in enumerate(color_map)])
    return class_2_color, color_2_class


def LabelToColor(im):
    '''
        input:
            im: numpy array of integer type
        output:
            color_map: numpy array with 3 channels
    '''
    inds = im.copy()
    r = np.zeros(im.shape, dtype=np.uint8)
    g = np.zeros(im.shape, dtype=np.uint8)
    b = np.zeros(im.shape, dtype=np.uint8)
    for i in range(8):
        np.bitwise_or(r, np.left_shift(bitwise_get(inds, 0), 7 - i), r)
        np.bitwise_or(g, np.left_shift(bitwise_get(inds, 1), 7 - i), g)
        np.bitwise_or(b, np.left_shift(bitwise_get(inds, 2), 7 - i), b)
        np.right_shift(inds, 3, inds)
    color_map = cv2.merge([b, g, r])
    return color_map


def ColorToLabel(color_map):
    '''
        input:
            color_map: numpy array with 3 channels
        output:
            inds: numpy array of class indices
    '''
    inds = np.zeros(color_map.shape[:2], dtype=np.uint8)
    r = color_map[:, :, 2]
    g = color_map[:, :, 1]
    b = color_map[:, :, 0]
    for i in range(8):
        r_ = np.left_shift(bitwise_get(r, i), (7 - i) * 3)
        g_ = np.left_shift(bitwise_get(g, i), (7 - i) * 3 + 1)
        b_ = np.left_shift(bitwise_get(b, i), (7 - i) * 3 + 2)
        np.bitwise_or(inds, r_, inds)
        np.bitwise_or(inds, g_, inds)
        np.bitwise_or(inds, b_, inds)
    return inds


if __name__ == '__main__':
    color_map = cv2.imread(sys.argv[1])
    inds = ColorToLabel(color_map)
    reverse_color_map = LabelToColor(inds)
    assert ((color_map == reverse_color_map).all() == True)
    print(GenColorMap(255))
