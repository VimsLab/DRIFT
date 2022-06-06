## Edited version originally from here:
# https://github.com/iwyoo/tf-bilinear_sampler/blob/master/bilinear_sampler.py

import tensorflow as tf
import numpy as np

def bilinear_sampler(x, v):


  def _get_grid_array( H, W, h, w):
    # N_i = np.arange(N)
    H_i = np.arange(h+1, h+H+1)
    W_i = np.arange(w+1, w+W+1)
    h, w = np.meshgrid(H_i, W_i, indexing='ij')
    # h = np.expand_dims(h, axis=3) # [H, W, 1]
    # w = np.expand_dims(w, axis=3) # [H, W, 1]

    h = h.astype(np.float32) # [H, W, 1]
    w = w.astype(np.float32) # [H, W, 1]

    return h, w

  shape = np.shape(x) # TRY : Dynamic shape


  H_ = H = shape[1]
  W_ = W = shape[2]
  h = w = 0


  # x = np.pad(x,((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
  x = np.pad(x,((0,0),(1,1),(1,1)), 'constant')
  #
  # vx, vy = np.split(v, 2, axis=0)
  vx = v[0,:,:]
  vy = v[1,:,:]


  h, w = _get_grid_array(H, W, h, w) # [H, W, 3]

  vx0 = np.floor(vx)
  vy0 = np.floor(vy)
  vx1 = np.ceil(vx)
  vy1 = np.ceil(vy) # [H, W, 1]

  iy0 = vy0 + h
  iy1 = vy1 + h
  ix0 = vx0 + w
  ix1 = vx1 + w

  H_f = float(H_)
  W_f = float(W_)
  mask = np.less(ix0, 1)
  mask = np.logical_or(mask,np.greater(ix0, W_f))

  mask = np.logical_or(mask, np.less(iy0, 1))
  mask = np.logical_or(mask, np.greater(iy0, H_f))

  mask = np.logical_or(mask, np.greater(ix1, W_f))
  mask = np.logical_or(mask, np.less(ix1, 1))
  mask = np.logical_or(mask, np.greater(iy1, H_f))
  mask = np.logical_or(mask, np.less(iy1, 1))

  iy0 = np.where(mask, np.zeros_like(iy0), iy0)
  iy1 = np.where(mask, np.zeros_like(iy1), iy1)
  ix0 = np.where(mask, np.zeros_like(ix0), ix0)
  ix1 = np.where(mask, np.zeros_like(ix1), ix1)


  i00 = np.stack([iy0, ix0])  #[N , H, W ]
  i01 = np.stack([iy1, ix0]) #[N , H, W ]
  i10 = np.stack([iy0, ix1])#[N , H, W ]
  i11 = np.stack([iy1, ix1]) #[N , H, W ]
  i00 = i00.astype(np.int32)
  i01 = i01.astype(np.int32)
  i10 = i10.astype(np.int32)
  i11 = i11.astype(np.int32)

  iy0_reshape = iy0.reshape(-1).astype(np.int32)
  iy1_reshape = iy1.reshape(-1).astype(np.int32)
  ix0_reshape = ix0.reshape(-1).astype(np.int32)
  ix1_reshape = ix1.reshape(-1).astype(np.int32)

  x00 = np.zeros((2, H_,W_))
  x00_a = x[0,:,:]
  x00_b = x[1,:,:]
  x00[0,:,:] = x00_a[(iy0_reshape, ix0_reshape)].reshape((H_,W_))
  x00[1,:,:] = x00_b[(iy0_reshape, ix0_reshape)].reshape((H_,W_))

  x10 = np.zeros((2, H_,W_))
  x10_a = x[0,:,:]
  x10_b = x[1,:,:]
  x10[0,:,:] = x10_a[(iy1_reshape, ix0_reshape)].reshape((H_,W_))
  x10[1,:,:] = x10_b[(iy1_reshape, ix0_reshape)].reshape((H_,W_))

  x01 = np.zeros((2, H_,W_))
  x01_a = x[0,:,:]
  x01_b = x[1,:,:]
  x01[0,:,:] = x01_a[(iy0_reshape, ix1_reshape)].reshape((H_,W_))
  x01[1,:,:] = x01_b[(iy0_reshape, ix1_reshape)].reshape((H_,W_))

  x11 = np.zeros((2, H_,W_))
  x11_a = x[0,:,:]
  x11_b = x[1,:,:]
  x11[0,:,:] = x11_a[(iy1_reshape, ix1_reshape)].reshape((H_,W_))
  x11[1,:,:] = x11_b[(iy1_reshape, ix1_reshape)].reshape((H_,W_))



  # x00 = np.take(x, i00)
  # x01 = np.take(x, i01)
  # x10 = np.take(x, i10)
  # x11 = np.take(x, i11)

  dx = (vx - vx0).astype(np.float32)
  dy = (vy - vy0).astype(np.float32)

  w00 = (1.-dx) * (1.-dy)
  w01 = (1.-dx) * dy
  w10 = dx * (1.-dy)
  w11 = dx * dy

  output =w00*x00 + w01*x01 + w10*x10+ w11*x11


  return output
