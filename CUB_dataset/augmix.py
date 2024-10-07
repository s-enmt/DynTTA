# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch AugMix training on CIFAR-10/100.

Supports WideResNet, AllConv, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

#import augmentations

# from models.cifar.allconv import AllConvNet
import numpy as np
# from third_party.ResNeXt_DenseNet.models.densenet import densenet
# from third_party.ResNeXt_DenseNet.models.resnext import resnext29
# from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

from PIL import Image, ImageOps, ImageEnhance

from PIL import ImageDraw



def aug(image, preprocess):
  """Perform AugMix augmentations and compute mixture.

  Args:
    image: PIL.Image input image
    preprocess: Preprocessing function which should return a torch tensor.

  Returns:
    mixed: Augmented and mixed image.
  """
  mixture_width = 3
  mixture_depth = -1
  aug_severity = 5
  aug_list = [
      autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,
      translate_x, translate_y, 
      color, contrast, brightness, sharpness, # all
      # color, contrast, hue, gamma, # s_noise5 top4
      # color, contrast, HPF, gamma, # g_blur5%spatter5&saturate5 top4
  ]
  # aug_list = [
  #             # contrast, color, sharpness, autocontrast,
  #             # HPF, hue, gamma, 
  #             # invert, LPF, # s_nosie5, spatter5, saturate5
  #             # scale, # g_blur5
  #             # contrast, HPF, color, hue, gamma, # s_noise5&spatter5 top5
  #             # contrast, HPF, color, sharpness, gamma, # g_blur5&saturate5 top5
  #             # contrast, HPF, color, # common top3
  #             ]
  ws = np.float32(np.random.dirichlet([1] * mixture_width))
  m = np.float32(np.random.beta(1, 1))

  mix = torch.zeros_like(preprocess(image))
  for i in range(mixture_width):
    image_aug = image.copy()
    depth = mixture_depth if mixture_depth > 0 else np.random.randint(
        1, 4)
    for _ in range(depth):
      op = np.random.choice(aug_list)
      image_aug = op(image_aug, aug_severity)
    # Preprocessing commutes since all coefficients are convex
    mix += ws[i] * preprocess(image_aug)

  mixed = (1 - m) * preprocess(image) + m * mix
  return mixed


class AugMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform AugMix augmentation."""

  def __init__(self, dataset, preprocess, no_jsd=False):
    self.dataset = dataset
    self.preprocess = preprocess
    self.no_jsd = no_jsd
    # ImageNet code should change this value
    global IMAGE_SIZE
    assert self.dataset[0][0].width==self.dataset[0][0].height, f'width{self.dataset[0][0].width}, height{self.dataset[0][0].height}'
    IMAGE_SIZE = self.dataset[0][0].width

  def __getitem__(self, i):
    x, y = self.dataset[i]
    if self.no_jsd:
      return aug(x, self.preprocess), y
    else:
      im_tuple = (self.preprocess(x), aug(x, self.preprocess),
                  aug(x, self.preprocess))
      return im_tuple, y

  def __len__(self):
    return len(self.dataset)


def int_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval .

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    An int that results from scaling `maxval` according to `level`.
  """
  return int(level * maxval / 10)


def float_parameter(level, maxval):
  """Helper function to scale `val` between 0 and maxval.

  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.

  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
  return float(level) * maxval / 10.


def sample_level(n):
  return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
  return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
  return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
  level = int_parameter(sample_level(level), 4)
  return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
  degrees = int_parameter(sample_level(level), 30)
  if np.random.uniform() > 0.5:
    degrees = -degrees
  return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
  level = int_parameter(sample_level(level), 256)
  return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
  level = float_parameter(sample_level(level), 0.3)
  if np.random.uniform() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
  level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
  if np.random.random() > 0.5:
    level = -level
  return pil_img.transform((IMAGE_SIZE, IMAGE_SIZE),
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


### original

def scale(pil_img, level):
  w, h = pil_img.size
  level = float_parameter(sample_level(level), 1.2) + 0.7
  img = ImageOps.scale(pil_img, level)
  return _crop_center(img, w, h)
  

def invert(pil_img, _):
  return ImageOps.invert(pil_img)


def hue(pil_img, level):
  level = float_parameter(sample_level(level), 2.0) - 0.5
  return transforms.functional.adjust_hue(pil_img, level)


def gamma(pil_img, level):
  level = float_parameter(sample_level(level), 1.8) + 0.1
  return transforms.functional.adjust_gamma(pil_img, level)


def LPF(pil_img, level):
  shifted_f_uv = _fourier_img(pil_img)
  level = float_parameter(sample_level(level), 1.8) + 0.1
  h, w = shifted_f_uv.shape[-2:]
  mask = np.zeros([h,w])
  mask[_create_circular_mask(h, w, radius=level)] = 1
  filtered_f_uv = shifted_f_uv * mask
  return _pil_img(filtered_f_uv)


def HPF(pil_img, level):
  shifted_f_uv = _fourier_img(pil_img)
  level = float_parameter(sample_level(level), 1.8) + 0.1
  h, w = shifted_f_uv.shape[-2:]
  mask = np.ones([h,w])
  mask[_create_circular_mask(h, w, radius=level)] = 0
  filtered_f_uv = shifted_f_uv * mask
  return _pil_img(filtered_f_uv)


def _fourier_img(pil_img):
  np_img = np.asarray(pil_img)
  f_uv = np.fft.fft2(np_img)
  shifted_f_uv = np.fft.fftshift(f_uv)
  return shifted_f_uv

def _pil_img(filtered_f_uv):
  unshifted_f_uv = np.fft.fftshift(filtered_f_uv)
  i_f_xy = np.fft.ifft2(unshifted_f_uv).real  
  pil_img = Image.fromarray((i_f_xy).astype(np.uint8))
  return pil_img


def _create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_from_center /= dist_from_center.max()
    if radius == 0.0:
        mask = dist_from_center < radius
    else:
        mask = dist_from_center <= radius
    return mask

def _crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))