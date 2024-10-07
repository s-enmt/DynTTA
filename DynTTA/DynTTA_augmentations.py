from typing import Tuple
import kornia
import torch
from torch.autograd import Function
import numpy as np

### private functions
class _STE(Function):
    @staticmethod
    def forward(ctx,
                input_forward: torch.Tensor,
                input_backward: torch.Tensor) -> torch.Tensor:
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx,
                 grad_in: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return None, grad_in.sum_to_size(ctx.shape)


def ste(input_forward: torch.Tensor,
        input_backward: torch.Tensor) -> torch.Tensor:
    return _STE.apply(input_forward, input_backward).clone()


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: 
        center = [int(w/2), int(h/2)]
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    dist_from_center /= dist_from_center.max()
    if radius == 0.0:
        mask = dist_from_center < radius
    else:
        mask = dist_from_center <= radius
    return mask


### Augmentations
def rotate(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.tanh(mag) * range
    return kornia.rotate(img, mag)


def scale(img: torch.Tensor,
         mag: torch.Tensor,
        range: float) -> torch.Tensor:
    mag = torch.tanh(mag) * range
    mag = mag.view(-1 ,1)
    return kornia.scale(img, mag + 1)


def saturation(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.sigmoid(mag) * range
    return kornia.adjust_saturation(img, mag)


def contrast(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.sigmoid(mag) * range
    return kornia.adjust_contrast(img, mag)


def sharpness(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.sigmoid(mag) * range
    return kornia.sharpness(img, mag)


def brightness(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.tanh(mag) * range
    return kornia.adjust_brightness(img, mag)


def auto_contrast(img: torch.Tensor,
           mag=None,
           range=None) -> torch.Tensor:
    with torch.no_grad():
        reshaped = img.flatten(0, 1).flatten(1, 2).clamp(0, 1) * 255
        min, _ = reshaped.min(dim=1, keepdim=True)
        max, _ = reshaped.max(dim=1, keepdim=True)
        output = (torch.arange(256, device=img.device, dtype=img.dtype) - min) * (255 / (max - min + 0.1))
        output = output.floor().gather(1, reshaped.long()).reshape_as(img) / 255
    return ste(output, img)


def hue(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.tanh(mag) * range
    return kornia.adjust_hue(img, mag)


def equalize(img: torch.Tensor,
           mag=None,
           range=None) -> torch.Tensor:
    return kornia.equalize(img)


def invert(img: torch.Tensor,
           mag=None,
           range=None) -> torch.Tensor:
    return torch.max(img) - img


def gamma(img: torch.Tensor,
           mag: torch.Tensor,
           range: float) -> torch.Tensor:
    mag = torch.sigmoid(mag) * range
    return kornia.adjust_gamma(img, mag)


def low_pass_filter(img: torch.Tensor,
           filter_size: float,
           mag=None,
           range=None) -> torch.Tensor:
    fft_img = torch.fft.fft2(img)
    shift_fft_img = torch.fft.fftshift(fft_img)
    h, w = img.shape[-2:]
    mask = torch.zeros(h,w).to(device=img.device)
    mask[create_circular_mask(h, w, radius=filter_size)] = 1
    shift_fft_img = shift_fft_img * mask
    ishift_fft_img = torch.fft.ifftshift(shift_fft_img)
    ifft_img = torch.fft.ifft2(ishift_fft_img).real
    return ifft_img


def lpf(filter_size, range=None):
    return lambda img, mag: low_pass_filter(img, filter_size, mag, range)


def high_pass_filter(img: torch.Tensor,
           filter_size: float,
           mag=None,
           range=None) -> torch.Tensor:
    fft_img = torch.fft.fft2(img)
    shift_fft_img = torch.fft.fftshift(fft_img)
    h, w = img.shape[-2:]
    mask = torch.ones(h,w).to(device=img.device)
    mask[create_circular_mask(h, w, radius=filter_size)] = 0
    shift_fft_img = shift_fft_img * mask
    ishift_fft_img = torch.fft.ifftshift(shift_fft_img)
    ifft_img = torch.fft.ifft2(ishift_fft_img).real
    return ifft_img


def hpf(filter_size, range=None):
    return lambda img, mag: high_pass_filter(img, filter_size, mag, range)


def URIE_augmentation(img: torch.Tensor,
            URIE: torch.nn.Module,
           mag=None,
           range=None,
           ) -> torch.Tensor:
    img, _ = URIE(img)
    return img