import torch
import torchvision.models as models
import DynTTA.DynTTA_augmentations as DynTTA_augmentations
import kornia
from URIE.skunet_model import SKUNet


class DynTTA(torch.nn.Module):
    def __init__(self, mean, std, urie_weight_path):
        super(DynTTA, self).__init__()
        self.mean = mean
        self.std = std
        self.augments = self.define_augmentations()
        self.backbone = models.resnet18(pretrained=True)
        num_output = 2 * len(self.augments)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_output)
        self.urie = SKUNet()
        urie_weight = torch.load(urie_weight_path)
        new_weights = {}
        for k, v in urie_weight.items():
            if "module." in k:
                new_weights[k.replace("module.", "")] = v
            else:
                new_weights[k] = v
        self.urie.load_state_dict(new_weights, strict=True)

    def define_augmentations(self):
        augments = [
            lambda image, mag: DynTTA_augmentations.rotate(image, mag, 30), 
            lambda image, mag: DynTTA_augmentations.scale(image, mag, 0.3),
            lambda image, mag: DynTTA_augmentations.saturation(image, mag, 5),
            lambda image, mag: DynTTA_augmentations.contrast(image, mag, 3),
            lambda image, mag: DynTTA_augmentations.sharpness(image, mag, 10),
            lambda image, mag: DynTTA_augmentations.brightness(image, mag, 0.6), 
            lambda image, mag: DynTTA_augmentations.auto_contrast(image, mag, None),
            lambda image, mag: DynTTA_augmentations.hue(image, mag, 2), 
            lambda image, mag: DynTTA_augmentations.equalize(image, mag, None),
            lambda image, mag: DynTTA_augmentations.invert(image, mag, None), 
            lambda image, mag: DynTTA_augmentations.gamma(image, mag, 3), 
        ]
        augments += [DynTTA_augmentations.lpf(i*0.05) for i in range(1,20)]
        augments += [DynTTA_augmentations.hpf(i*0.05) for i in range(1,20)]
        augments += [lambda image, mag: DynTTA_augmentations.URIE_augmentation(image, self.urie, mag, None)]
        return augments

    def forward(self, x):
        magnitudes = self.backbone(x)[:,:len(self.augments)]
        weights = self.backbone(x)[:,len(self.augments):]
        weights = torch.nn.functional.softmax(weights, dim=1)
        x = self.augmentation_layer(x, magnitudes, weights)
        return x, magnitudes, weights

    def augmentation_layer(self, x, magnitudes, weights):
        x_orig = kornia.denormalize(x, self.mean, self.std)
        x_orig = torch.clamp(x_orig, 0.0, 1.0)
        for i, augment in enumerate(self.augments):
            if i == 0:
                x = weights[:,i].view(-1,1,1,1) * augment(x_orig, magnitudes[:,i])
            else:
                x = x + weights[:,i].view(-1,1,1,1) * augment(x_orig, magnitudes[:,i])
        x = torch.clamp(x, 0.0, 1.0)
        x = kornia.normalize(x, self.mean, self.std)
        return x