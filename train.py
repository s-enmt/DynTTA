import torch
from DynTTA.DynTTA import DynTTA
from Identity.Identity import Identity
from URIE.skunet_model import SKUNet
from torchvision import models
import torch.optim as optim
import os, argparse
from torch.utils.data import DataLoader
from CUB_dataset.cub2011 import Cub2011
from CUB_dataset.corruption_dataset import Corruption_dataset, Val_Corruption_cub
import torchvision.transforms as transforms
from CUB_dataset.augmix import AugMixDataset
from tqdm import tqdm
import json


parser = argparse.ArgumentParser(description='Trains a CUB Enhancer')
parser.add_argument('--gpu', default='0,1')
parser.add_argument('--num_workers', '--cpus', default=16, type=int)
# dataset:
parser.add_argument('--dataset', '--ds', default='CUB', choices=["CUB"], help='which dataset to use')
parser.add_argument('--data_root_path', '--drp',  default='../dataset', help='Where you save all your datasets.')
# setting 
parser.add_argument('--blind', action='store_true', help='Blind Setting')
# classification model
parser.add_argument('--model', '--md', default='R50', help='which model to use')
parser.add_argument('--classification_model_weight', default="./pretrained_classification_models/CUB/r50.pth", help='classification model weight path')
# enhancement model
parser.add_argument('--enhancer', default="DynTTA", choices=["DynTTA", "URIE", "Identity"], help='which enhancer to use')
parser.add_argument('--urie_weight', default="./URIE/ECCV_SKUNET_OURS.ckpt.pt", help='urie weight')
parser.add_argument('--ckp_dir', default="./ckp/", help='which model to use')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=60, help='Number of epochs to train.')
parser.add_argument('--opt', default='sgd', choices=['sgd', 'adam'], help='which optimizer to use')
parser.add_argument('--decay', default='steps', choices=['cos', 'multisteps', 'steps'], help='which lr decay method to use')
parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size for training.')
parser.add_argument('--test_batch_size', '-tb', type=int, default=256, help='Batch size for validation.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--wd', type=float, default=1e-3, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.5, help='decay gamma.')
parser.add_argument('--d_step', type=int, default=10, help='decay step.')

args = parser.parse_args()


# set CUDA:
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

def main():  
    # save dir
    if not os.path.exists(args.ckp_dir):
        os.makedirs(args.ckp_dir)
    save_dir = os.path.join(args.ckp_dir, args.dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # dataset
    if args.dataset == "CUB":
        num_classes = 200
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        geometric_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
        ])
        tensor_preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_dataset = Cub2011(
            root=args.data_root_path, 
            transform=geometric_transform,
            train=True,
        )
        # Blind Setting
        if args.blind:
            train_dataset = AugMixDataset(
                train_dataset,
                preprocess=tensor_preprocessing,
                no_jsd=True,
            )
        # Non-Blind Setting
        else:
            train_dataset = Corruption_dataset(
                train_dataset,
                prob=15/16,
                transform=tensor_preprocessing,
            )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        clean_val_data = Cub2011(
            root=args.data_root_path, 
            transform=test_transform,
            train=False,
        )
        clean_val_loader = DataLoader(clean_val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        corruption_val_data = Val_Corruption_cub(
            corruption_root=f'{args.data_root_path}/Corruption_CUB/', 
            csv_path = 'CUB_dataset/val_corrupution_cub.csv',
            transform=test_transform,
        )
        corruption_val_loader = DataLoader(corruption_val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError 

    # classification model
    if args.model == 'R18':
        classification_model = models.resnet18(pretrained=False)
        classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, num_classes)
    elif args.model == 'R50':
        classification_model = models.resnet50(pretrained=False)
        classification_model.fc = torch.nn.Linear(classification_model.fc.in_features, num_classes)
    else:
        raise NotImplementedError
    classification_model = torch.nn.DataParallel(classification_model)
    classification_model.load_state_dict(torch.load(args.classification_model_weight))
    classification_model = classification_model.cuda().eval()

    # enhancement model
    if args.enhancer == "DynTTA":
        assert "ECCV_SKUNET_OURS" not in args.urie_weight or not args.blind, "This urie weight was trained in the non-blind setting."
        enhancement_model = DynTTA(mean, std, args.urie_weight)
        enhancement_model.backbone = torch.nn.DataParallel(enhancement_model.backbone)
        enhancement_model.urie = torch.nn.DataParallel(enhancement_model.urie)
    elif args.enhancer == "URIE":
        enhancement_model = SKUNet()
        enhancement_model = torch.nn.DataParallel(enhancement_model)
    elif args.enhancer == "Identity":
        enhancement_model = Identity()
        enhancement_model = torch.nn.DataParallel(enhancement_model)
        args.epochs = 0
        print("Perform only inference without training")
    else:
        raise NotImplementedError
    enhancement_model = enhancement_model.cuda().train()

    # training enhancer
    # optimizer & scheduler
    if args.enhancer != "Identity":
        print(f"Start training {args.enhancer}")
        if args.opt == 'sgd':
            optimizer = optim.SGD(enhancement_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        elif args.opt == 'adam':
            optimizer = optim.Adam(enhancement_model.parameters(), lr=args.lr, weight_decay=args.wd)
        if args.decay == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        elif args.decay == 'steps':
            scheduler = optim.lr_scheduler.StepLR(optimizer, args.d_step, gamma=args.gamma)
    
        # criterion
        criterion = torch.nn.CrossEntropyLoss().cuda()

        # train:
        for epoch in range(1, args.epochs+1):
            enhancement_model.train()
            for images, labels in tqdm(train_loader):
                images = images.cuda()
                labels = labels.cuda()
                
                enhancer_outputs = enhancement_model(images)
                if args.enhancer == "DynTTA":
                    enhanced_images, magnitudes, weights = enhancer_outputs
                elif args.enhancer == "URIE":
                    enhanced_images, diff = enhancer_outputs
                else:
                    enhanced_images = enhancer_outputs

                outputs = classification_model(enhanced_images)

                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # lr schedualr update at the end of each epoch:
            scheduler.step()

            # print & save
            if epoch%10 == 0 or epoch==args.epochs:
                print(f"epoch: {epoch}, loss={loss.item()}")

                # save checkpoint
                setting = "Blind" if args.blind else "NonBlind"
                torch.save(enhancement_model.state_dict(), os.path.join(save_dir, f"{setting}_{args.enhancer}_latest.pth"))

    # validation
    # val for clean
    print("Start validation")
    enhancement_model.eval()
    acc = 0
    for images, labels in tqdm(clean_val_loader):
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            enhancer_outputs = enhancement_model(images)
            if args.enhancer == "DynTTA":
                enhanced_images, magnitudes, weights = enhancer_outputs
            elif args.enhancer == "URIE":
                enhanced_images, diff = enhancer_outputs
            else:
                enhanced_images = enhancer_outputs
            outputs = classification_model(enhanced_images)

        _, preds = torch.max(outputs, 1)
        hit = (preds == labels).sum().item()
        acc += hit
    clean_acc = acc / len(clean_val_loader.dataset)
    print(f"clean_acc = {clean_acc}")

    # val for corruption
    enhancement_model.eval()
    acc = 0
    for images, labels in tqdm(corruption_val_loader):
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            enhancer_outputs = enhancement_model(images)
            if args.enhancer == "DynTTA":
                enhanced_images, magnitudes, weights = enhancer_outputs
            elif args.enhancer == "URIE":
                enhanced_images, diff = enhancer_outputs
            else:
                enhanced_images = enhancer_outputs
            outputs = classification_model(enhanced_images)

        _, preds = torch.max(outputs, 1)
        hit = (preds == labels).sum().item()
        acc += hit
    corruption_acc = acc / len(corruption_val_loader.dataset)
    print(f"corruption_acc = {corruption_acc}")

if __name__ == '__main__':
    main()