import numpy as np
import random
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import torch
from unet.unet import UNet
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import os
"""
Before you run this code, please install the following package:
pip install tensorflow
pip install tensorboard
"""
def iou(labels, predicted_labels):
    _, tag = torch.max(predicted_labels, dim=1)
    intersection = torch.logical_and(labels, tag)
    union = torch.logical_or(labels, tag)
    # return np.sum(intersection) / np.sum(union)
    return torch.sum(intersection) / torch.sum(union)

def precision(iou):
    # Calculate average precision using a sliding threshold and average the results
    precision = []
    for t in range(50, 95, 5):
        # Divide by 100 to get percentage threshold
        threshold = t / 100
        true_positive = []
        false_positive = []
        for i in range(len(iou)):
            if iou[i] >= threshold:
                true_positive.append(iou[i])
            else:
                false_positive.append(iou[i])
        precision.append(len(true_positive) / len(true_positive) + len(false_positive))
    # Return average precision
    return sum(precision) / len(precision)

def validation(epoch, val_loader, model, criterion, device):
    iou_vals = []
    precision_vals = []
    valid_loss = 0

    for idx, (images, target) in enumerate(val_loader):
        images = Variable(images).to(device=device, dtype=torch.float32)
        target = Variable(target).to(device=device, dtype=torch.long).squeeze(1)
        
        with torch.no_grad():
            pred = model(images)
            loss = criterion(pred, target)
            valid_loss += loss.item()

            curr_iou = iou(target.cpu(), pred.cpu())
            #curr_precision = precision(curr_iou)
            iou_vals.append(curr_iou)
            #precision_vals.append(curr_precision)
    
    average_iou = sum(iou_vals)/len(iou_vals)
    val_loss = valid_loss/len(val_loader)
    return average_iou, valid_loss


def visualize_masks(model, data):
    for idx, (images, target) in enumerate(data):
        with torch.no_grad():
            pil_transform = transforms.ToPILImage()
            plt.imsave(f'image-{idx}.png', pil_transform(images[0]))
            plt.imsave(f'target-{idx}.png', pil_transform(target[0]))
            mask = model(images)
            # Combine masks to get image mask
            comb_mask, _ = torch.max(mask, dim=1)
            # Visualize mask
            a_mask = comb_mask[0]
            pil_mask = transforms.ToPILImage()(a_mask)
            plt.imsave(f'mask-{idx}.png', pil_mask)
            if idx == 3:
                break


def main():
    print('UNet training initiated ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('your computer is using: {}'.format(device))

    # set batch_size = 2 for toy example
    batch_size = 1
    epochs = 100

    # load model
    model = UNet().to(device=device)
    
    # dataloader
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        transforms.Resize((572, 572))
    ])
    target_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((388, 388))
    ])

    trainset = torchvision.datasets.VOCSegmentation(
        root='unet/data/',
        year='2012',
        image_set='train',
        download=False,
        transform=image_transform,
        target_transform=target_transform
    )
    testset = torchvision.datasets.VOCSegmentation(
        root='unet/data',
        year='2012',
        image_set='val',
        download=False,
        transform=image_transform,
        target_transform=target_transform
    )
    print('Training images: {} Testing images: {}'.format(len(trainset), len(testset)))
    
    # Reduce a number of size
    indices = torch.arange(50)
    trainset = data_utils.Subset(trainset, indices)
    testset = data_utils.Subset(testset, indices)

    # Create Training Loader
    train_data = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_data = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # initialize optimizer and criterion for multi-class segmentation
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # tensorboard
    tb = SummaryWriter()

    # Visualize mask
    if os.path.exists('model/best_model.pth'):
        model.load_state_dict(torch.load('model/best_model.pth'))
    visualize_masks(model, test_data)
    """
    # training code
    for epoch in range(epochs):
        t1 = time.time() # record starting time
        model.train()
        epoch_loss = 0
        iou_vals = []
        precision_vals = []
        # for training
        for idx, (images, target) in enumerate(train_data):
            # convert to pytorch tensor
            images = Variable(images).to(device=device, dtype=torch.float32)
            target = Variable(target).to(device=device, dtype=torch.long).squeeze(1)
            
            pred = model(images)
            pred_combine = torch.max(pred, dim=1)[:, None, :, :]
            train_loss = criterion(pred, target)
            epoch_loss += train_loss.item()

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            curr_iou = iou(target.cpu(), pred.cpu())
            
            # curr_precision = precision(curr_iou)
            iou_vals.append(curr_iou)
            print("idx {} loss: {} iou: {}".format(idx, loss, curr_iou))
        avg_precision = precision(iou_vals)
        tb.add_scalar("train loss", epoch_loss/len(train_data), epoch)
        tb.add_scalar("IoU", sum(iou_vals)/len(iou_vals), epoch)
        tb.add_scalar("precision:", avg_precision, epoch)

        t2 = time.time() # record time
        print("epoch: {} training loss: {} IoU: {} time: {}".format(epoch, epoch_loss/len(train_data), sum(iou_vals)/len(iou_vals), t2-t1))
        
        # for evaluation
        average_iou, val_loss = validation(epoch, test_data, model, criterion, device)
        tb.add_scalar("val loss", val_loss, epoch)
        print(f"validation iou epoch {epoch}: {average_iou}")
    
    # save model
    torch.save(model.state_dict(), 'best_model.pth')
    tb.close()
    """

if __name__ == "__main__":
    main()
