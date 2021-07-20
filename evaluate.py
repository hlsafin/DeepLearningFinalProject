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
import cv2
from gluoncv.utils.viz import get_color_pallete

def iou(labels, predicted_labels):
    _, tag = torch.max(predicted_labels, dim=1)
    intersection = np.logical_and(labels, tag)
    union = np.logical_or(labels, tag)
    #return np.sum(intersection) / np.sum(union)
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

def image_show(image, target, pred_mask):
    image = image[0].detach().cpu().numpy().transpose((1, 2, 0))
    target = target[0].detach().cpu().numpy()
    pred_mask = pred_mask[0].detach().cpu().numpy()
    target_mask = get_color_pallete(target, dataset='pascal_voc')
    pred_mask = get_color_pallete(pred_mask, dataset='pascal_voc')
    mean = np.array([.485, .456, .406])
    std = np.array([.229, .224, .225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # show images
    plt.subplot(1,4,1)
    #plt.plot(image)
    plt.subplot(1,4,2)
    plt.imshow(target_mask)
    plt.subplot(1,4,3)
    plt.imshow(pred_mask)
    plt.pause(100)  # pause a bit so that plots are updated

def main():
    print('UNet training initiated ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('your computer is using: {}'.format(device))

    # set batch_size = 2 for toy example
    batch_size = 1
    epochs = 1

    # load model
    model = UNet().to(device=device)
    model.load_state_dict((torch.load('best_model.pth')))
    model.eval
    
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

    testset = torchvision.datasets.VOCSegmentation(
        root='unet/data',
        year='2012',
        image_set='val',
        download=False,
        transform=image_transform,
        target_transform=target_transform
    )
    print('Testing images: {}'.format(len(testset)))
    
    # Reduce a number of size
    indices = torch.arange(1)
    testset = data_utils.Subset(testset, indices)

    # Create Training Loader
    test_data = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    # initialize optimizer and criterion for multi-class segmentation
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # training code
    for epoch in range(epochs):
        t1 = time.time() # record starting time
        epoch_loss = 0
        iou_vals = []
        precision_vals = []
        # for training
        for idx, (images, target) in enumerate(test_data):
            with torch.no_grad():
            # convert to pytorch tensor
                images = Variable(images).to(device=device, dtype=torch.float32)
                target = Variable(target).to(device=device, dtype=torch.long).squeeze(1)
                #image_show(images, target)
                pred = model(images)
                pred, indices = torch.max(pred, dim=1)
                image_show(images, target, pred)

                train_loss = criterion(pred, target)
                epoch_loss += train_loss.item()

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                curr_iou = iou(target.cpu(), pred.cpu())
                #curr_precision = precision(curr_iou)
                iou_vals.append(curr_iou)
                #precision_vals.append(curr_precision)
                print("idx {} loss: {} iou: {}".format(idx, train_loss, curr_iou))

if __name__ == "__main__":
    main()
