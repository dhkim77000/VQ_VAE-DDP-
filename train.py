import torchvision
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler
import torch.nn as nn
import torchvision.utils as utils
#from ultralytics import YOLO
from keras.applications.vgg16 import preprocess_input 
from PIL import Image, ImageDraw, ImageFile
from collections import Counter
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from random import randint
import pandas as pd
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse
from datetime import datetime
import os
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import repeat
from PIL import Image, ImageEnhance
from tqdm import tqdm
import pdb
from skimage.feature import hog
from skimage import data, exposure
import cv2
import glob
import matplotlib as mpl
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from PIL import ImageFilter
from PIL import Image
from warnings import simplefilter
from model import VQ_CVAE
import torch.distributed as dist
import torch.multiprocessing as mp
import glob
import gc
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import logging

class CustomDataset(Dataset):
    def __init__(self,root):
        self.imgs = glob.glob(f'{root}/*.jpg')
        self.transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, img_path


def write_images(data, outputs, writer, suffix):
    original = data.mul(0.5).add(0.5)
    original_grid = make_grid(original[:6])
    writer.add_image(f'original/{suffix}', original_grid)
    reconstructed = outputs[0].mul(0.5).add(0.5)
    reconstructed_grid = make_grid(reconstructed[:6])
    writer.add_image(f'reconstructed/{suffix}', reconstructed_grid)


def save_reconstructed_images(data, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)

def save_checkpoint(model, epoch, save_path):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    checkpoint_path = os.path.join(save_path, 'checkpoints', f'model_{epoch}.pth')
    torch.save(model.state_dict(), checkpoint_path)

def train(epoch, model, device,  train_loader, optimizer, log_interval, save_path, writer):
    model.train()
    loss_dict = model.module.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, images, = None, None
    for batch_idx, (images, _ ) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = model.module.loss_function(images, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.module.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(images), total_batch=len(train_loader) * len(images),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            # logging.info('z_e norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[1][0].contiguous().view(256,-1),2,0)))))
            # logging.info('z_q norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[2][0].contiguous().view(256,-1),2,0)))))
            for key in latest_losses:
                losses[key + '_train'] = 0

        #if batch_idx == (len(train_loader) - 1):
        #    save_reconstructed_images(images, epoch, outputs[0], save_path, 'reconstruction_train')
        #    write_images(images, outputs, writer, 'train')
        if  batch_idx * len(images) > 50000:
            break

        del images, outputs, loss, latest_losses
        gc.collect()
        torch.cuda.empty_cache()

    for key in epoch_losses:
        epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))


    return epoch_losses

def test(epoch, model, device, test_loader, save_path, writer):
    model.eval()
    loss_dict = model.module.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, images = None, None
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            model.module.loss_function(images, *outputs)
            latest_losses = model.module.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                #write_images(images, outputs, writer, 'test')
                #save_reconstructed_images(images, epoch, outputs[0], save_path, 'reconstruction_test')
                save_checkpoint(model, epoch, save_path)

            if  i * len(images) > 1000: break
            
            del images, outputs, loss
            gc.collect()
            torch.cuda.empty_cache()

    for key in losses:
        losses[key] /= (i * len(images))

    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))

    return losses


def run(rank, world_size):
    batch_size = 4
    NUM_EPOCH = 10000
    learning_rate = 2e-4
    k = 512
    hidden = 128
    device = torch.device('cuda', rank)
    max_memory = torch.cuda.get_device_properties(device).total_memory
    memory_limit = int(max_memory * 0.8)
    torch.cuda.set_per_process_memory_fraction(0.8, device)
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method= "tcp://127.0.0.1:2568", world_size=world_size, rank=rank)

    # Load the dataset and create a data loader

    dataset = CustomDataset(root="/home/dhkim/cafe/flickr30k_images/flickr30k_images")
    print(len(dataset))
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers = 4, pin_memory=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, num_workers = 4, pin_memory=True)

    dataloaders = {
        "train" : train_loader,
        "test" : test_loader}

    # Define the model and wrap it with DistributedDataParallel
    model = VQ_CVAE(128, k = k, num_channels = 3)
    model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10)

    save_path = '/home/dhkim/cafe/model'
    writer = SummaryWriter(save_path)

    for epoch in range(1, NUM_EPOCH+1):
        train_losses = train(epoch, model, device, dataloaders['train'], optimizer, 10, save_path, writer)
        test_losses = test_net(epoch, model, device, dataloaders['test'],  save_path, writer)

        for k in train_losses.keys():
            name = k.replace('_train', '')
            train_name = k
            test_name = k.replace('train', 'test')
            writer.add_scalars(name, {'train': train_losses[train_name],
                                      'test': test_losses[test_name],
                                      })
        scheduler.step()
   
    dist.destroy_process_group()

def main():
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
