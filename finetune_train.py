"""
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan,Udbhav, Thu Nguyen, Minh Hoai

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""
import torch.nn as nn
from model import  Resnet50FPN,CountRegressor,weights_normal_init
from utils import MAPS, Scales, Transform,TransformTrain,extract_features, visualize_output_and_save
from PIL import Image
import os
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from os.path import exists,join
import random
import torch.optim as optim
import torch.nn.functional as F
import ast


parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='/home/hoai/DataSets/AgnosticCounting/FSC147_384_V2/', help="Path to the FSC147 dataset")
parser.add_argument("-o", "--output_dir", type=str,default="./logsSave", help="/Path/to/output/logs/")
parser.add_argument("-ts", "--test-split", type=str, default='val', choices=["train", "test", "val"], help="what data split to evaluate on on")
parser.add_argument("-ep", "--epochs", type=int,default=1500, help="number of training epochs")
parser.add_argument("-g", "--gpu", type=int,default=0, help="GPU id")
parser.add_argument("-lr", "--learning-rate", type=float,default=1e-5, help="learning rate")
args = parser.parse_args()


data_path = args.data_path

# Loading train dataset
anno_file_train = data_path + '/train_data_annotation_final.txt'
im_dir_train = data_path + '/train_data/images'
gt_dir_train = data_path + '/train_data/ground_truth'

# Loading valid dataset
anno_file_val = data_path + '/val_data_annotation_final.txt'
im_dir_val = data_path + '/val_data/images'
gt_dir_val = data_path + '/val_data/ground_truth'

if not exists(args.output_dir):
    os.mkdir(args.output_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

criterion = nn.MSELoss().cuda()

resnet50_conv = Resnet50FPN()
resnet50_conv.cuda()
resnet50_conv.eval()

regressor = CountRegressor(6, pool='mean')
weights_normal_init(regressor, dev=0.001)
regressor.train()
regressor.cuda()
optimizer = optim.Adam(regressor.parameters(), lr = args.learning_rate)

# Training

annotations_train = {}
with open(anno_file_train) as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if line:
        data = ast.literal_eval(line)
        annotations_train.update(data)

def train():
    print("Training on your custom dataset")
    im_ids = list(annotations_train.keys())
    random.shuffle(im_ids)
    train_mae = 0
    train_rmse = 0
    train_loss = 0
    cnt = 0
    
    for im_id in im_ids:
        cnt += 1
        anno = annotations_train[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = anno['points']

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir_train, im_id))
        image = image.convert("RGB")
        image.load()
        density_path = gt_dir_train + '/' + im_id.split(".jpg")[0] + ".npy"
        density = np.load(density_path).astype('float32')    
        sample = {'image':image,'lines_boxes':rects,'gt_density':density}
        sample = TransformTrain(sample)
        image, boxes,gt_density = sample['image'].cuda(), sample['boxes'].cuda(),sample['gt_density'].cuda()

        with torch.no_grad():
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        features.requires_grad = True
        optimizer.zero_grad()
        output = regressor(features)

        #if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != gt_density.shape[2] or output.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(output.shape[2],output.shape[3]),mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)
        loss = criterion(output, gt_density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        
    train_loss = train_loss / len(im_ids)
    train_mae = (train_mae / len(im_ids))
    train_rmse = (train_rmse / len(im_ids))**0.5
    
    return train_loss,train_mae,train_rmse


# Evaluating
annotations_val = {}
with open(anno_file_val) as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if line:
        data = ast.literal_eval(line)
        annotations_val.update(data)
   
def eval():
    cnt = 0
    SAE = 0 # sum of absolute errors
    SSE = 0 # sum of square errors

    print("Evaluation on your dataset")
    im_ids = list(annotations_val.keys())
    for im_id in im_ids:
        cnt += 1
        anno = annotations_val[im_id]
        bboxes = anno['box_examples_coordinates']
        dots = anno['points']

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(im_dir_val, im_id))
        image = image.convert("RGB")
        image.load()
        sample = {'image':image,'lines_boxes':rects}
        sample = Transform(sample)
        image, boxes = sample['image'].cuda(), sample['boxes'].cuda()

        with torch.no_grad():
            output = regressor(extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales))

        gt_cnt = dots
        pred_cnt = output.sum().item()
        cnt = cnt + 1
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

    print('On {} data, MAE: {:6.2f}, RMSE: {:6.2f}'.format("training", SAE/cnt, (SSE/cnt)**0.5))
    return SAE/cnt, (SSE/cnt)**0.5


best_mae, best_rmse = 1e7, 1e7
stats = list()

checkpoint_file = join(args.output_dir, "checkpoint.pth")
start_epoch = 0  # Default start_epoch when training from scratch
if exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    regressor.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_mae = checkpoint['best_mae']
    best_rmse = checkpoint['best_rmse']
    start_epoch = checkpoint['epoch']
    print(f"Resuming training from epoch {start_epoch}")

for epoch in tqdm(range(start_epoch, args.epochs), desc='Training Epochs'):
    regressor.train()
    train_loss, train_mae, train_rmse = train()
    regressor.eval()
    val_mae, val_rmse = eval()
    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(args.output_dir, "stats.txt")
    with open(stats_file, 'a') as f:  # Append to the file
        f.write(f"Epoch {epoch + 1}, {','.join(map(str, stats[-1]))}\n")

    if best_mae >= val_mae:
        best_mae = val_mae
        best_rmse = val_rmse
        model_name = args.output_dir + '/' + "FamNet.pth"
        torch.save(regressor.state_dict(), model_name)

    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': regressor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mae': best_mae,
        'best_rmse': best_rmse
    }, checkpoint_file)

    print(f"Epoch {epoch + 1}, Avg. Epoch Loss: {stats[-1][0]}, Train MAE: {stats[-1][1]}, Train RMSE: {stats[-1][2]}, Val MAE: {stats[-1][3]}, Val RMSE: {stats[-1][4]}, Best Val MAE: {best_mae}, Best Val RMSE: {best_rmse}")
