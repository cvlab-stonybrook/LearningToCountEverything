"""
Demo file for Few Shot Counting

By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Created: 19-Apr-2021
Last modified: 19-Apr-2021
"""

import cv2
from model import CountRegressor, Resnet50FPN
from utils import MAPS, Scales, Transform, extract_features
from utils import visualize_output_and_save, select_exemplar_rois
from PIL import Image
import os
import torch
import argparse


parser = argparse.ArgumentParser(description="Few Shot Counting Demo code")
parser.add_argument("-i", "--input-image", type=str, required=True, help="/Path/to/input/image/file/")
parser.add_argument("-b", "--bbox-file", type=str, help="/Path/to/file/of/bounding/boxes")
parser.add_argument("-o", "--output-dir", type=str, default=".", help="/Path/to/output/image/file")
parser.add_argument("-m",  "--model_path", type=str, default="./data/pretrainedModels/FamNet_Save1.pth", help="path to trained model")
parser.add_argument("-g",  "--gpu-id", type=int, default=0, help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
args = parser.parse_args()

if not torch.cuda.is_available() or args.gpu_id < 0:
    use_gpu = False
    print("===> Using CPU mode.")
else:
    use_gpu = True
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

resnet50_conv = Resnet50FPN()
regressor = CountRegressor(6, pool='mean')

if use_gpu:
    resnet50_conv.cuda()
    regressor.cuda()
    regressor.load_state_dict(torch.load(args.model_path))
else:
    regressor.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))

resnet50_conv.eval()
regressor.eval()

image_name = os.path.basename(args.input_image)
image_name = os.path.splitext(image_name)[0]

if args.bbox_file is None: # if no bounding box file is given, prompt the user for a set of bounding boxes
    out_bbox_file = "{}/{}_box.txt".format(args.output_dir, image_name)
    fout = open(out_bbox_file, "w")

    im = cv2.imread(args.input_image)
    # rects = cv2.selectROIs("Image", im,False,False)
    cv2.imshow('image', im)
    rects = select_exemplar_rois(im)

    rects1 = list()
    for rect in rects:
        x1, y1 = rect[0], rect[1]
        x2 = x1 + rect[2]
        y2 = y1 + rect[3]
        rects1.append([y1, x1, y2, x2])
        fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))

    fout.close()
    cv2.destroyWindow("Image")
    print("selected bounding boxes are saved to {}".format(out_bbox_file))
else:
    with open(args.bbox_file, "r") as fin:
        lines = fin.readlines()

    rects1 = list()
    for line in lines:
        data = line.split()
        y1 = int(data[0])
        x1 = int(data[1])
        y2 = int(data[2])
        x2 = int(data[3])
        rects1.append([y1, x1, y2, x2])

print("Bounding boxes: ", end="")
print(rects1)

image = Image.open(args.input_image)
image.load()
sample = {'image': image, 'lines_boxes': rects1}
sample = Transform(sample)
image, boxes = sample['image'], sample['boxes']

if use_gpu:
    image = image.cuda()
    boxes = boxes.cuda()

with torch.no_grad():
    features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
    output = regressor(features)

print('===> The predicted count is: {:6.2f}'.format(output.sum().item()))

rslt_file = "{}/{}_out.png".format(args.output_dir, image_name)
visualize_output_and_save(image.detach().cpu(), output.detach().cpu(), boxes.cpu(), rslt_file)
print("===> Visualized output is saved to {}".format(rslt_file))


