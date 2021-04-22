# Learning To Count Everything

This is the official implementation of the following CVPR 2021 paper:

```
Learning To Count Everything
Viresh Ranjan, Udbhav Sharma, Thu Nguyen and Minh Hoai
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
```
Link to arxiv preprint: https://arxiv.org/pdf/2104.08391.pdf

## Dataset download
Validation and test sets can be found here: https://drive.google.com/file/d/1XDPeOOqavF1CTaOe3sLjFj-SByQwhxss/view?usp=sharing

Place the unzipped image directory inside the data directory.

Entire dataset will be released soon.

## Installation with Conda

conda create -n fscount python=3.7 -y

conda activate fscount

python -m pip install matplotlib opencv-python notebook tqdm

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch


## Quick demo

Provide the input image and also provide the bounding boxes of exemplar objects using a text file:

``` bash
python demo.py --input-image orange.jpg --bbox-file orange_box_ex.txt 
```

Use our provided interface to specify the bounding boxes for exemplar objects


``` bash
python demo.py --input-image orange.jpg
```


## Evaluation
We are providing our pretrained FamNet model, and the evaluation code can be used without the training.
### Testing on validation split without adaptation
```bash 
python test.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ --test_split val
```
### Testing on val split adaptation
```bash 
python test.py --data_path /PATH/TO/YOUR/FSC147/DATASET/ --test_split val --adapt
```


## Training 
``` bash
python train.py --gpu 0
```

## Citation

If you find the code useful, please cite:
```
@inproceedings{m_Ranjan-etal-CVPR21,
  author = {Viresh Ranjan and Udbhav Sharma and Thu Nguyen and Minh Hoai},
  title = {Learning To Count Everything},
  year = {2021},
  booktitle = {Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```


