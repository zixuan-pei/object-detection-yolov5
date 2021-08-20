# Introduction
This project is based on [yolov5](https://github.com/ultralytics/yolov5). 

It provides a GUI to make it much easier to use. 

You can upload images, videos and folder with only images and videos inside. You can also choose any directory you like to save the output files.

Here is a demo video:

<img src="https://github.com/zixuan-pei/object-detection-yolov5/blob/master/demo.jpg" alt="alt" title="title">

# Quick Start
### Requirement
[Python>=3.6.0](https://www.python.org/) and pip is required with all [requirements.txt](https://github.com/zixuan-pei/object-detection-yolov5/blob/master/requirements.txt) installed including PyTorch>=1.7.

If you want to run the program on GPU, make sure your torch supports CUDA. You can download [here](https://pytorch.org/get-started/locally/#no-cuda-1) or use the command below. This step should be done before installation.
```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Install
```
git clone https://github.com/zixuan-pei/object-detection-yolov5
cd object-detection-yolov5
pip3 install -r requirements.txt
```
### Run
```
python3 Object_Detection.py
```
