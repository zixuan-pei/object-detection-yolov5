# Introduction
This project is based on [yolov5](https://github.com/ultralytics/yolov5). 

It provides a GUI so that you do not need to know yolov5 code to generate detected images and videos.

This program is super easy to use. You can choose images, videos and folder to run. Any directory can be chosen as you like to save the output files. You can directly click the Result button to see or the outputs.

Here is a demo video:

<img src="https://github.com/zixuan-pei/object-detection-yolov5/blob/master/demo.jpg" alt="alt" title="title">

# Quick Start
### Requirement
[Python>=3.6.0](https://www.python.org/) and pip is required with all [requirements.txt](https://github.com/zixuan-pei/object-detection-yolov5/blob/master/requirements.txt) installed including PyTorch>=1.7.

If you want to run the program on GPU, make sure your PyTorch version supports CUDA. You can download [here](https://pytorch.org/get-started/locally/#no-cuda-1) or use the command below. This step should be done before installation.
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
