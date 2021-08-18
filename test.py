import torch, os

# # Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images

# Images from Web
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# # Images from Folder
# img = []
# folder = 'TestImages'
# for filename in os.listdir(folder):
#     img.append(os.path.join(folder,filename))

# # Single image
# img = 'TestImages/some_image.jpg'


# # Inference
# results = model(img)

# # Results
# results.save()  # or .show(), .save(), .crop(), .pandas(), etc.

# MP4 Video, do not need to load the model
video = 'TestImages/broke_4_8.mp4'
os.system('python3 detect.py --source TestImages/broke_4_8.mp4')
# # Use python3 detect.py --source TestImages/daacb31b4940c3db99016ca68a0cfff1.mp4 in ternimal




