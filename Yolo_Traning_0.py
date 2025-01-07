import os
from roboflow import Roboflow
from ultralytics import YOLO
from IPython.display import display, Image
import glob

# 检查 GPU
os.system('nvidia-smi')

# 获取当前工作目录
HOME = os.getcwd()
print(HOME)

# 安装 YOLOv8
os.system('pip install ultralytics==8.0.196')
import ultralytics
ultralytics.checks()

# 创建目录并下载数据集
os.makedirs('/home/wl/4ipipeline/PIPLINE/MODEL_0402/dataset', exist_ok=True)
os.chdir('/home/wl/4ipipeline/PIPLINE/MODEL_0402/dataset')

# 登录roboflow并记载数据集
rf = Roboflow(api_key="pYeHK8W1XUeIDMMFs57U")
project = rf.workspace("longwu-mzvlg").project("a549-gemia")
dataset = project.version(15).download("yolov8")

# 自定义训练
os.chdir('/home/wl/4ipipeline/PIPLINE/MODEL_0402')
os.system(f'yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml epochs=200 imgsz=640 plots=True')

# 列出训练结果
os.system('ls /home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/')

# 显示混淆矩阵和结果图像
os.chdir('/home/wl/4ipipeline/PIPLINE/MODEL_0402')
display(Image(filename=f'/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/confusion_matrix.png', width=600))
display(Image(filename=f'/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/results.png', width=600))
display(Image(filename=f'/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/val_batch0_pred.jpg', width=1600))

# 验证自定义模型
os.chdir('/home/wl/4ipipeline/PIPLINE/MODEL_0402')
os.system(f'yolo task=detect mode=val model=/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/weights/best.pt data={dataset.location}/data.yaml')

# 使用自定义模型进行推理
os.chdir('/home/wl/4ipipeline/PIPLINE/MODEL_0402')
os.system(f'yolo task=detect mode=predict model=/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True')

# 显示推理结果
for image_path in glob.glob(f'/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/predict/*.jpg')[:3]:
    display(Image(filename=image_path, width=600))
    print("\n")

# 部署模型到 Roboflow
project.version(dataset.version).deploy(model_type="yolov8", model_path=f"/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/")