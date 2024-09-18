import os, re, cv2
from ultralytics import YOLO

# Load the model
model = YOLO('/home/wl/4ipipeline/PIPLINE/MODEL_0402/runs/detect/train2/weights/best.pt')
Group_path = f'/home/wl/4ipipeline/PIPLINE/4I_Formal/WT_Stitched'
Result_path = f'/home/wl/4ipipeline/PIPLINE/4I_Formal/results/WT'
print('Load YOLO model Successfully!')


def round_counters(gpath = Group_path):
    #this function counts the number of rounds of fluroscence images
    file_path = os.path.join(gpath, f'frame_0')
    R = 0
    for root, dirs, files in os.walk(file_path):
        for dir in dirs:
            if re.match(r'^R\d+$', dir):
                R += 1
    return R



def yolo_prediction(z_number, frame, gpath = Group_path):
    # this functions choose pictures from all four rounds that share the same z-height
    # then predict the bounding boxes based on the first round using yolo model
    image = os.path.join(gpath, f'frame_{frame}', 'R1', f'{z_number}.png')
    predict = model.predict(image, conf = 0.5, iou = 0.3, imgsz = 640, verbose = False)
    boxes = predict[0].boxes
    return boxes



def yolo_prediction_per_frame(frame = int, gpath = Group_path, rpath = Result_path):
    # this function predicts the bounding boxes of the cells in each round of the imaging and save the preview in the results folder.
    max_detections = 0
    max_detections_boxes = None
    file_path = os.path.join(gpath, f'frame_{frame}/R1')
    z_max = len(os.listdir(file_path))
    R = round_counters(gpath)
    
    for z in range(z_max):
        boxes = yolo_prediction(z, frame, gpath)
        if len(boxes.xyxy) > max_detections:
            max_detections = len(boxes.xyxy)
            max_detections_boxes = boxes
            max_detection_z = z   
            
    filtered_detections_boxes = []      
    for r in range(1, R+1):
        img = cv2.imread(os.path.join(file_path, f'{max_detection_z}.png'))
        for box in max_detections_boxes:
            tensor_boxes = box.xyxy.clone().detach()
            tensor_boxes_cpu = tensor_boxes.cpu()
            box1 = tensor_boxes_cpu.numpy()
            if r == R:
                filtered_detections_boxes.append(box1)
            x1, y1, x2, y2 = int(box1[0][0]), int(box1[0][1]), int(box1[0][2]), int(box1[0][3])
            color = (0, 255, 0); thickness = 2
            img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.imwrite(os.path.join(rpath, f'f{frame}R{r}_yolo.png'), img)           
   
    return filtered_detections_boxes
