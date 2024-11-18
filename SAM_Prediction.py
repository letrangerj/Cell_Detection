import os, torch, cv2, csv
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import tqdm, trange
from itertools import product
from Yolo_Prediction import yolo_prediction_per_frame


Group_path = f'/home/wl/4ipipeline/PIPLINE/4I_Histone/Test_Stitched'
Result_path = f'/home/wl/4ipipeline/PIPLINE/4I_Histone/results/Test'

HOME = '/home/wl/4ipipeline/PIPLINE/pipeline'
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print('CHECKPOINT_PATH', "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_predictor = SamPredictor(sam)
print('Load SAM model Successfully!')


def cal_background(image_bgr):
    # This function is used to calculate the background intensity using the minimunm intensity of the center region of the image.
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) 
    height, width = image_gray.shape
    bbox_width = int(width / 2)
    bbox_height = int(height / 2)
    center_x = int(width / 2)
    center_y = int(height / 2)
    bounding_box = (center_x - bbox_width // 2, center_y - bbox_height // 2, bbox_width, bbox_height)
    
    roi = image_gray[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]
    background = np.min(roi)
    return background



def getting_new_mask(image, masks, background):   
    #this function is used to filter out the pixels that are below 20% of the maximum intensity of the masked region.
    #to make sure the mask of the cell does not include background pixels.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    new_masks = []
    for mask in masks:
        masked_pixels = image_gray[mask == 1]
        max_pixel = max(masked_pixels)
        filtered_pixels = []
        for pixel in masked_pixels:
            if pixel > (max_pixel - background) * 0.2:
                filtered_pixels.append(pixel)
            else:
                filtered_pixels.append(0)
        new_mask = np.zeros_like(mask)
        new_mask[mask == 1] = filtered_pixels
        new_masks.append(new_mask)
    return np.array(new_masks)



def calculate_average_intensity(image, mask):
    #caluculate the average intensity of the masked region
    masked_pixels = image[mask == 1]
    #average_intensity = np.mean(masked_pixels)
    average_intensity = np.sum(masked_pixels)
    return average_intensity



def get_countour(mask):
    #get the countour of the mask
    mask_opencv = np.uint8(mask * 255)
    contours, _ = cv2.findContours(mask_opencv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def calculate_intensity_distribution(image, mask):    
    # Used for analyse the intensity distribution of the masked region for each cell
    # Function included in 11/19/2024 version
    
    # Get the pixels that are within the mask
    masked_pixels = image[mask == 1]
    # Calculate the unique intensities and their counts
    intensities, counts = np.unique(masked_pixels, return_counts=True)
    # Create a dictionary that maps each intensity to its count
    intensity_distribution = dict(zip(intensities, counts))

    return intensity_distribution



def SAM_per_frame(n = int, get_countour = False, gpath = Group_path, rpath = Result_path):
    #this function is used to predict the intensity of the cells in each round.
    cells = []
    countours = []
    distribution = []
    filtered_boxes = yolo_prediction_per_frame(n, gpath, rpath)
    file_path = os.path.join(gpath, f'frame_{n}')
    
    files = os.listdir(os.path.join(file_path, 'channels'))
    files = [file for file in files if file.endswith('.png')]
    files.sort()
    
    for box in tqdm(filtered_boxes, desc = 'Processing Cells', leave = False): #each box is a cell
        cell = [((box[0][0]+box[0][2])/2, (box[0][1]+box[0][3])/2)] #center position of the cell as the first element
        countour =  [((box[0][0]+box[0][2])/2, (box[0][1]+box[0][3])/2)]
        distribution = [((box[0][0]+box[0][2])/2, (box[0][1]+box[0][3])/2)]
        for file in files:
            IMAGE_PATH = os.path.join(file_path, 'channels', file)
            image_bgr = cv2.imread(IMAGE_PATH)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            mask_predictor.set_image(image_rgb)
            masks, scores, logits = mask_predictor.predict(
                box=box,
                multimask_output=True
            )
            background = cal_background(image_bgr)
            new_masks = getting_new_mask(image_bgr, masks, background)
            
            # get average intensity
            average_intensity = []
            for i in range(new_masks.shape[0]):
                average_intensity.append(calculate_average_intensity(image_bgr, new_masks[i]))
            cell.append(sum(average_intensity)/3 - background)
            
            # get contour
            if file.endswith('R1ch0.png'): #only get the countour of the first channel DAPI
                if get_countour:
                    mask = masks[0]
                    mask_opencv = np.uint8(mask * 255)
                    contours, _ = cv2.findContours(mask_opencv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    countour.append(contours[0])
                    
            # get intensity distribution
            intensity_distribution = calculate_intensity_distribution(image_bgr, new_masks[0])
                
        cell.append(f'frame{n}')  # Use for clonal analysis
        cells.append(cell)
        countours.append(countour)
        distribution.append(intensity_distribution)
        
    return cells, countours, distribution



def main(get_contour = True, get_distribution = True, gpath = Group_path, rpath = Result_path):
    if not os.path.exists(rpath):
        os.makedirs(rpath)
    
    num_frame = len(os.listdir(gpath))
    all_cells = []
    all_contours = []
    all_distribution = []
    for frame in trange(num_frame, desc = 'Processing frames', leave = False):
        cells, countours, distribution = SAM_per_frame(frame, get_contour, gpath, rpath)
        all_cells += cells
        all_contours += countours
        all_distribution += distribution
        
    #save intensity results in csv file
    with open(os.path.join(rpath, f'intensity.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        for cell in all_cells:
            writer.writerow(cell)
                
    #save countour results in csv file
    if get_contour:
        with open(os.path.join(rpath, f'countour.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            for countour in all_contours:
                writer.writerow(countour)
                
    #save intensity distribution results in csv file
    if get_distribution:
        with open(os.path.join(rpath, f'intensity_distribution.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            for distribution in all_distribution:
                writer.writerow(distribution)
    return