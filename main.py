# This script is used to predict the bounding boxes of the cells and the intensity of the cells in each round.
from SAM_Prediction import main

'''
Batch Processing for the Model

* File Arrangement:
N rounds and pictures have been cropped to make sure they have been aligned.
Cropping is done by imageJ (alignment) and python (batch processing).

Main folders frame_1, frame_2, frame_3,... each representing a frame, 
in each of them there are subfolders R1, R2, R3, R4,... RN, representing each round of the imaging,
the image from R1 is used to predict the bounding boxes of the cells from YOLO model,
also a subfolder named channels, which is a folder containing the overlayed picture of each channel along z-axis, which is the input of SAM model.
and each subfolder contains pictures of different z-heights.

Example: 
Group_path/frame_1/R1/1.png; Group_path/frame_1/channels/R1ch0.png


* Model:
Yolo model is trained to detect the cells and prodice bounding boxes for each cell.
The model is loaded in the package Yolo_Prediction.py and used in the SAM_per_frame function.
Fine-tuning the model is done by another script.

SAM model is from public source and used in the SAM_per_frame function.

* Output:
The output of YOLO prediction is saved in the results folder, with the same structure as the input folder.

The intensity of the cells are given by main function below, which returns a list of cells, each cell is a list of intensity values for each round.

The contour of the cells are also saved (this function is still being tested).
'''


#Group_path is the main folder containing all the frames
Group_path = f'/home/wl/4ipipeline/PIPLINE/4I_Histone/Test_Stitched'
Result_path = f'/home/wl/4ipipeline/PIPLINE/4I_Histone/results/Test'
get_countour = True



if __name__ == "__main__":
    main(get_countour, Group_path, Result_path)
    print('Done!')