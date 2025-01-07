# Use imagej aligned resulted as input, make sure images are aligned so that cells from different rounds can overlap 
# Using different stitching files for each frame;
# Only after stitching the images, we can use the SAM_Prediction.py and Yolo_Prediction.py to predict the cells in each round.

# Before using this program, the file arrangement should be like this:
# Origin_path/frame_1/R1/1.png; Origin_path/frame_1/R1/xxxch0.png; the former is the channel-merged image, the latter is the single channel image.
# The imagej alignment file of each frame should be under:
# Origin_path/frame_x, named TileConfiguration.registered.txt
# Then the output will be saved like: Group_path/frame_1/R1/1.png; Group_path/frame_1/channels/R1ch0.png

# The output will be saved in the Group_path folder, and can be used in the SAM_Prediction.py and Yolo_Prediction.py


import os, re, ast, cv2
import numpy as np
from tqdm import tqdm

# This part should be modified accordinglly, based on the file arrangement listed above.
celltype = 'Test'
Group_path = f'/home/wl/4ipipeline/PIPLINE/4I_Histone/{celltype}_Stitched' #path to the aligned images
Origin_path = f'/home/wl/4ipipeline/PIPLINE/4I_Histone/{celltype}' #path to the original images

def coordination(dir, Round):
    with open(f'{dir}/TileConfiguration.registered.txt', 'r') as f:
        lines = f.readlines()

    pattern = r'\(-?\d+\.\d+, -?\d+\.\d+\)'
    coordination = []
    for line in lines:
        elements = re.findall(pattern, line)
        for element in elements:
            #convert to tuple
            tup=ast.literal_eval(element)
            coordination.append(tup)
    
    width, height = 2048, 2048
    
    x, y = coordination[Round]
    relative_coor = [(i[0]-x, i[1]-y) for i in coordination]
    nx, ny = map(round,[max([i[0] for i in relative_coor]), max([i[1] for i in relative_coor])])
    max_x, max_y = map(round, [width + min([i[0] for i in relative_coor]), height + min([i[1] for i in relative_coor])])
    
    return nx, ny, max_x, max_y

def count_channels_in_folder(folder_path):
    channels = set()
    channel_pattern = re.compile(r'ch(\d+)')
    
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        # Search for channel pattern in the filename
        match = channel_pattern.search(filename)
        if match:
            channels.add(match.group(0))
            
    return len(channels)



def stitch_files(frame, rounds = 3, PCNA = False):
    for Round in range(rounds):
        nx, ny, max_x, max_y = coordination(f'{Origin_path}/frame_{frame}/', Round)
        
        file_dir = f'{Origin_path}/frame_{frame}/R{Round+1}'
        images = os.listdir(file_dir)
        images.sort()
        merged = [img for img in images if 'ch' not in img]
        
        num_channels = count_channels_in_folder(file_dir)
        channels = [[] for _ in range(num_channels)]
        pattern = r'^(\d+)'
        pattern1 = r'[A-Za-z]+(\d+)ch(\d+)'
        merged = sorted(merged, key=lambda x: int(re.search(pattern, x).group(0)))
        
        for img in images:
            match = re.match(pattern1, img)
            if match:
                channel_num = int(match.group(2))
                if channel_num < num_channels:
                    channels[channel_num].append(img)
        
        for ch in channels:
            ch.sort(key=lambda x: int(re.match(pattern1, x).group(1)))
        
        out_path = f'{Group_path}'
        ''''
        outs = [f'{out_path}/frame_{frame}/channels/z{i}' for i in range(len(merged))]
        outs.append(f'{out_path}/frame_{frame}/R{Round+1}')
        for out in outs:  
            if not os.path.exists(out):
                os.makedirs(out)
        '''
        
        img_channels = [[] for _ in range(num_channels)]
        
        for i, image in enumerate(merged):
            img = cv2.imread(file_dir+'/'+image)
            cropped = img[ny:max_y, nx:max_x]
            cv2.imwrite(f'{out_path}/frame_{frame}/R{Round+1}/{i}.png', cropped)
        
        for ch_idx, ch in enumerate(channels):
            for i, image in enumerate(ch):
                img = cv2.imread(file_dir+'/'+image)
                cropped = img[ny:max_y, nx:max_x]
                img_channels[ch_idx].append(cropped)
        
        for ch_idx, img_ch in enumerate(img_channels):
            ch_stack = np.array(img_ch)
            ch_max = np.max(ch_stack, axis=0)
            cv2.imwrite(f'{out_path}/frame_{frame}/channels/R{Round+1}ch{ch_idx}.png', ch_max)

            # Save the PCNA channel for Cell Cycle Analysis
            if PCNA:
                if Round == 3 and ch_idx == 2:
                    cv2.imwrite(f'{out_path}/frame_{frame}/channels/PCNA.png', img_ch[-2])
            


num_frames = len(os.listdir(f'{Origin_path}'))
for frame in tqdm(range(num_frames)):
    stitch_files(frame, 4, True)