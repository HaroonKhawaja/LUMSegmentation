import yaml
import requests
import cv2
import numpy as np
import ndjson
import os
import tqdm

def logits2rgb(img):
    # Specify custom colours
    colors = [
        [0, 0, 0],        # black for background
        [187, 207, 74],    # green for plants
        [0, 108, 132],     # blue
        [255, 204, 184],   # yellow
        [232, 167, 53],    # orange
    ]
    col = np.zeros((img.shape[0], img.shape[1], 3))
    unique = np.unique(img)
    
    for i, val in enumerate(unique):
        mask = np.where(img == val)
        col[mask] = colors[i % len(colors)]
        
    return col.astype(int)

def draw_bbox(image, bounding_box):
    top = int(bounding_box['top'])
    left = int(bounding_box['left'])
    height = int(bounding_box['height'])
    width = int(bounding_box['width'])

    top_left = (left, top)
    bottom_right = (left + width, top + height)
    
    color = (255, 0, 0)
    thickness = 2 
    return cv2.rectangle(image, top_left, bottom_right, color, thickness)
    
def get_mask(PROJECT_ID, api_key, colour, class_indices, dest_path):
    # Open export json. Change name if required
    with open('instance_segmentation_data.ndjson') as f:
        data = ndjson.load(f)
        
        # create mask directory if it doesnt exist
        if not os.path.isdir(dest_path):
            os.mkdir(dest_path)
        
        # Iterate over all images
        for i, d in enumerate(data):
            files_in_folder = os.listdir(dest_path)
            image_name = data[i]['data_row']['external_id']
            label_name = image_name.replace(".jpg", "") + '_mask.jpg'
            bbox_img_name = image_name.replace(".jpg", "") + '_bbox.jpg'

            # if (label_name not in files_in_folder) and (bbox_img_name not in files_in_folder):
            if  (image_name not in files_in_folder) and (label_name not in files_in_folder) and (bbox_img_name not in files_in_folder):
                mask_full = np.zeros((data[i]['media_attributes']['height'], data[i]['media_attributes']['width']))

                img_url = data[i]['data_row']['row_data']
                img_response = requests.get(img_url, stream=True)
                img_array = np.asarray(bytearray(img_response.raw.read()), dtype="uint8")
                original_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Iterate over all masks
                for idx, obj in enumerate(data[i]['projects'][PROJECT_ID]['labels'][0]['annotations']['objects']):
                    # Extract mask name and mask url
                    name = obj['name']
                    if "bounding_box" in obj:
                        bbox = obj["bounding_box"]
                        bbox_image = draw_bbox(original_img.copy(), bbox)
                    
                    if "mask" in obj:
                        url = obj['mask']['url']
                    else:
                        continue

                    # Download mask
                    headers = {'Authorization': api_key}
                    with requests.get(url, headers=headers, stream=True) as r:
                        r.raw.decode_content = True
                        mask = r.raw
                        image = np.asarray(bytearray(mask.read()), dtype="uint8")
                        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
                    
                    # Assign mask index to image-mask 
                    cl = idx + 1
                    mask_full[np.where(image == 255)] = cl

                unique = np.unique(mask_full)
                print(f"Detected {len(unique) - 1} instances (excluding background) in {image_name}")

                if len(unique) > 1:
                    if colour:
                        mask_full_colour = logits2rgb(mask_full)
                        mask_full_colour = cv2.cvtColor(mask_full_colour.astype('float32'), cv2.COLOR_RGB2BGR)
                    # Save image mask
                    cv2.imwrite(os.path.join(dest_path, image_name), original_img)
                    cv2.imwrite(os.path.join(dest_path, label_name), mask_full_colour)
                    cv2.imwrite(os.path.join(dest_path, bbox_img_name), bbox_image)
            else:
                print('Files already processed!')

if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    project_id = config['project_id']
    api_key = config['api_key']
    colour = True

    dest_path = './Data/lums_instance_segmentation_data'

    # Specify your custom class indices
    class_indices = {
        "Plant" : 1,
        "background": 0
    }
    
    get_mask(project_id, api_key, colour, class_indices, dest_path)