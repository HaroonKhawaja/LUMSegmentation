import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

from PIL import Image
import numpy as np 
import cv2
import os
import json
import random

def load_annotations(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def visualize_annotations(image_path, label_path, image_id, annotations):
    # loading the image
    image = Image.open(image_path)
    image = np.array(image, dtype=np.uint8)

    # loading the label
    label = Image.open(label_path)
    label = np.array(label, dtype=np.uint8)

    # creating plots
    num_plots = 4
    fig, ax = plt.subplots(1, num_plots, figsize=(10, 5))
    ax[0].imshow(image)
    ax[1].imshow(label)
    ax[2].imshow(image)
    ax[3].imshow(image)
    
    # getting the image's annotation
    image_annotations = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image_id]

    # plotting the boundign boxes
    for annotation in image_annotations:
        bbox = annotation['bbox']
        rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]), (bbox[3]), linewidth=1, edgecolor='r', facecolor='none')
        ax[2].add_patch(rect)

        segmentation = annotation['segmentation']
        segmentation = np.array(segmentation).reshape((-1, 2))

        color = [random.random() for _ in range(3)]
        polygon = Polygon(segmentation, linewidth=1, facecolor='white', edgecolor='black')
        ax[3].add_patch(polygon)
            
    for i in range(num_plots):
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

def create_annotations(data_path, output_path):
    annotations = {'images': [], 'annotations': [], 'categories': [{'id':1, 'name': 'plant'}]}
    
    rgb_filenames = [file for file in os.listdir(data_path) if file.endswith("cam2.jpg")]
    label_filenames = [file for file in os.listdir(data_path) if file.endswith("cam2_mask.jpg")]
    
    image_id = 0
    annotations_id = 0
    for rgb_file, label_file in zip(rgb_filenames, label_filenames):
        # saving image data
        annotations['images'].append({
            'id': image_id,
            'file_name': rgb_file,
            'label_file': label_file,
        })

        # loading label images
        label_path = os.path.join(data_path, label_file)
        label_image = np.array(Image.open(label_path).convert("L"))

        # getting all objects in label
        objects = np.unique(label_image)
        objects = objects[1:]   # removing the background

        for obj in objects:
            mask = (label_image == obj).astype('uint8')
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 150:
                    continue
                
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approximation = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approximation) < 3:
                    continue
                
                segmentation = approximation.flatten().tolist()
                x, y, w, h = cv2.boundingRect(contour)
                if w <= 0. or h <= 0.:
                    continue
                    
                bbox = [int(x), int(y), int(w), int(h)]
                area = float(cv2.contourArea(contour))
                annotation = {
                    'id': annotations_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'segmentation': segmentation,
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0,
                }
                annotations["annotations"].append(annotation)
                annotations_id += 1
        image_id += 1

    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=4)