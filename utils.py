
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import torch
import random
import shutil
import json



def createIfNotExist(directory):
    """ Create directory if it does not exist

    params:
        directory (string): directory to be created
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def getImagesInFolder(path, extensions = ('.jpg','.png')):
    """ Get a list of all images in a folder, including subfolders

    params:
        path: path to be searched
        extensions: extensions to be searched for
    
    returns:
        lstImages: list of tuples containing folder and filename for each image
    """
    lstImages = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(extensions):
                lstImages.append((root,name))
    return lstImages

def readJSONAnnotation(pathAnno):
    """ Load JSON annotation file

    params:
        pathAnno: Where the JSON annotation file is located

    returns:
        anno: Dictionary containing the annotation data
    """
    
    with open(pathAnno) as data_file:
        anno = json.load(data_file)
    
    return anno

def writeJSONAnnotation(pathAnno, anno):
    """ Write annotation to JSON file

    params:
        pathAnno: Where to save the JSON annotation file
        anno: Dictionary containing the annotation data
    """
    with open(pathAnno, 'w') as data_file:
        json.dump(anno, data_file)

def printWithStyle(msg, colour=None, formating=None):
    """ Print to terminal with colour and formating style,

    params:
        msg: Message to print
        colour: string with color for msg. Supported values {'GRAY','RED','GREEN','YELLOW','BLUE','MAGENTA','CYAN'}
        formating: list of strings with formating options. Supported values: {'BOLD','ITALIC','UNDERLINE'}
    """

    class pcolors:
        ENDC = '\033[0m'
        GRAY = '\033[90m'
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\33[96m'

    class pformats:
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'

    # ADD colour
    if colour == 'GRAY':
        msg = pcolors.GRAY + msg + pcolors.ENDC
    elif colour == 'RED':
        msg = pcolors.RED + msg + pcolors.ENDC
    elif colour == 'GREEN':
        msg = pcolors.GREEN + msg + pcolors.ENDC
    elif colour == 'YELLOW':
        msg = pcolors.YELLOW + msg + pcolors.ENDC
    elif colour == 'BLUE':
        msg = pcolors.BLUE + msg + pcolors.ENDC
    elif colour == 'MAGENTA':
        msg = pcolors.MAGENTA + msg + pcolors.ENDC
    elif colour == 'CYAN':
        msg = pcolors.CYAN + msg + pcolors.ENDC
    else:
        msg = msg
    
    # ADD formating
    if formating:
        if "BOLD" in formating:
            msg = pformats.BOLD + msg + pformats.ENDC
        if "ITALIC" in formating:
            msg = pformats.ITALIC + msg + pformats.ENDC
        if "UNDERLINE" in formating:
            msg = pformats.UNDERLINE + msg + pformats.ENDC

    print(msg)

def unique_lst(lst): 
    # intilize a null list
    lst_unique = []
      
    # traverse for all elements 
    for element in lst:
        if element not in lst_unique:
            lst_unique.append(element)

    return lst_unique

def addPolygon2Image(img, polygon, color=(255,0,0), alpha=0.2, thickness=5):
    """ Add polygon to an image

    params:
        img: cv2 image
        polygon: array of corner coordinates
        color: color for polygon
        alpha: transparency of polygon fill
        thickness: thickness of polygon lines

    returns:
        output: image with added polygon
    """
    overlay = img.copy()
    output = img.copy()
    polygon = [np.int32(polygon)]

    # cv2.polylines(overlay, [corners], 1, color)
    cv2.fillPoly(overlay, polygon, color)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    cv2.polylines(output, polygon, 1, color, thickness)

    return output

def bndbox2polygon(bndbox):
    polygon = np.array([[bndbox['xmin'],bndbox['ymin']],
                        [bndbox['xmin'],bndbox['ymax']],
                        [bndbox['xmax'],bndbox['ymax']],
                        [bndbox['xmax'],bndbox['ymin']]],dtype=np.int32)
    return polygon

def addBndBoxes2Image(path_img, color=(255,0,0), alpha=0.2, thickness=5):
    """ Add a polygon for each bounding box annotation associated with the image

    params:
        path_img: path to image
        color: color of the polygons
        alpha: transparency of polygons fill
        thickness: thickness of polygons lines

    returns:
        output: image with added polygons for each bounding box
    """
    path_anno = path_img.replace('.jpg','.json')
    image = cv2.imread(path_img)
    anno = readJSONAnnotation(path_anno)

    output = image.copy()
    for plant in anno['plants']:
        bndbox = plant['bndbox']
        coor_bndbox = bndbox2polygon(bndbox)

        output = addPolygon2Image(output,coor_bndbox, color,alpha, thickness)
    
    return output

def move_files(image_list, dest_dir, anno_list=None):
    if anno_list == None:   # if only images need to be moved to a different folder
        for image in image_list:
            src_dir, path_image = image
            shutil.copy(os.path.join(src_dir, path_image), os.path.join(dest_dir, path_image))
            
    else:                   # if both images and annotations are present
        for image, anno in zip(image_list, anno_list):
            src_dir, path_image = image
            src_dir, path_anno = anno
            shutil.copy(os.path.join(src_dir, path_image), os.path.join(dest_dir, path_image))
            shutil.copy(os.path.join(src_dir, path_anno), os.path.join(dest_dir, path_anno))

def process_folders(folder_path):
    num_images = 0
    images_by_index = {}
    masks_by_index = {}
    
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)
        
        for repo_name in os.listdir(subfolder_path):
            repo_subfolder_path = os.path.join(subfolder_path, repo_name)
            plant_images = []
            
            for subsubfolder_name in os.listdir(repo_subfolder_path):

                if subsubfolder_name == "PNGImages":
                    subsubfolder_path = os.path.join(repo_subfolder_path, subsubfolder_name)
                    
                    for index, image_name in enumerate(os.listdir(subsubfolder_path)):
                        image_path = os.path.join(subsubfolder_path, image_name)
                        image = Image.open(image_path).resize((128, 128)) 
                        
                        if index not in images_by_index:
                            images_by_index[index] = []
                        images_by_index[index].append(np.array(image))
                
                if subsubfolder_name == "SegmentationObject":
                    subsubfolder_path = os.path.join(repo_subfolder_path, subsubfolder_name)
                    
                    for index, mask_name in enumerate(os.listdir(subsubfolder_path)):
                        mask_path = os.path.join(subsubfolder_path, mask_name)
                        mask = Image.open(mask_path).resize((128, 128)) 
                        
                        if index not in masks_by_index:
                            masks_by_index[index] = []
                        masks_by_index[index].append(np.array(mask))
                    
    return images_by_index, masks_by_index

def create_pictures_from_patches(set_of_images, save_path, is_mask=False, images_per_picture=10, padding_size=128):
    pictures = []
    image_name = "plant_image_"
    file_name = "image_"
    
    for i, images in enumerate(set_of_images):
        for j in range(0, len(images), images_per_picture):
            patches = images[j:j+images_per_picture]

            if patches.shape[0] == 10:
                patches = patches.permute((0, 3, 1, 2))
                padded_image = torch.zeros((padding_size*5, padding_size*5, 3))
                
                image_index = 0
                for r in range(0, padded_image.shape[0], padding_size):
                    for k, c in enumerate(range(0, padded_image.shape[1], padding_size)):
                        if k%2 == 1:
                            img = transforms.ToPILImage()(patches[image_index])
                            plt.show()
                            padded_image[r:r+padding_size, c:c+padding_size] = torch.tensor(np.array(img)/255.0)
                            image_index += 1
                        
                image_array = (padded_image.numpy() * 255).astype(np.uint8)
                image = Image.fromarray(image_array)
                if is_mask:
                    image.save(os.path.join(save_path, file_name + f"t{i}_no{j}_mask.jpg"))
                else:
                    image.save(os.path.join(save_path, file_name + f"t{i}_no{j}.jpg"))                 

def load_weights(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=(torch.device(device)))
    model.load_state_dict(state_dict)
    return model