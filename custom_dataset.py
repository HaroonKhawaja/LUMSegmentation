import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon2mask
import json

class SegmentationDataset():
    def __init__(self, image_folder_path, mask_folder_path, transform=None, resize=256):
        self.image_folder_path = image_folder_path
        self.mask_folder_path = mask_folder_path
        self.transform = transform
        self.resize = resize

        self.images = [os.path.join(self.image_folder_path, image) 
                       for image in os.listdir(self.image_folder_path) 
                       if image.endswith(".jpg")]
        
        self.masks = [os.path.join(self.mask_folder_path, mask) 
                      for mask in os.listdir(self.mask_folder_path) 
                      if any(mask.endswith(suffix) for suffix in ["-mask.png", "_mask.jpg"])]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # image processing
        path_image = self.images[idx]
        image = Image.open(path_image).convert('RGB')
        image = self.transform(image)
        
        # mask processing
        path_mask = self.masks[idx]
        mask = Image.open(path_mask).convert('L')
        mask = transforms.ToTensor()(mask)
        mask = transforms.Resize((self.resize, self.resize))(mask)
        
        return image, mask

class MaskRCNNSegmentationDataset():
    def __init__(self, data_folder, annotation_file, transforms=None):
        self.data_folder = data_folder
        self.transforms = transforms

        # initializing the annotations
        self.annotations = json.load(open(annotation_file))

        # storing the image rgb filenames in self.images list
        self.images = [os.path.join(data_folder, image['file_name']) for image in self.annotations['images']]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx, resize=256):
        # image processing
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)
        
        # mask processing
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
        ])
        mask_path = image_path.replace('cam2.jpg', 'cam2_mask.jpg')
        mask_original = Image.open(mask_path).convert('L')
        width, height = mask_original.size

        image_annotations = [annotation for annotation in self.annotations['annotations'] if annotation['image_id'] == idx]

        # max processing
        masks = []
        for annotation in image_annotations:
            segmentation = annotation['segmentation']
            poly = np.array(segmentation).reshape((-1, 2))
            poly = poly[:, [1, 0]]
            mask = polygon2mask((height, width), poly)
            mask = mask_transform(mask)
            masks.append(mask)


        # bounding box processing
        scale_x = resize/width
        scale_y = resize/height
        scaled_boxes = []
        for annotation in image_annotations:
            x, y, w, h = annotation['bbox']
            x = int(x * scale_x)
            w = int(w * scale_x)
            y = int(y * scale_y)
            h = int(h * scale_y)

            box = torch.tensor([x, y, x+w, y+h], dtype=torch.float32)
            scaled_boxes.append(box)

        boxes = torch.stack(scaled_boxes, dim=0)
        labels = torch.ones((len(image_annotations),), dtype=torch.int64)
        masks = torch.stack(masks, dim=0)
        masks = masks.squeeze(1)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([idx])

        return image, target
      
def visualize_semantic_dataloader(data_loader):
    images, masks = next(iter(data_loader))

    for image, mask in zip(images, masks):
        denormalized_image = denormalize(image).detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        denormalized_image = denormalized_image.clip(0, 1)

        image = image.detach().cpu().numpy().transpose((1, 2, 0))
        image = image.clip(0, 1)
        
        mask = mask.detach().cpu().numpy().transpose((1, 2, 0))
        mask = mask.clip(0, 1)
        break

    # plotting 
    num_pics = 4
    fig, ax = plt.subplots(1, num_pics, figsize=(10, 7))
    
    ax[0].imshow(denormalized_image)
    ax[0].set_title('Image')

    ax[1].imshow(image)
    ax[1].set_title('Preprocessed image')

    ax[2].imshow(mask)
    ax[2].set_title('Image Mask')

    ax[3].imshow(denormalized_image)
    ax[3].imshow(mask, alpha=0.5)
    ax[3].set_title('Image with mask')

    for i in range(num_pics):
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_maskrcnn_dataloader(data_loader):
    for images, targets in data_loader:
        image = images[0]
        target = targets[0]
        break
    
    denormalized_image = denormalize(image).detach().cpu().numpy().squeeze().transpose((1, 2, 0))
    denormalized_image = denormalized_image.clip(0, 1)

    image = image.detach().cpu().numpy().transpose((1, 2, 0))
    image = image.clip(0, 1)
    
    height, width = image.shape[:2]
    combined_mask = np.zeros((height, width), dtype=np.uint8)

    # plotting 
    num_pics = 5
    fig, ax = plt.subplots(1, num_pics, figsize=(20, 8))
    
    for i, mask in enumerate(target['masks']):
        single_mask = mask.numpy()
        single_mask = single_mask.clip(0, 1)
        combined_mask = np.maximum(combined_mask, single_mask)

    for bbox in target["boxes"]:
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
        else: 
            x1, y1, width, height = bbox
            
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax[4].add_patch(rect)
    
    ax[0].imshow(denormalized_image)
    ax[0].set_title('Image')

    ax[1].imshow(image)
    ax[1].set_title('Processed image')

    ax[2].imshow(combined_mask)
    ax[2].set_title('Image Mask')

    ax[3].imshow(denormalized_image)
    ax[3].imshow(combined_mask, alpha=0.5)
    ax[3].set_title('Image with mask')

    ax[4].imshow(denormalized_image)
    ax[4].imshow(combined_mask, alpha=0.5)
    ax[4].set_title('Image with mask and boxes')

    for i in range(num_pics):
        ax[i].axis('off')

    plt.tight_layout()
    plt.show()

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(image.device)
    return (image * std) + mean
