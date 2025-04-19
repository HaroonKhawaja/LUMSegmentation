import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
import torchvision

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def create_mask_rcnn_model(num_classes=2, device="cpu"):
    # initializing the mask rcnn model with default weights
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT, trainable_backbone_layers=5)

    # getting the number of input features for classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # getting the number of output channels for the mas predictor
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels

    # replacing the box predictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=2)

    # replacing the mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=2)

    # setting the model's device
    model = model.to(device=device)
    
    return model

def visualize_mask_rcnn_predictions(model, data_loader, device, score_threshold=0.5):
    model = model.eval()
    model = model.to(device)

    # take one batch
    with torch.no_grad():
        for images, _ in data_loader:
            images = list(image.to(device) for image in images)
            # getting batch of output images
            outputs = model(images)

            # from one batch, visualize one image
            for i, output in enumerate(outputs):
                fig, ax = plt.subplots(1, 1, figsize=(15,10))
                colors = plt.cm.jet(np.linspace(0, 1, len(outputs[i]['masks'])))
                image = images[i]
                denormalized_image = denormalize(image).detach().cpu().numpy().squeeze().transpose((1, 2, 0))
                denormalized_image = denormalized_image.clip(0, 1)

                # displaying original image
                ax.imshow(denormalized_image)    
                ax.set_title('Original Image with mask')

                image = image.cpu().numpy().transpose((1, 2, 0))
                image = image.clip(0, 1)

                height, width = image.shape[:2]
                combined_mask = np.zeros((height, width), dtype=np.uint8)

                masks = []
                for i, (box, label, score, mask) in enumerate(zip(output['boxes'], output['labels'], output['scores'], output['masks'])):
                    if score > score_threshold:
                        # processing predicted masks
                        mask = mask.detach().cpu().numpy().squeeze()
                        mask = (mask > 0.5).astype('uint8')
                        masks.append(mask)
                        combined_mask = np.maximum(combined_mask, mask)

                        # processing predicted boxes
                        xmin, ymin, xmax, ymax = box.cpu()
                        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                        # Adding score to each predicted leaf
                        if label == 1:
                            label = 'Crop'
                        # ax.text(xmin, ymin, f'Label: {label} - Score: {score:.3f}', color="white", bbox=dict(facecolor='red', alpha=0.5))
                
                # displaying generated mask
                plt.imshow(combined_mask, alpha=0.5)    
                ax.axis('off')
                plt.show()
       
def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(image.device)
    return (image * std) + mean
