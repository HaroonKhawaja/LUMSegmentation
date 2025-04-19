import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class MaskRCNNPredictorWithLSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(MaskRCNNPredictorWithLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=in_channels, 
                            hidden_size=hidden_dim, 
                            num_layers=1, 
                            batch_first=True)
        
        self.conv5_mask = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
        self.mask_fcn_logits = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
    
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        lstm_in = x.view(batch_size, channels, (height * width))
        lstm_in = lstm_in.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out = lstm_out.permute(0, 2, 1).view(batch_size, self.hidden_dim, height, width)
        
        x = self.conv5_mask(lstm_out)
        x = nn.functional.relu(x)
        x = self.mask_fcn_logits(x)
        return x

def create_opt_mask_rcnn_model(num_classes=2, device="cuda"):
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
    hidden_dim = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictorWithLSTM(in_channels=in_features_mask, hidden_dim=hidden_dim, num_classes=2)

    # setting the model's device
    model = model.to(device=device)
    
    return model
