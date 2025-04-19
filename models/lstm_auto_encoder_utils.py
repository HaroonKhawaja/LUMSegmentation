import torch
import matplotlib.pyplot as plt


def visualize_model_predictions(model, data_loader, device):
    model = model.eval()
    model = model.to(device)

    # take one batch
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)

            # getting batch of output images
            outputs = model(images, masks)
            outputs = (outputs > 0.5).float()

            # from one batch, visualize one image
            for image, mask, output in zip(images, masks, outputs):
                fig, ax = plt.subplots(1, 2, figsize=(10,10))

                denormalized_image = denormalize(image).detach().cpu().numpy().squeeze().transpose((1, 2, 0))
                denormalized_image = denormalized_image.clip(0, 1)

                image = image.cpu().numpy().transpose((1, 2, 0))
                image = image.clip(0, 1)
                
                mask = mask.detach().cpu().numpy().transpose((1, 2, 0))
                mask = mask.clip(0, 1)

                output = output.cpu().numpy().transpose((1, 2, 0))
                output = output.clip(0, 1)

                # displaying original image with prediction
                ax[0].set_title('Original Image with Predicted Mask')
                ax[0].imshow(denormalized_image)    
                ax[0].imshow(output, alpha=0.5)    
                ax[0].axis('off')

                # displaying original mask with prediction
                ax[1].set_title('Original Mask with Predicted Mask')
                ax[1].imshow(mask)    
                ax[1].imshow(output, alpha=0.5)   
                ax[1].axis('off')

                plt.show()

def IoU(groundtruth_mask, pred_mask, smooth=1e-6):
    intersection = torch.sum(groundtruth_mask * pred_mask)
    union = torch.sum(groundtruth_mask) + torch.sum(pred_mask) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou

def precision_score(groundtruth_mask, pred_mask, smooth=1e-6):
    intersection = torch.sum(pred_mask * groundtruth_mask)
    total_pixel_pred = torch.sum(pred_mask)

    precision = (intersection + smooth) / (total_pixel_pred + smooth)
    return precision

def recall_score(groundtruth_mask, pred_mask, smooth=1e-6):
    intersection = torch.sum(pred_mask * groundtruth_mask)
    total_pixel_truth = torch.sum(groundtruth_mask)

    recall = (intersection + smooth) / (total_pixel_truth + smooth)
    return recall

def accuracy(groundtruth_mask, pred_mask, smooth=1e-6):
    true_positive = torch.sum(pred_mask * groundtruth_mask)
    true_negative = torch.sum((1 - pred_mask) * (1 - groundtruth_mask))
    total = groundtruth_mask.numel()

    acc = (true_positive + true_negative + smooth) / (total + smooth)
    return acc

def dice_coef(groundtruth_mask, pred_mask, smooth=1e-6):
    intersect = torch.sum(pred_mask * groundtruth_mask)
    total_sum = torch.sum(pred_mask) + torch.sum(groundtruth_mask)
    
    dice = (2 * intersect + smooth) / (total_sum + smooth)
    return dice

def evaluate_model(model, data_loader, device):
    model = model.eval()
    model = model.to(device)

    iou_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    dice_list = []
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images, masks)
            pred_masks = (outputs > 0.5).float()

            for prediction_mask, true_mask in zip(pred_masks, masks):
                iou_list.append(IoU(prediction_mask, true_mask))
                precision_list.append(precision_score(prediction_mask, true_mask))
                recall_list.append(recall_score(prediction_mask, true_mask))
                accuracy_list.append(accuracy(prediction_mask, true_mask))
                dice_list.append(dice_coef(prediction_mask, true_mask))
    m_iou = torch.mean(torch.tensor(iou_list))
    m_precision = torch.mean(torch.tensor(precision_list))
    m_recall = torch.mean(torch.tensor(recall_list))
    m_acc = torch.mean(torch.tensor(accuracy_list))
    m_dice = torch.mean(torch.tensor(dice_list))

    print(f'Precison: {m_precision:.3f}')
    print(f'Recall: {m_recall:.3f}')
    print(f'Accuracy: {m_acc:.3f}')
    print(f'Dice: {m_dice:.3f}')
    print(f'mIoU: {m_iou:.3f}')

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(image.device)
    return (image * std) + mean