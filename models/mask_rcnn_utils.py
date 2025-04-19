import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

def maskrcnn_IoU(groundtruth_mask, pred_mask, smooth=1e-6):
    intersection = np.sum(groundtruth_mask * pred_mask)
    union = np.sum(groundtruth_mask + pred_mask) - intersection

    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou

def maskrcnn_precision_score(groundtruth_mask, pred_mask):
    intersection = np.sum(pred_mask * groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    
    precision = np.mean(intersection/total_pixel_pred)
    return precision

def maskrcnn_recall_score(groundtruth_mask, pred_mask):
    intersection = np.sum(pred_mask * groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    
    recall = np.mean(intersection/total_pixel_truth)
    return recall

def maskrcnn_accuracy(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask == pred_mask)

    acc = np.mean(xor/(union + xor - intersect))
    return acc

def maskrcnn_dice_coef(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_sum = np.sum(pred_mask) + np.sum(groundtruth_mask)

    dice = np.mean(2 * intersect/total_sum)
    return dice

def evaluate_maskrcnn_model(model, data_loader, device, score_threshold=0.5):
    model = model.eval()
    model = model.to(device)
    
    resize = 256
    correct = 0
    count = 0

    iou_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []
    dice_list = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            target_batch = [{k: v.to(device) for k,v in t.items()} for t in targets]

            output_batch = model(images)

            for target, output in zip(target_batch, output_batch):
                
                score_mask = output['scores'] > score_threshold
                pred_boxes = output['boxes'][score_mask]
                true_boxes = target['boxes']
                pred_leaves_count = pred_boxes.shape[0]
                true_leaves_count = true_boxes.shape[0]

                if true_leaves_count == pred_leaves_count:
                    correct += 1
                # count += 1

                combined_mask_true = np.zeros((resize, resize), dtype=np.uint8)
                combined_mask_pred = np.zeros((resize, resize), dtype=np.uint8)
                for true_mask, pred_mask in zip(target['masks'], output['masks'][score_mask]):
                    true_mask = true_mask.detach().cpu().numpy().squeeze()
                    true_mask = (true_mask > 0.5).astype('uint8')
                    combined_mask_true = np.maximum(combined_mask_true, true_mask)

                    pred_mask = pred_mask.detach().cpu().numpy().squeeze()
                    pred_mask = (pred_mask > 0.5).astype('uint8')
                    combined_mask_pred = np.maximum(combined_mask_pred, pred_mask)

                iou_list.append(maskrcnn_IoU(combined_mask_pred, combined_mask_true))
                precision_list.append(maskrcnn_precision_score(combined_mask_pred, combined_mask_true))
                recall_list.append(maskrcnn_recall_score(combined_mask_pred, combined_mask_true))
                accuracy_list.append(maskrcnn_accuracy(combined_mask_pred, combined_mask_true))
                dice_list.append(maskrcnn_dice_coef(combined_mask_pred, combined_mask_true))

    m_iou = torch.mean(torch.tensor(iou_list))
    m_precision = torch.mean(torch.tensor(precision_list))
    m_recall = torch.mean(torch.tensor(recall_list))
    m_acc = torch.mean(torch.tensor(accuracy_list))
    m_dice = torch.mean(torch.tensor(dice_list))
    leaf_count_accuracy = (correct / count) * 100
    print(f'Leaf Count Accuracy: {leaf_count_accuracy:.3f}%')
    print(f'Precison: {m_precision:.3f}')
    print(f'Recall: {m_recall:.3f}')
    print(f'Accuracy: {m_acc:.3f}')
    print(f'Dice: {m_dice:.3f}')
    print(f'mIoU: {m_iou:.3f}')

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, -1, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, -1, 1, 1).to(image.device)
    return (image * std) + mean