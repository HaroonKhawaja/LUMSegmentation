import torch
from torchvision import transforms
from tqdm import tqdm

def segmentation_training(model, train_loader, num_epochs, criterion, optimizer, device, weights_path, resize=None):
    train_loss_history = []

    model.train()
    for epoch in range(num_epochs):
        batch_loss = 0
        
        # Training
        for images, masks in tqdm(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            output = model(images, masks)
            
            if resize:
                masks = transforms.Resize((resize, resize))(masks)
            loss = criterion(output, masks)
            
            loss.backward()
            optimizer.step()

            batch_loss += loss
        
        average_batch_loss = batch_loss / len(train_loader)
        train_loss_history.append(average_batch_loss)

        print(f"[{epoch + 1} / {num_epochs}] Train Loss: {average_batch_loss:3f}")

        # Saving the weights with the least amount of loss
        if (epoch==0) or weights_path and average_batch_loss < min(train_loss_history[:-1]):
            torch.save(model.state_dict(), weights_path)

    return train_loss_history 


def train_mask_rcnn(model, train_loader, optimizer, scaler, num_epochs, device, weights_path):
    train_losses = []

    for epoch in range(num_epochs):
        loss = train_epoch(model, train_loader, optimizer, scaler, num_epochs, device, epoch)
        train_losses.append(loss)

        # saving the weights
        if weights_path and (epoch == 0 or loss < min(train_losses[:-1])):
                torch.save(model.state_dict(), weights_path)
    
    return train_losses

def train_epoch(model, train_loader, optimizer, scaler, num_epochs, device, epoch):
    model = model.train()
    total_loss = 0.0

    batch = tqdm(train_loader)
    for images, targets in batch:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        # Initializing gradients
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            losses = sum([loss for loss in loss_dict.values()])
        # Back propagation and optimization
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()
        batch.set_postfix({'Batch Loss': total_loss})
    avg_loss = total_loss / len(train_loader)

    print(f'EPOCH: [{epoch + 1} / {num_epochs}] - Loss: {avg_loss:.3f}')
    return avg_loss