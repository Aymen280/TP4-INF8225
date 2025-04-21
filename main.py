from cplx_dataset import cplx_SAR_dataset_npy
from cplx_unet import CplxUnet

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchinfo import summary

import matplotlib.pyplot as plt

def load_dataset(data_path, labels_path, train_part=0.7, batch_size=4):
    full_dataset = cplx_SAR_dataset_npy(data_path, labels_path)

    total_size = len(full_dataset)
    train_size = int(train_part * total_size)
    val_part = (1-train_part)/2
    val_size = int(val_part * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def train_epoch(model, optimizer, loss_fn, train_loader, epoch, device,):
    model.train()
    train_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Training epoch {epoch+1}"):  
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        outputs_real = outputs.real

        loss = loss_fn(outputs_real, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)  # * batch_size

    return train_loss / len(train_loader.dataset)

def validate_epoch(model, loss_fn, val_loader, epoch, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, mask in tqdm(val_loader, desc=f"Validation epoch {epoch+1}"):
            images = images.to(device)
            mask = mask.to(device)

            outputs = model(images)
            outputs_real = outputs.real
            loss = loss_fn(outputs_real, mask)
            val_loss += loss.item() * images.size(0)
    return val_loss / len(val_loader.dataset)

def visualize(images ,outputs, mask, num_samples=4):
    plt.figure(figsize=(12, 6))
    print(images.dtype)
    if images.dtype == torch.complex64 or images.dtype == torch.complex128:
        amplitude = torch.abs(images)
        phase = torch.angle(images)
        outputs = outputs.real
    for i in range(min(num_samples, images.size(0))):
        # Amplitude de l'image originale
        plt.subplot(4, num_samples, i+1)
        plt.imshow(amplitude[i].permute(1, 2, 0).cpu(), cmap='jet')
        plt.title("Amplitude de l'image")
        plt.axis('off')

        # Phase de l'image originale
        plt.subplot(4, num_samples, i+1+num_samples)
        plt.imshow(phase[i].permute(1, 2, 0).cpu(), cmap='jet')
        plt.title("Phase de l'image")
        plt.axis('off')

        # Masque réel
        plt.subplot(4, num_samples, i+1+2*num_samples)
        plt.imshow(mask[i][0].cpu(), cmap='gray')
        plt.title("Mask")
        plt.axis('off')

        # Prédiction
        plt.subplot(4, num_samples, i+1+3*num_samples)
        plt.imshow((outputs[i][0].cpu() > 0.5), cmap='gray')
        plt.title("Pred")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def test_model(model, loss_fn, test_loader, device, threshold=0.5):
    model.eval()
    test_loss = 0

    tp = 0
    fp = 0
    fn = 0

    iou_scores = []
    dice_scores = []

    with torch.no_grad():
        for i, (images, mask) in enumerate(tqdm(test_loader, desc=f"Test")):
            images = images.to(device)
            mask = mask.to(device)

            outputs = model(images)
            outputs_real = outputs.real
            loss = loss_fn(outputs_real, mask)
            test_loss += loss.item() * images.size(0)

            outputs = (outputs_real>threshold).float()

            tp += ((outputs == 1) & (mask == 1)).sum().item()
            fp += ((outputs == 1) & (mask == 0)).sum().item()
            fn += ((outputs == 0) & (mask == 1)).sum().item()

            intersection = (outputs * mask).sum(dim=(1,2,3))
            union = ((outputs + mask)>=1).float().sum(dim=(1,2,3))

            iou = (intersection / (union + 1e-8)).cpu().numpy()
            iou_scores.extend(iou)

            total = outputs.sum(dim=(1,2,3)) + mask.sum(dim=(1,2,3))

            dice = (2 * intersection / (total + 1e-8)).cpu().numpy()
            dice_scores.extend(dice)

    total_loss = test_loss / len(test_loader.dataset)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    iou = sum(iou_scores) / len(iou_scores)

    dice = sum(dice_scores) / len(dice_scores)
    
    visualize(images, outputs, mask, num_samples = images.size(0))

    return total_loss, precision, recall, f1, iou, dice

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    num_epochs = 10 
    learning_rate = 1e-3
    batch_size = 4

    verbose = False

    train_loader, val_loader, test_loader = load_dataset(
        data_path="data/sar_data_cplx/", 
        labels_path="data/labels_1c/",
        train_part=0.7,
        batch_size=batch_size
    )

    if verbose:
        print("Taille du dataset d'entrainement :", len(train_loader.dataset))
        print("Taille du dataset de validation :", len(val_loader.dataset))
        print("Taille du dataset de test :", len(test_loader.dataset))

        for i, batch in enumerate(train_loader):
            print("Batch", i, ":", batch)
            break

        for images, labels in train_loader:
            print("dtype des images :", images.dtype) 
            print("Shape des images :", images.shape)

            print("dtype des labels :", labels.dtype) 
            print("Shape des labels :", labels.shape)
            break

    model = CplxUnet(1,1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model = model
    best_epoch = 0
    training = False
    if training:
        for epoch in range(num_epochs):
            epoch_train_loss = train_epoch(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                train_loader=train_loader,
                epoch=epoch,
                device=device
            )
            epoch_val_loss = validate_epoch(
                model=model,
                loss_fn=loss_fn,
                val_loader=val_loader,
                epoch=epoch,
                device=device
            )
            
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model = model
                best_epoch = epoch+1
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {epoch_train_loss:.4f}| Val Loss: {epoch_val_loss:.4f}")

        torch.save(best_model.state_dict(),"best_model.pth")
        print(f"Meilleur modèle sauvegardé à l'epoch {best_epoch}")
    
    new_model = CplxUnet(1,1).to(device)
    new_model.load_state_dict(torch.load("best_model.pth"))
    print("Test du meilleur modèle : ")
    total_loss, precision, recall, f1, iou, dice = test_model(
        model = new_model,
        loss_fn=loss_fn,
        test_loader=test_loader,
        threshold=0.5,
        device=device
    )
    print(f"Perte du modèle sur le dataset de test : {total_loss:.4f}")
    print(f"Precision du modèle sur le dataset de test : {precision:.4f}")
    print(f"Recall du modèle sur le dataset de test : {recall:.4f}")
    print(f"F1-score du modèle sur le dataset de test : {f1:.4f}")
    print(f"IOU du modèle sur le dataset de test : {iou:.4f}")
    print(f"Dice score du modèle sur le dataset de test : {dice:.4f}")
