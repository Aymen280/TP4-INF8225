from cplx_dataset import cplx_SAR_dataset_npy
from cplx_unet import ComplexUNet

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from torchinfo import summary


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

    model = ComplexUNet(1,1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_model = model.state_dict()
    best_epoch = 0
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

