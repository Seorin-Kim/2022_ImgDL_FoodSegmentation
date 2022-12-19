import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import UNet
import UNetDataset
from utils import iou_metric
from utils import seed_everything

seed_everything(42)
device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_val = np.load('x_val.npy')
y_val = np.load('y_val.npy')

train_dataset = UNetDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataset = UNetDataset(x_val, y_val)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

model = UNet(in_channel=3, out_channel=1).to(device)

epochs = 25
alpha = 5
batch_size = 16
criterion=nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_iou_sum = 0
valid_iou_sum = 0

train_loss_list = []
val_loss_list = []
train_iou_list = []
val_iou_list = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    train_iou = 0

    for image, mask in train_loader:
        image = image.to(device)
        mask = mask.to(device)
        optimizer.zero_grad()
        outputs = model(image.float())
    
        loss = criterion(outputs.float(), mask.float())
        train_loss += loss

        train_iou += iou_metric(outputs, mask)
        rev_iou = 16 - iou_metric(outputs, mask)
        loss += alpha * rev_iou

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_loss = 0
        valid_iou = 0

        for image_val, mask_val in valid_loader:
            image_val = image_val.to(device)
            mask_val = mask_val.to(device)
            output_val = model(image_val.float())
            valid_loss += criterion(output_val.float(), mask_val.float())
            valid_iou += iou_metric(output_val, mask_val)

    print("Epoch ", epoch + 1, " Training Loss: ", train_loss/len(train_loader), "Validation Loss: ", valid_loss/len(valid_loader))
    print("Training mAP: ", train_iou/len(train_loader), "Validation mAP: ", valid_iou/len(valid_loader))
    train_iou_sum += train_iou/len(train_loader)
    valid_iou_sum += valid_iou/len(valid_loader)

    # visualization
    train_loss_list.append(train_loss/len(train_loader))
    val_loss_list.append(valid_loss/len(valid_loader))
    train_iou_list.append(train_iou/len(train_loader))
    val_iou_list.append(valid_iou/len(valid_loader)
)
    
print("Training Mean mAP: {:.2f}".format(train_iou_sum/epochs), " Validation Mean mAP: {:.2f}".format(valid_iou_sum/epochs))
