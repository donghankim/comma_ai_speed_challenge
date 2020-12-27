import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb, os

def train(config, model, train_set):
    train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True)
    model_path = config.model_path + config.model + ".pth"

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)
    loss_hist = []

    for epoch in range(config.epochs):
        print(f"Training epoch {epoch+1}...")
        for images, labels in tqdm(train_loader):
            labels = torch.unsqueeze(labels, 0).view(-1,1)
            y_hat = model(images)
            loss = torch.sqrt(criterion(y_hat, labels))
            loss_hist.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"loss: {loss_hist[-1]}\n")

    torch.save(model.state_dict(), model_path)
    print("success")

def validate(config, model, val_set):
    test_loader = DataLoader(val_set, batch_size = config.batch_size, shuffle = False)
    model_path = config.model_path + config.model + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No available model weights.")
        return

    criterion = nn.MSELoss()
    loss_hist = []
    with torch.no_grad():
        for images, labels in test_loader:
            labels = torch.unsqueeze(labels, 0).view(-1, 1)
            y_hat = model(images)
            loss = torch.sqrt(criterion(y_hat, labels))
            loss_hist.append(loss.item())

    print(f"Test Loss: {sum(loss_hist)/len(loss_hist)}")


def test(config, model, test_set):
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    model_path = config.model_path + config.model + ".pth"
    predictions = []

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No available model weights.")
        return

    with torch.no_grad():
        print("Making predictions...")
        for images, labels in tqdm(test_loader):
            labels = torch.unsqueeze(labels, 0).view(-1, 1)
            y_hat = model(images)
            for i in range(len(y_hat)):
                predictions.append(y_hat[i].item())

    # writing to text file
    with open(os.path.join(config.result_path, f'{config.model}_result.txt'), 'w') as file_obj:
        print("Writing to text file...")
        for speed in tqdm(predictions):
            file_obj.write("%f\n" % speed)

    print("Predictions Complete")
    pdb.set_trace()




