import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb, os

def train(config, model, train_set):
    if config.device == torch.device('cpu'):
        train_loader = DataLoader(train_set, batch_size = config.batch_size, shuffle = True)
    else:
        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, pin_memory = True)

    writer = SummaryWriter('runs/train')
    step = 0
    model_path = config.model_path + config.model + ".pth"
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)

    for epoch in range(config.epochs):
        print(f"Training epoch {epoch+1}...")
        for images, labels in tqdm(train_loader):
            labels = torch.unsqueeze(labels, 0).view(-1,1)
            y_hat = model(images)
            loss = torch.sqrt(criterion(y_hat, labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Training Loss", loss, global_step = step)
            step += 1

        print(f"Final loss: {loss.item()}\n")

    torch.save(model.state_dict(), model_path)
    print("Training Success.")


def validate(config, model, val_set):
    model_path = config.model_path + config.model + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("No available model weights.")
        return

    # hyperparam tuning
    batch_sizes = [1,64,1024]
    l_r = [0.1,0.01,0.001,0.0001]

    for size in tqdm(batch_sizes):
        for lr in l_r:
            writer = SummaryWriter(f'runs/val/{config.model}_{size,lr}')
            step = 0
            if config.device == torch.device('cpu'):
                val_loader = DataLoader(val_set, batch_size=size,shuffle=False)
            else:
                val_loader = DataLoader(val_set, batch_size=size, shuffle=False,pin_memory=True)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            for epoch in range(1):
                losses = []
                for images, labels in val_loader:
                    labels = torch.unsqueeze(labels, 0).view(-1, 1)
                    y_hat = model(images)
                    loss = torch.sqrt(criterion(y_hat, labels))
                    losses.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    img_grid = torchvision.utils.make_grid(images)
                    writer.add_image("images", img_grid)
                    writer.add_histogram("fc1", model.fc4.weight)
                    writer.add_scalar("Training Loss", loss, global_step = step)
                    writer.add_hparams({'lr': lr, 'bsize':size},
                                        {'loss':sum(losses)/len(losses)})
                    step += 1

    #torch.save(model.state_dict(), model_path)

    """
    with torch.no_grad():
        for images, labels in test_loader:
            labels = torch.unsqueeze(labels, 0).view(-1, 1)
            y_hat = model(images)
            loss = torch.sqrt(criterion(y_hat, labels))
            print(loss.item())
    """


def test(config, model, test_set):
    if config.device == torch.device('cpu'):
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    else:
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, pin_memory = True)

    predictions = []
    model_path = config.model_path + config.model + ".pth"
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




