from config import get_args
import preprocess, models, runner
import torch
from torchvision import datasets, transforms
from dataset import FrameDataset
from torch.utils.data import random_split
import pdb, os

def main():
    config = get_args()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # preprocess
    if config.preprocess_frames:
        preprocess.get_frames(config.train_vid, config.train_frames)
        preprocess.get_frames(config.test_vid, config.test_frames)

    if config.create_csv:
        train_speeds = preprocess.read_speed(config.train_speeds)
        preprocess.create_csv(config.train_frames, train_speeds, config.csv_path)

    # dataset creation
    dataset = FrameDataset(config.csv_path, config.train_frames)
    train_set, val_set = random_split(dataset, [16320, 4080])

    # test set creation
    transform = transforms.Compose([
        transforms.Resize((66, 220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    test_set = datasets.ImageFolder(config.test_frames, transform = transform)

    # model selection
    if config.model == 'simpleCNN':
        model = models.simpleCNN().to(config.device)
    elif config.model == 'ResNet':
        model = models.ResNet().to(config.device)

    # train/val/test
    if config.train:
        runner.train(config, model, train_set)
    elif config.val:
        runner.validate(config, model, val_set)
    elif config.test:
        runner.test(config, model, test_set)


if __name__ == '__main__':
    main()


"""
Things to get done:
1. Normalize features before entering into model
1. ResNet model (transfer learning)
2. LSTM connected

Potential model ideas:
1. feature matching - get (x,y) cooridnates
2. select 100 most prominent macthes
3. find the difference in pixel values
4. encode with the speed -> feed into FCN
"""


"""
RESULTS:
1. simpleCNN - train = 5.77, val = 3.06 -> extreme overfitting
2. ResNet (transfer) -
3. LSTM embedding -
4. Potential Model -
"""
