from tqdm import tqdm

from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.models.resnet import resnet20
from diswotv2.searchspace.interactions.parallel import ParaInteraction

# model
model_s = resnet20()
model_t = resnet20()

# dataloader
train_loader, val_loader = get_cifar100_dataloaders('./data', 128, 0)

# mini-batch
image, label = next(iter(train_loader))

# build structure with many trials
valid = 0
for i in tqdm(range(1000)):
    criterion = ParaInteraction()
    loss_kd = criterion(image, label, model_t, model_s)
    if loss_kd != -1:
        valid += 1

print(f'validity rate: {valid/1000}')
