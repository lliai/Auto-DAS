from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloaders(batch_size):
    # Define transforms to normalize the data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # Define the MNIST dataset
    train_dataset = datasets.MNIST(
        root='data/', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root='data/', train=False, transform=transform, download=True)

    # Define the dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader
