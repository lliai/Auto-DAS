import unittest
from unittest import TestCase

from diswotv2.datasets.cifar100 import get_cifar100_dataloaders

trainloader, valloader = get_cifar100_dataloaders(
    './data/cifar100', batch_size=128, num_workers=4)
