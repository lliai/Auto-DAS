import unittest
from copy import deepcopy

from diswotv2.datasets.cifar100 import get_cifar100_dataloaders
from diswotv2.models.resnet import resnet20
from diswotv2.searchspace.interactions import ParaInteraction


class TestParallel(unittest.TestCase):

    def setUp(self) -> None:
        # prepare model
        self.t = resnet20()
        self.s = resnet20()

        # prepare data
        t_loader, s_loader = get_cifar100_dataloaders(
            data_folder='./data', batch_size=32, num_workers=2)
        dataiter = iter(t_loader)
        self.images, self.labels = next(dataiter)

        # prepare parallel structure
        self.para1 = ParaInteraction(n_nodes=3, mutable=True)
        self.para2 = ParaInteraction(n_nodes=3, mutable=True)

    def test_repr(self):
        print('test repr')
        print(self.para1)
        print(self.para2)

    def test_delete(self):
        print('test delete unary')
        print(self.para1)
        self.para1.delete_unary(0)
        print(self.para1)

    def test_insert(self):
        print('test insert unary')
        print(self.para1)
        self.para1.insert_unary(0)
        print(self.para1)

        print('test insert abslog')
        self.para1.insert_unary(3, 'abslog')
        print(self.para1)

    def test_mutate(self):
        print('test mutate')
        tmp = deepcopy(self.para1)
        mutated = self.para1.mutate()
        print(self.para1)
        print(mutated)

        self.assertEqual(self.para1, tmp)

    def test_crossover(self):
        print('test crossover')
        print('original structure para1, para2')
        print('para1:', self.para1)
        print('para2:', self.para2)
        tmp = deepcopy(self.para1)
        print('tmp:', tmp)
        crossed = self.para1.cross_over(self.para2)
        print('After crossover')
        print(crossed)

        print('below should be the same')
        self.assertEqual(self.para1, tmp)

    def test_forward(self):
        loss = self.para1(self.images, self.labels, self.t, self.s)
        print('loss:', loss)


if __name__ == '__main__':
    unittest.main()
