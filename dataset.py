import logging
from typing import Optional
import copy
import numpy as np
import torch
import torchvision
import tqdm

import util

def read_mnist(path: str) -> tuple:
    logging.info('Start reading mnist...')

    mnist_train = torchvision.datasets.MNIST(path, download=True, train=True)
    mnist_test = torchvision.datasets.MNIST(path, download=True, train=False)

    train_x = mnist_train.data.detach().numpy() / 256
    train_x = np.expand_dims(train_x, 1)
    train_y = mnist_train.targets.detach().numpy()

    test_x = mnist_test.data.detach().numpy() / 256
    test_x = np.expand_dims(test_x, 1)
    test_y = mnist_test.targets.detach().numpy()

    logging.info('Finish reading mnist.')
    return (train_x, train_y), (test_x, test_y)

def read_cifar(path: str) -> tuple:
    logging.info('Start reading cifar...')

    cifar_train = torchvision.datasets.CIFAR10(path, download=True, train=True)
    cifar_test = torchvision.datasets.CIFAR10(path, download=True, train=False)

    train_x = cifar_train.data / 256
    train_y = cifar_train.targets
    train_x = train_x.transpose(0, 3, 1, 2)

    test_x = cifar_test.data / 256
    test_y = cifar_test.targets
    test_x = test_x.transpose(0, 3, 1, 2)

    logging.info('Finish reading cifar...')
    return (train_x, train_y), (test_x, test_y)

def attack(
        x: np.ndarray,
        y: np.ndarray,
        p: float,
        backdoor_type: util.AttackType,
        src: Optional[int]=None,
        dst: Optional[int]=None
) -> tuple:
    """
    Perform single-target or all-to-all attack.
    :param x:
    :param y:
    :param p:
    :param backdoor_type:
    :param src: if not specified, perform all-to-all attack.
    :param dst: same.
    :return:
        x_new:
        y_new:
        backdoor_idxs:
        clean_idxs:
    """

    backdoor_count = int(len(x) * p)

    if src is not None:
        idxs = np.where(y == src)[0]
        backdoor_idxs = np.random.choice(idxs, backdoor_count)
    else:
        backdoor_idxs = np.random.choice(np.arange(0, len(x)), backdoor_count)

    x_new_list = []
    y_new_list = []
    channel = x.shape[1]
    size = x.shape[2]
    for idx in backdoor_idxs:
        x_new = copy.deepcopy(x[idx])
        if dst is not None:
            y_new = dst
        else:
            y_new = (y[idx] + 1) % 10

        for c in range(channel):
            if backdoor_type == util.AttackType.SINGLE_PIXEL:
                x_new[c, size - 2, size - 2] = 1
            else:
                x_new[c, size - 2, size - 2] = 1
                x_new[c, size - 3, size - 3] = 1
                x_new[c, size - 4, size - 2] = 1
                x_new[c, size - 2, size - 4] = 1
        x_new_list.append(x_new)
        y_new_list.append(y_new)

    x_new_arr = np.concatenate([x, np.array(x_new_list)])
    y_new_arr = np.concatenate([y, np.array(y_new_list)])
    return x_new_arr, y_new_arr, np.arange(len(x), len(x_new_arr)), np.arange(0, len(x))

class DataLoader:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 batch_size: int,
                 normalize: Optional[tuple]=None):

        assert len(x) == len(y)
        if normalize is not None:
            mu, sigma = normalize
            x = (np.array(x) - mu) / sigma

        self._data = np.array([(x[idx], y[idx]) for idx in range(len(x))], dtype=object)
        self._data = np.random.permutation(self._data)

        self._batch_size = batch_size
        self._cache = []
        self._cnt = 0

    def __len__(self):
        return (len(self._data) + self._batch_size - 1) // self._batch_size

    def sample_count(self):
        return len(self._data)

    def __iter__(self):
        return self

    def __next__(self):
        batch_x = []
        batch_y = []
        remaining = self.sample_count() - self._cnt
        idx = self._cnt // self._batch_size

        if remaining > 0:
            # Batch size of the current batch
            size = min(self._batch_size, remaining)
            if len(self._cache) > idx:
                result = self._cache[idx]
            else:
                for i in range(self._cnt, min(self._cnt + self._batch_size, self.sample_count())):
                    x, y = self._data[i]
                    batch_x.append(x)
                    batch_y.append(y)

                batch_x = torch.Tensor(np.array(batch_x))
                batch_y = torch.Tensor(np.array(batch_y)).long()
                result = (batch_x, batch_y)

                self._cache.append(result)
            self._cnt += size
            return result
        else:
            self._cnt = 0
            raise StopIteration
