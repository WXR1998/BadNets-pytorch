import logging

import torch

logging.basicConfig(level=logging.INFO)
import numpy as np

import dataset
import configs
import model
import util

def test(
        data: util.Dataset,
        epoch: int,
        lr: float=0.003,
        device: torch.device=torch.device('cpu'),
):
    if data == util.Dataset.MNIST:
        train_dataset, test_dataset = dataset.read_mnist(configs.data_path)
    elif data == util.Dataset.CIFAR:
        train_dataset, test_dataset = dataset.read_cifar(configs.data_path)
    else:
        logging.fatal('Dataset invalid.')
        return

    logging.info(f'Running on {str(data)} ...')

    train_x, train_y = train_dataset
    test_x, test_y = test_dataset
    mu = np.mean(train_x)
    sigma = np.std(train_x)

    logging.info(f'Poisoning...')
    train_x_p, train_y_p, train_idx_p, train_idx_c = \
        dataset.attack(train_x, train_y, 0.2, util.AttackType.PATTERN)
    test_x_p, test_y_p, test_idx_p, test_idx_c = \
        dataset.attack(test_x, test_y, 0.2, util.AttackType.PATTERN)

    # Original dataset
    logging.info(f'Processing dataloader...')
    train_loader = dataset.DataLoader(
        x=train_x,
        y=train_y,
        batch_size=configs.batch_size,
        normalize=(mu, sigma)
    )

    # Poisoned datasets
    logging.info(f'Processing poisoned dataloader...')
    train_loader_p = dataset.DataLoader(
        x=train_x_p,
        y=train_y_p,
        batch_size=configs.batch_size,
        normalize=(mu, sigma)
    )
    test_loader_c = dataset.DataLoader(
        x=test_x_p[test_idx_c],
        y=test_y_p[test_idx_c],
        batch_size=configs.batch_size,
        normalize=(mu, sigma)
    )
    test_loader_p = dataset.DataLoader(
        x=test_x_p[test_idx_p],
        y=test_y_p[test_idx_p],
        batch_size=configs.batch_size,
        normalize=(mu, sigma)
    )

    m_c = model.Model(
        epochs=epoch,
        lr=lr,
        model_type=data,
        device=device
    )
    m_p = model.Model(
        epochs=epoch,
        lr=lr,
        model_type=data,
        device=device
    )
    m_c.train(train_loader)
    m_p.train(train_loader_p)

    res_p = m_c.inference(test_loader_p)
    res_c = m_c.inference(test_loader_c)
    print('Model trained on original dataset')
    print(f'Acc on clean: {res_c:.3f}\nAcc on backdoor: {res_p:.3f}')
    res_p = m_p.inference(test_loader_p)
    res_c = m_p.inference(test_loader_c)
    print('Model trained on poisoned dataset')
    print(f'Acc on clean: {res_c:.3f}\nAcc on backdoor: {res_p:.3f}')

if __name__ == '__main__':
    test(data=util.Dataset.CIFAR,
         epoch=200,
         lr=0.001,
         device=torch.device('cuda:0'))
