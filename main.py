import logging
import torch
logging.basicConfig(level=logging.INFO)
import datetime
import numpy as np
from matplotlib import pyplot as plt

import dataset
import configs
import model
import util

def train(
        data: util.Dataset,
        epoch: int,
        attack_type: util.AttackType=util.AttackType.PATTERN,
        lr: float=0.003,
        device: torch.device=torch.device('cpu'),
        model_ckpt: str=None,
):
    """
    Train a backdoored and a baseline model on the given dataset.
    :param data:
    :param epoch:
    :param attack_type:
    :param lr:
    :param device:
    :param model_ckpt:
    :return:
    """

    if data == util.Dataset.MNIST:
        train_dataset, test_dataset, _ = dataset.read_mnist(configs.data_path)
    elif data == util.Dataset.CIFAR:
        train_dataset, test_dataset, _ = dataset.read_cifar(configs.data_path)
    else:
        logging.fatal('Dataset invalid.')
        return
    model_name = datetime.datetime.now().strftime(f'{data.name}_%m%d_%H%M%S')

    logging.info(f'Running on {str(data)} ...')

    train_x, train_y = train_dataset
    test_x, test_y = test_dataset
    mu = np.mean(train_x)
    sigma = np.std(train_x)

    logging.info(f'Poisoning...')
    train_x_p, train_y_p, train_idx_p, train_idx_c = \
        dataset.attack(train_x, train_y, 0.2, attack_type)
    test_x_p, test_y_p, test_idx_p, test_idx_c = \
        dataset.attack(test_x, test_y, 0.2, attack_type)

    # Original dataset
    logging.info(f'Processing dataloader...')
    train_loader = dataset.DataLoader(
        x=train_x,
        y=train_y,
        batch_size=configs.batch_size,
        normalize=(mu, sigma),
        shuffle=True
    )

    # Poisoned datasets
    logging.info(f'Processing poisoned dataloader...')
    train_loader_p = dataset.DataLoader(
        x=train_x_p,
        y=train_y_p,
        batch_size=configs.batch_size,
        normalize=(mu, sigma),
        shuffle=True
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
    if model_ckpt is not None:
        m_c.load(f'{model_ckpt}_c')
        m_p.load(f'{model_ckpt}_p')
    else:
        m_c.train(train_loader)
        m_p.train(train_loader_p)
        m_c.dump(f'{model_name}_c')
        m_p.dump(f'{model_name}_p')

    res_p, _ = m_c.inference(test_loader_p)
    res_c, _ = m_c.inference(test_loader_c)
    print('Model trained on original dataset')
    print(f'Acc on clean: {res_c:.3f}\nAcc on backdoor: {res_p:.3f}')
    res_p, _ = m_p.inference(test_loader_p)
    res_c, _ = m_p.inference(test_loader_c)
    print('Model trained on poisoned dataset')
    print(f'Acc on clean: {res_c:.3f}\nAcc on backdoor: {res_p:.3f}')

def inspect(
        data: util.Dataset,
        model_ckpt: str,
        num_samples: int=5,
        attack_type: util.AttackType=util.AttackType.PATTERN,
        device: torch.device=torch.device('cpu'),
):
    """
    Randomly choose some figures in the testset, and inference them on the baseline model and backdoored model.
    :param data:
    :param model_ckpt:
    :param num_samples:
    :param attack_type:
    :param device:
    :return:
    """

    if data == util.Dataset.MNIST:
        train_dataset, test_dataset, classes = dataset.read_mnist(configs.data_path)
    elif data == util.Dataset.CIFAR:
        train_dataset, test_dataset, classes = dataset.read_cifar(configs.data_path)
    else:
        logging.fatal('Dataset invalid.')
        return

    logging.info(f'Inspecting on {str(data)} ...')

    train_x, train_y = train_dataset
    test_x, test_y = test_dataset
    mu = np.mean(train_x)
    sigma = np.std(train_x)

    idxs = np.random.choice(np.arange(len(test_x)), num_samples)
    sample_x_c = test_x[idxs]
    sample_y_c = test_y[idxs]

    sample_x_p = []
    for i in range(num_samples):
        sample_x_p.append(np.expand_dims(dataset.attack_one_image(sample_x_c[i], attack_type), axis=0))
    sample_x_p = np.concatenate(sample_x_p)

    sample_c_loader = dataset.DataLoader(
        x=sample_x_c,
        y=sample_y_c,
        batch_size=configs.batch_size,
        normalize=(mu, sigma)
    )
    sample_p_loader = dataset.DataLoader(
        x=sample_x_p,
        y=sample_y_c,
        batch_size=configs.batch_size,
        normalize=(mu, sigma)
    )

    m_c = model.Model(
        model_type=data,
        device=device
    )
    m_p = model.Model(
        model_type=data,
        device=device
    )
    m_c.load(f'{model_ckpt}_c')
    m_p.load(f'{model_ckpt}_p')

    _, cc_pred = m_c.inference(sample_c_loader)
    _, pc_pred = m_p.inference(sample_c_loader)
    _, cp_pred = m_c.inference(sample_p_loader)
    _, pp_pred = m_p.inference(sample_p_loader)

    plt.figure(figsize=(1.5 * num_samples, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        img = sample_x_c[i].transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(f'GT:    {classes[sample_y_c[i]]}\nBASE: {classes[cc_pred[i]]}\nBAD:  {classes[pc_pred[i]]}')
        plt.xticks([])
        plt.yticks([])

    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1 + num_samples)
        img = sample_x_p[i].transpose(1, 2, 0)
        plt.imshow(img)
        plt.title(f'BASE: {classes[cp_pred[i]]}\nBAD:  {classes[pp_pred[i]]}')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda:0')

    train(
        data=util.Dataset.MNIST,
        epoch=50,
        attack_type=util.AttackType.PATTERN,
        lr=0.001,
        device=device
    )

    train(
        data=util.Dataset.CIFAR,
        epoch=200,
        attack_type=util.AttackType.PATTERN,
        lr=0.001,
        device=device
    )

    # inspect(
    #     data=util.Dataset.MNIST,
    #     attack_type=util.AttackType.PATTERN,
    #     num_samples=10,
    #     model_ckpt='MNIST_1110_225542',
    #     device=device
    # )
    #
    # inspect(
    #     data=util.Dataset.CIFAR,
    #     attack_type=util.AttackType.PATTERN,
    #     num_samples=10,
    #     model_ckpt='CIFAR_1110_234320',
    #     device=device
    # )
