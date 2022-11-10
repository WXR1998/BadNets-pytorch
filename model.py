import torch
import tqdm
import logging
import numpy as np
import os

import nn
import configs
import dataset
import util

class Model:
    def __init__(self,
                 epochs: int=0,
                 lr: float=0,
                 model_type: util.Dataset=util.Dataset.MNIST,
                 device: torch.device=torch.device('cpu')):
        self._model = nn.Conv(model_type).to(device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._epochs = epochs
        self._loss = torch.nn.CrossEntropyLoss()
        self._device = device

    def dump(self, name: str):
        state = self._model.state_dict()
        os.makedirs(configs.ckpt_path, exist_ok=True)
        torch.save(state, os.path.join(configs.ckpt_path, f'{name}.pt'))

    def load(self, name: str):
        os.makedirs(configs.ckpt_path, exist_ok=True)
        ckpt_path = os.path.join(configs.ckpt_path, f'{name}.pt')

        if not os.path.exists(ckpt_path):
            logging.error(f'Ckpt file "{ckpt_path}" does not exist.')
            exit(0)

        ckpt = torch.load(ckpt_path)
        self._model.load_state_dict(ckpt)

    def train(self, train_loader: dataset.DataLoader):
        logging.info('Start training...')

        for epoch in range(self._epochs):
            loss_sum = 0
            acc_sum = 0
            for batch_x, batch_y in tqdm.tqdm(train_loader):
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                self._optimizer.zero_grad()
                pred = self._model(batch_x)
                loss = self._loss(pred, batch_y)
                loss.backward()
                self._optimizer.step()

                pred_v = np.argmax(pred.detach().cpu().numpy(), axis=-1)
                batch_y_v = batch_y.detach().cpu().numpy()
                loss_sum += loss.cpu().item() * batch_x.shape[0]
                acc_sum += sum(pred_v == batch_y_v)

            sample_count = train_loader.sample_count()
            loss_avg = loss_sum / sample_count
            acc_avg = acc_sum / sample_count
            logging.info(f'Epoch {epoch + 1}/{self._epochs}:\tloss = {loss_avg:.3f}\tacc = {acc_avg:.3f}')

        logging.info('Finish training.')

    def inference(self, test_loader: dataset.DataLoader):
        """
        :param test_loader:
        :return: accuracy, raw output values
        """
        logging.info('Start inference...')
        preds = []

        with torch.no_grad():
            acc_sum = 0
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                pred = self._model(batch_x)
                pred_v = np.argmax(pred.detach().cpu().numpy(), axis=-1)
                batch_y_v = batch_y.detach().cpu().numpy()
                acc_sum += sum(pred_v == batch_y_v)

                preds.append(pred_v)

        logging.info('Finish inference.')

        sample_count = test_loader.sample_count()
        return acc_sum / sample_count, np.concatenate(preds)
