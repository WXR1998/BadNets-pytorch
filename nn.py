import torch
import util

class Conv(torch.nn.Module):
    def __init__(self,
                 model_type: util.Dataset):
        super().__init__()
        self._model_type = model_type

        self._encoder_mnist = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
            torch.nn.Conv2d(16, 32, 5),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(2),
        )
        self._classifier_mnist = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )

        self._encoder_cifar = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2, stride=1, dilation=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(32, 32, kernel_size=(5, 5), padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
            torch.nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self._classifier_cifar = torch.nn.Sequential(
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor):
        if self._model_type == util.Dataset.MNIST:
            x = self._encoder_mnist(x)
            x = x.view(-1, 512)
            x = self._classifier_mnist(x)
        elif self._model_type == util.Dataset.CIFAR:
            x = self._encoder_cifar(x)
            x = x.view(-1, 1024)
            x = self._classifier_cifar(x)
        else:
            x = None

        return x
