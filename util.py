import enum

class AttackType(enum.Enum):
    SINGLE_PIXEL = 1
    PATTERN = 2

class Dataset(enum.Enum):
    CIFAR = 1
    MNIST = 2
