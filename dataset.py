import torchvision
import torch.utils.data as Data


# Load the dataset
def load_train_data(batch_size, download):
    train_data = torchvision.datasets.MNIST(
        root='./dataset/mnist/',
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=download,  # download it if you don't have it
    )
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return train_loader


def load_test_data(batch_size, download):
    test_data = torchvision.datasets.MNIST(
        root='./dataset/mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=download,
    )
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return test_loader

