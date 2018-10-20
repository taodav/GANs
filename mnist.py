from torchvision import transforms, datasets

def mnist_data():
    compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return datasets.MNIST(root='./data', train=True, transform=compose, download=True)

