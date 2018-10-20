from torchvision import datasets
from mnist import mnist_data
from train import MNISTTrainer

if __name__ == "__main__":
    dataset = mnist_data()
    trainer = MNISTTrainer(dataset)
