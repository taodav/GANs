import torch

from argparse import ArgumentParser
from mnist import mnist_data
from train import MNISTTrainer
from models import Generator, Discriminator


def get_args():
    parser = ArgumentParser(description='HRED Model')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--resume_path', type=str, default='./checkpoints/model_checkpoint.pth.tar')

    return parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()
    dataset = mnist_data()
    image_shape = dataset.train_data[0].size()


    generator = Generator(args.latent_dim, image_shape).to(device)
    discriminator = Discriminator(image_shape).to(device)
    trainer = MNISTTrainer(dataset, generator, discriminator, args, device)

    trainer.train()

