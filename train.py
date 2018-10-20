from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

class MNISTTrainer:
    def __init__(self, dataset, generator, discriminator):

        self.data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        self.generator = generator
        self.discriminator = discriminator
