import torch
import datetime

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from tensorboard_logger import configure, log_value

from utils.helpers import save_checkpoint
from constants import ROOT_DIR

class MNISTTrainer:
    def __init__(self, dataset, generator, discriminator, args, device):
        self.data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        self.generator = generator
        self.discriminator = discriminator
        self.bce_loss = nn.BCELoss()
        self.args = args
        self.device = device

        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=args.lr)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=args.lr)

        now = datetime.datetime.now().strftime("%H-%M-%S_%Y-%m-%d")

        description = "vanilla_GAN_MNIST-" + now
        configure('./stats/' + description, flush_secs=2)

    def train_discriminator(self, real_data, fake_data):
        b_size = real_data.size(0)
        self.d_optimizer.zero_grad()

        # train on real data
        real_logits, real_pred = self.discriminator(real_data)
        ones_target = torch.ones(b_size, 1).to(self.device)
        err_real = self.bce_loss(real_pred, ones_target)

        # train on fake data
        fake_logits, fake_pred = self.discriminator(fake_data)
        zeros_target = torch.zeros(b_size, 1).to(self.device)
        err_fake = self.bce_loss(fake_pred, zeros_target)

        total_err = err_real + err_fake

        total_err.backward()

        self.d_optimizer.step()

        return total_err, err_real, err_fake, real_pred, fake_pred

    def train_generator(self, gen_data):
        b_size = gen_data.size(0)

        self.g_optimizer.zero_grad()

        # D(G(z))
        d_logits, prediction = self.discriminator(gen_data)
        ones_target = torch.ones(b_size, 1).to(self.device)
        err = self.bce_loss(prediction, ones_target)

        err.backward()

        self.g_optimizer.step()

        return err

    def train(self):
        total_it = 0
        for epoch in range(self.args.epochs):
            d_total_err = 0
            g_total_err = 0
            pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
            pbar.set_description('ep: 0, TL: 0, DL: 0, GL: 0')
            for i, (data, label) in pbar:

                total_it += 1
                data = data.to(self.device)
                # first train discriminator
                real_data = data
                latent_vec = torch.randn(self.args.batch_size, self.args.latent_dim).to(self.device)
                fake_logits, fake_data = self.generator(latent_vec)
                fake_data = fake_data.detach()

                d_err, dr_err, df_err, real_pred, fake_pred = self.train_discriminator(real_data, fake_data)
                d_total_err += d_err.data / len(data)

                # Now train generator
                latent_vec = torch.randn(self.args.batch_size, self.args.latent_dim).to(self.device)
                gen_logit, gen_data = self.generator(latent_vec)

                g_err = self.train_generator(gen_data)
                g_total_err += g_err.data / len(data)

                total_err = d_err.data + g_err.data
                pbar.set_description('ep: %d, TL: %.4f, DL: %.4f, GL: %.4f' % (epoch, total_err,
                                                                                d_err.data,
                                                                                g_err.data))
                if i % self.args.log_every == 0:
                    avg_d_err = d_total_err / self.args.log_every
                    d_total_err = 0

                    avg_g_err = g_total_err / self.args.log_every
                    g_total_err = 0

                    log_value("discriminator_loss", avg_d_err, total_it)
                    log_value("generator_loss", avg_g_err, total_it)

                if i % self.args.save_every == 0:
                    save_checkpoint({
                        'epoch': epoch,
                        'generator_dict': self.generator.state_dict(),
                        'discriminator_dict': self.discriminator.state_dict(),
                        'g_optimizer': self.g_optimizer.state_dict(),
                        'd_optimizer': self.d_optimizer.state_dict()
                    }, False,
                        filename=self.args.resume_path)
            else:
                gen_images = gen_data.view(*data.shape)
                save_image(gen_images[:20], ROOT_DIR + '/images/%d.png' % epoch, nrow=4, normalize=True)





