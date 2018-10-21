import torch
import shutil


def save_checkpoint(state, is_best,
                    filename='./checkpoints/model_checkpoint.pth.tar',
                    best_filename='./checkpoints/model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


def load_checkpoint(resume_path, generator, discriminator, d_optimizer=None, g_optimizer=None):
    print("Loading checkpoint")
    checkpoint = torch.load(resume_path)
    generator.load_state_dict(checkpoint['generator_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_dict'])
    epoch = checkpoint['epoch']

    if g_optimizer is not None and d_optimizer is not None:
        print('Loading optimizers')
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])

        return generator, discriminator, g_optimizer, d_optimizer, epoch
    return generator, discriminator, epoch
