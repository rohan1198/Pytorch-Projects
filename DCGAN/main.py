import os
import cv2
import argparse
import numpy as np
from imutils import build_montages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model import Generator, Discriminator



def init_weights(net):
    classname = net.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type = str, required = True, help = "Path to output directory")
    parser.add_argument("--epochs", type = int, default = 50, help = "Number of epochs to train")
    parser.add_argument("--batch-size", type = int, default = 128, help = "Batch size for training")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation = transforms.Compose([transforms.ToTensor(), 
                                       transforms.Normalize((0.5), (0.5))]
                                       )
    
    train_dataset = MNIST(root = "data", train = True, download = True, transform = augmentation)
    test_dataset = MNIST(root = "data", train = False, download = True, transform = augmentation)
    data = torch.utils.data.ConcatDataset((train_dataset, test_dataset))

    dataloader = DataLoader(data, shuffle = True, batch_size = args.batch_size)

    steps = len(dataloader.dataset) // args.batch_size

    gen = Generator(input_dim = 100, output_channels = 1)
    gen.apply(init_weights).to(device)

    disc = Discriminator(depth = 1)
    disc.apply(init_weights).to(device)

    generator_optim = optim.Adam(gen.parameters(), lr = 0.0002, betas = (0.5, 0.999), weight_decay = 0.0002 / args.epochs)
    discriminator_optim = optim.Adam(disc.parameters(), lr = 0.0002, betas = (0.5, 0.999),  weight_decay = 0.0002 / args.epochs)

    criterion = nn.BCELoss()

    val_input = torch.randn(256, 100, 1, 1, device=device)

    real_labels = 1
    fake_labels = 0

    for epoch in range(args.epochs):
        print(f"epoch {epoch + 1}/{args.epochs}")

        gen_loss = 0
        disc_loss = 0

        for x in dataloader:
            disc.zero_grad()

            images = x[0]
            images = images.to(device)

            bs =  images.size(0)
            labels = torch.full((bs,), real_labels, dtype=torch.float, device=device)
            output = disc(images).view(-1)

            loss_real = criterion(output, labels)
            loss_real.backward()

            noise = torch.randn(bs, 100, 1, 1, device=device)

            fake = gen(noise)
            labels.fill_(fake_labels)
            output = disc(fake.detach()).view(-1)

            loss_fake = criterion(output, labels)
            loss_fake.backward()

            loss_disc = loss_real + loss_fake
            discriminator_optim.step()

            gen.zero_grad()

            labels.fill_(real_labels)
            output = disc(fake).view(-1)

            loss_gen = criterion(output, labels)
            loss_gen.backward()

            generator_optim.step()

            disc_loss += loss_disc
            gen_loss += loss_gen
    
        print(f"Generator Loss: {(gen_loss / steps):.4f} | Discriminator Loss: {(disc_loss / steps):.4f}")

        if (epoch + 1) % 1 == 0:
            gen.eval()

            images = gen(val_input)
            images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = np.repeat(images, 3, axis=-1)

            vis = build_montages(images, (28, 28), (16, 16))[0]

            p = os.path.join(args.output, f"epoch_{str(epoch + 1).zfill(4)}.png")

            cv2.imwrite(p, vis)
            gen.train()