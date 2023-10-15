import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=16):
        super().__init__()
        # encoder
        self.img_to_hidden = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, hidden_dim),
            nn.ReLU()
        )
        self.hidden_to_mean = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_img = nn.Sequential(
            nn.Linear(hidden_dim, 7 * 7 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = F.relu(self.img_to_hidden(x))
        mean = self.hidden_to_mean(h)
        logvar = self.hidden_to_logvar(h)
        stddev = torch.exp(0.5 * logvar)
        return mean, stddev

    def decode(self, z):
        h = F.relu(self.latent_to_hidden(z))
        return self.hidden_to_img(h)

    def forward(self, x):
        mean, stddev = self.encode(x)
        epsilon = torch.randn_like(stddev)
        z_reparametrized = mean + epsilon * stddev
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mean, stddev


if __name__ == "__main__":
    x = torch.randn(4, 28 * 28)  # 28*28
    vae = VariationalAutoEncoder(input_dim=28 * 28)
    x_reconstructed, mean, stddev = vae(x)
    print(x_reconstructed.shape)
    print(mean.shape)
    print(stddev.shape)
