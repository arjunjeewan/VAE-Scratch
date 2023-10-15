import torch
import torchvision.datasets as datasets
from tqdm.auto import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import os
import config

# Dataset
dataset = datasets.MNIST (root = "dataset/", train = True, transform = T.ToTensor(), download = True)
train_loader = DataLoader(dataset = dataset, batch_size = config.BATCH_SIZE, shuffle = True)
model = VariationalAutoEncoder(input_dim = config.INPUT_DIM, hidden_dim = config.H_DIM, latent_dim = config.Z_DIM).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE)
loss_fn = nn.MSELoss()


def train():

    training_loss = {
        'reconstruction_loss': [],
        'kl_divergence': [],
        'total_loss': []
    }
    outer_bar = tqdm(total=config.EPOCHS, position=0, desc="Training", postfix="Loss: N/A", disable=not config.PROGRESS_BAR, leave=False)
    inner_bar = tqdm(total=len(train_loader), position=1, leave=False, desc="", disable=not config.PROGRESS_BAR)


    # Training
    for epoch in range(config.EPOCHS):
        epoch_reconstruction_loss = []
        epoch_kl_divergence = []
        epoch_total_loss = []

        outer_bar.set_description(f"Epoch {epoch}")
        inner_bar.reset()
        inner_bar.refresh()
        inner_bar.set_description_str(f"Epoch: {epoch+1}")

        for x, _ in train_loader:
            # Forward Pass
            model.train()
            x = x.to(config.DEVICE)
            x_reconstructed, mean, stddev = model(x)

            # Loss
            reconstruction_loss = loss_fn(x_reconstructed.view(-1), x.view(-1))
            kl_divergence = torch.mean(-0.5 * torch.sum(1 + 2 * stddev.log() - mean ** 2 - stddev.square(), dim = 1), dim = 0)

            # Backpropagation
            loss = reconstruction_loss + 0.00025 * kl_divergence # Play around with 0.00025. try increasing/decreasing
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_reconstruction_loss.append(reconstruction_loss.detach())
            epoch_kl_divergence.append(kl_divergence.detach())
            epoch_total_loss.append(loss.detach())
            inner_bar.set_postfix(loss = loss.item())
            inner_bar.update(1)


        epoch_loss = {
            'reconstruction_loss': torch.stack(epoch_reconstruction_loss).mean().item(),
            'kl_divergence': torch.stack(epoch_kl_divergence).mean().item(),
            'total_loss': torch.stack(epoch_total_loss).mean().item()
        }
        for key, value in epoch_loss.items():
            training_loss[key].append(value)

        outer_bar.set_postfix_str(f"Loss: {epoch_loss['total_loss']:.4f}")
        outer_bar.update(1)

        if (epoch + 1) % 10 == 0:
            torch.save(model, f"saved/epoch_{epoch+1}.pth")
            plot_metrics(training_loss)
            inference({}, class_label=f"train/epoch_{epoch+1}", num_examples=10)
            inner_bar.write(f"Loss: {epoch_loss['total_loss']:.4f}")


    inner_bar.close()

    return training_loss


def plot_metrics(training_loss):
    _, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (metric, values) in zip(axes, training_loss.items()):
        ax.plot(values, label=metric)
        ax.set_title(metric)
    plt.savefig("metrics.png")


@torch.no_grad()
def get_classwise_distribution():
    model.eval()
    class_distribution = {}

    for x, y in train_loader:
        x = x.to(config.DEVICE)
        mean, stddev = model.encode(x)

        for mean_i, stddev_i, y_i in zip(mean, stddev, y):
            y_i = y_i.item()
            if y_i not in class_distribution:
                class_distribution[y_i] = ([], [])
            class_distribution[y_i][0].append(mean_i)
            class_distribution[y_i][1].append(stddev_i)

    for class_label, (mean, stddev) in class_distribution.items():
        class_distribution[class_label] = (torch.stack(mean).mean(dim=0).cpu(), torch.stack(stddev).mean(dim=0).cpu())

    return class_distribution


@torch.no_grad()
def inference(classwise_distribution, class_label, num_examples):
    model.eval()
    if class_label is not None and class_label in classwise_distribution:
        mean, stddev = classwise_distribution[class_label]
    else:
        mean = torch.randn(num_examples, config.Z_DIM)
        stddev = torch.ones(num_examples, config.Z_DIM)
        if class_label is None:
            class_label = "random"
    epsilon = torch.randn_like(stddev)
    z = mean + stddev * epsilon
    out = model.decode(z.to(config.DEVICE))
    out = torch.round(out.view(-1, 1, 28, 28) * 255)
    
    save_image(out, f"generated/{class_label}.png")


if __name__ == "__main__":
    
    os.makedirs('generated/train', exist_ok=True)
    os.makedirs('saved', exist_ok=True)


    training_loss = train()
    plot_metrics(training_loss)
    classwise_distribution = get_classwise_distribution()

    for class_label in range(10):
        

        mean, std = classwise_distribution[class_label]
        z = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(1, config.Z_DIM)
        out = model.decode(z.to(config.DEVICE))
        plt.imshow(out.squeeze().cpu().detach().numpy(), cmap="gray")