import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# Load data
f_train = np.load('/Users/goat/project-6/data/f_train.npy')  # shape: (2000, Nx)
f_test = np.load('/Users/goat/project-6/data/f_test.npy')    # shape: (100, Nx)
u_train = np.load('/Users/goat/project-6/data/u_training_data.npy')  # shape: (2000, Nx)
u_test = np.load('/Users/goat/project-6/data/u_test_data.npy')    # shape: (100, Nx)

Nx = u_train.shape[1]
z_dim = 30
num_epochs = 1000
batch_size = 128
lambd = 10
noise_level = 0.05
Nv_list = [90, 80,70,60,50]  # Occlusion levels to try
critic_num = 2  # Number of critic updates per generator update

class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

# Define Generator
class Generator(nn.Module):
    def __init__(self, y_dim, z_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim + z_dim, 128),
            Sin(),
            nn.Linear(128, 128),
            Sin(),
            nn.Linear(128, x_dim)
        )
    def forward(self, y, z):
        input = torch.cat([y, z], dim=1)
        return self.net(input)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, y_dim, x_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(y_dim + x_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, y, x):
        input = torch.cat([y, x], dim=1)
        return self.net(input)

for Nv in Nv_list:
    print(f"\n--- Running for Nv = {Nv} ---")
    x_dim = Nx - Nv  # Remaining dimensions after occlusion
    y_dim = Nv + Nx 

    # Add noise to left-side input
    u_train_noisy = u_train[:, :Nv] + noise_level * np.random.randn(*u_train[:, :Nv].shape)
    u_test_noisy = u_test[:, :Nv] + noise_level * np.random.randn(*u_test[:, :Nv].shape)

    # Build input: [Y (noisy u), f]
    Y_train = np.concatenate([u_train_noisy, f_train], axis=1)
    Y_test = np.concatenate([u_test_noisy, f_test], axis=1)

    X_train = u_train[:, Nv:]
    X_test = u_test[:, Nv:]

    train_dataset = TensorDataset(
        torch.tensor(Y_train, dtype=torch.float32),
        torch.tensor(X_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(Y_test, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    G = Generator(y_dim, z_dim , x_dim).to(device)
    D = Discriminator(y_dim, x_dim).to(device)

    optimizer_G = Adam(G.parameters(), lr=1e-3)
    optimizer_D = Adam(D.parameters(), lr=5e-4)
    sch_G = StepLR(optimizer_G, step_size=50, gamma=0.95)
    sch_D = StepLR(optimizer_D, step_size=50, gamma=0.95)

    # Loss
    criterion = nn.MSELoss()

    # Training
    for epoch in range(num_epochs):
        G.train()
        D.train()
        for Y_batch, X_batch in train_loader:
            Y_batch = Y_batch.to(device)
            X_batch = X_batch.to(device)
            batch_size_cur = Y_batch.shape[0]

            # Train Discriminator #--------------------
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size_cur, 1, device=device)
            fake_labels = torch.zeros(batch_size_cur, 1, device=device)

            d_real = D(Y_batch, X_batch)
            loss_d_real = criterion(d_real, real_labels)

            z = torch.randn(batch_size_cur, z_dim, device=device)

            X_fake = G(Y_batch, z)
            d_fake = D(Y_batch, X_fake.detach())
            loss_d_fake = criterion(d_fake, fake_labels)

            # Calculate loss
            loss_d = .5* (loss_d_fake + loss_d_real)
            loss_d.backward()
            optimizer_D.step()

            # Train Generator
            #----------------
            if epoch % critic_num == 0:
                optimizer_G.zero_grad()
                d_fake = D(Y_batch, X_fake)
                loss_g =0.5 * torch.mean((d_fake - 1)**2)
                loss_g += lambd * F.mse_loss(X_fake, X_batch) # penalize generator
                loss_g.backward()
                optimizer_G.step()

        # LR scheduler update
        sch_G.step()
        sch_D.step()

        if epoch % 100 == 0 :
            print(f"Epoch {epoch}: D loss {loss_d.item():.4f}, G loss {loss_g.item():.4f}")
            lambd = min(lambd*1.5, 100)
    
    # Get test samples
    G.eval()
    num_samples = 100  # generate 100 samples
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with torch.no_grad(): # No gradient tracking needed for testing
        for y_batch, x_batch in test_loader:
            y_batch = y_batch.to(device)
            batch_size = y_batch.shape[0]

            # store all X predictions
            x_gen = []

            #Generate 100 Xs given same U
            for _ in range(num_samples):
                z = torch.randn(batch_size, z_dim).to(device)
                x_pred = G(y_batch, z)
                x_gen.append(x_pred.cpu().numpy())

            x_preds = np.stack(x_gen, axis=0)

         # Compute mean and std dev of generated Fs
            x_mean = np.mean(x_preds, axis=0)
            x_std  = np.std(x_preds, axis=0)
            y_batch = y_batch.cpu().numpy()
            x_batch = x_batch.cpu().numpy()
            break

    #Plot
    num_to_plot = 4
    x = np.linspace(0, 1, x_batch.shape[1])
    plt.figure(figsize=(18, 8))
    for i in range(num_to_plot):
        print(f'Sample {i+1}:', np.linalg.norm(x_batch[i] - x_mean[i]))
        plt.subplot(2, 3, i + 1)
        plt.plot(x, x_batch[i], label='True X', color='black')
        plt.plot(x, x_mean[i], label='Inferred field X')
        plt.fill_between(x, x_mean[i] - x_std[i], x_mean[i] + x_std[i], color='grey', alpha=0.5, label='±1 SD interval')
        plt.title(f'Sample {i+1}')
        plt.suptitle("Generated X Given noisy Y vs True X", fontsize=24)
        plt.xlabel('x')
        plt.ylabel('X')
        plt.legend()
        plt.tight_layout()
    plt.show()