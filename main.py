import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

Nv_list = [90, 80, 70, 60]
noise_level = 0.05

# Load training and test data
u_traine = np.load('/Users/goat/project-6/data/u_training_data.npy')
u_teste = np.load('/Users/goat/project-6/data/u_test_data.npy')
f_traine = np.load('/Users/goat/project-6/data/f_train.npy')
f_teste = np.load('/Users/goat/project-6/data/f_test.npy')

# Get training/test set
f_train = f_traine[0:1500,0::Nskip]
f_test  = f_teste[1500:,0::Nskip]

u_train = u_traine[0:1500,0::Nskip]
u_test  = u_teste[1500:,0::Nskip]

f_train_tensor = torch.tensor(f_train, dtype=torch.float32)
u_train_tensor = torch.tensor(u_train, dtype=torch.float32)


dataset = TensorDataset(u_train_tensor, f_train_tensor)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

for Nv in Nv_list:
    print(f"\n==== Training with Nv = {Nv} observed nodes ====")
    u_train_noisy = u_train[:, :Nv] + noise_level * np.random.randn(*u_train[:, :Nv].shape)
    u_test_noisy = u_test[:, :Nv] + noise_level * np.random.randn(*u_test[:, :Nv].shape)

    # Clean target: the rest of the solution (x)
    x_train = u_train[:, Nv:]  
    x_test = u_test[:, Nv:] 

    # Hyperparameters
    y_dim = Nv
    x_dim = x_train.shape[1]
    batch_size = 128
    z_dim = 10
    num_epochs = 200
    lambd = 20

    # Convert to torch tensors
    Y_train = torch.tensor(u_train_noisy, dtype=torch.float32)
    X_train = torch.tensor(x_train, dtype=torch.float32)
    Y_test = torch.tensor(u_test_noisy, dtype=torch.float32)
    X_test = torch.tensor(x_test, dtype=torch.float32)

    train_dataset = TensorDataset(Y_train, X_train)
    test_dataset = TensorDataset(Y_test, X_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Custom sinusoidal activation
    class Sin(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    # Generator
    class Generator(nn.Module):
        def __init__(self, y_dim, z_dim, x_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(y_dim + z_dim, 128),
                Sin(),
                nn.Linear(128, 128),
                Sin(),
                nn.Linear(128, 64),
                Sin(),
                nn.Linear(64, x_dim)
            )
        def forward(self, y, z):    
            input = torch.cat([y, z], dim=1)
            return self.net(input)

    # Discriminator
    class Discriminator(nn.Module):
        def __init__(self, y_dim, x_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(y_dim + x_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        def forward(self, y, x):
            input = torch.cat([y, x], dim=1)
            return self.net(input)

    # Initialize models
    generator = Generator(y_dim, z_dim, x_dim).to(device)
    discriminator = Discriminator(y_dim, x_dim).to(device)

    print("Trainable parameters in Generator:", sum(p.numel() for p in generator.parameters() if p.requires_grad))
    print("Trainable parameters in Discriminator:", sum(p.numel() for p in discriminator.parameters() if p.requires_grad))

    # Optimizers and scheduler
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-3)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=5e-4)
    scheduler_G = StepLR(optimizer_G, step_size=50, gamma=0.95)
    scheduler_D = StepLR(optimizer_D, step_size=50, gamma=0.4)
    criterion = nn.MSELoss()

    # ========== Training Loop ==========
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        for U_batch, F_batch in train_loader:
            U_batch = U_batch.to(device)
            F_batch = F_batch.to(device)
            batch_size_cur = U_batch.shape[0]

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size_cur, 1, device=device)
            fake_labels = torch.zeros(batch_size_cur, 1, device=device)

            d_real = discriminator(U_batch, F_batch)
            loss_d_real = criterion(d_real, real_labels)

            z = torch.randn(batch_size_cur, z_dim, device=device)
            F_fake = generator(U_batch, z)
            d_fake = discriminator(U_batch, F_fake.detach())
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_d = 0.5 * (loss_d_fake + loss_d_real)
            loss_d.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            d_fake = discriminator(U_batch, F_fake)
            loss_g = 0.5 * torch.mean((d_fake - 1)**2)
            loss_g += lambd * F.mse_loss(F_fake, F_batch)
            loss_g.backward()
            optimizer_G.step()

        scheduler_G.step()
        scheduler_D.step()

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f"[Nv={Nv}] Epoch {epoch}: D loss = {loss_d.item():.4f}, G loss = {loss_g.item():.4f}")
            lambd = min(lambd * 1.5, 100)

    # ========== Inference ==========
    generator.eval()
    Y_test = Y_test.to(device)
    X_test = X_test.to(device)
    num_samples = 100

    for i in range(4):
        y_i = Y_test[i].unsqueeze(0).repeat(num_samples, 1)
        z = torch.randn(num_samples, z_dim, device=device)
        with torch.no_grad():
            x_hat_samples = generator(y_i, z)

        mean = x_hat_samples.mean(dim=0).cpu().numpy()
        std = x_hat_samples.std(dim=0).cpu().numpy()
        true = X_test[i].cpu().numpy()

        Nx = u_train.shape[1]
        x_coords = np.linspace(0, 1, Nx)[Nv:]

        plt.figure(figsize=(8, 4))
        plt.plot(x_coords, true, label='True', color='black')
        plt.plot(x_coords, mean, label='Predicted Mean', color='blue')
        plt.fill_between(x_coords, mean - std, mean + std, color='blue', alpha=0.3, label='±1 SD')
        plt.title(f"Nv = {Nv}, Test Sample {i}")
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
