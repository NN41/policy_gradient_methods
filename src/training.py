#%%

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dataloaders_for_value_network(batch_observations, batch_future_returns):
    X = torch.tensor(batch_observations, dtype=torch.float32)
    y = torch.tensor(batch_future_returns, dtype=torch.float32)
    full_dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader

def test(model, loss, test_dataloader):
    loss_total = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            n_samples += len(y)
            loss_total += loss(y_pred, y).item() * len(y)
    avg_loss = loss_total / n_samples
    return avg_loss

def train(model, loss, optimizer, train_dataloader, n_updates=0):

    n_batches = len(train_dataloader)
    total_loss_update = 0
    total_loss = 0
    n_samples_update = 0
    n_samples = 0

    if n_updates == -1:
        n_updates = n_batches
    update_batches = np.linspace(-1, n_batches-1, min(n_batches,n_updates)+1, dtype=int)[1:]
    avg_loss_updates = []

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X).squeeze()
        batch_loss = loss(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        n_samples += len(y)
        total_loss += batch_loss.item() * len(y)

        n_samples_update += len(y)
        total_loss_update += batch_loss.item() * len(y)
        if batch in update_batches:
            avg_loss_update = total_loss_update / n_samples_update
            avg_loss_updates.append(avg_loss_update)
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_update:.5f}")
            n_samples_update = 0
            total_loss_update = 0
    
    avg_loss = total_loss / n_samples
    return avg_loss, (avg_loss_updates, update_batches.tolist())

def train_value_network(model, loss, optimizer, train_dataloader, test_dataloader, n_epochs, n_updates=0):

    test_losses = []
    train_losses = []

    if n_updates != 0:
        test_loss = test(model, loss, test_dataloader)
        print(f"\t\t(Value Net) Epoch {0} / {n_epochs} | train loss = NaN    | test loss = {test_loss:.5f}")

    if n_updates == -1:
        n_updates = n_epochs
    update_epochs = np.linspace(0, n_epochs-1, min(n_epochs,n_updates), dtype=int)

    for epoch in range(n_epochs):
        train_loss, _ = train(model, loss, optimizer, train_dataloader, n_updates=0)
        train_losses.append(train_loss)
        if epoch in update_epochs:
            test_loss = test(model, loss, test_dataloader)
            test_losses.append(test_loss)
            print(f"\t\t(Value Net) Epoch {epoch+1} / {n_epochs} | train loss = {train_loss:.5f} | test loss = {test_loss:.5f}")

    test_loss_info = (test_losses, update_epochs.tolist())
    train_loss_info = (train_losses, range(n_epochs))
    return test_loss_info, train_loss_info

# %%

if __name__ == '__main__':
    
    from src.networks import ValueMLP

    print(f"Using {device} device")

    batch_size = 500
    model = ValueMLP(4,2,1).to(device)
    batch_features = np.random.rand(batch_size,4)
    batch_targets = np.random.rand(batch_size,)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_features, batch_targets)

    assert model(torch.rand(100,4).to(device)).shape == (100,1), "Failed output shape test"
    assert isinstance(test(model, loss, test_dataloader), float), 'Failed "test" function test'
    train(model, loss, optimizer, train_dataloader, n_updates=-1)
    train_value_network(model, loss, optimizer, train_dataloader, test_dataloader, n_epochs=7, n_updates=-1);

    