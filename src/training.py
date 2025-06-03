import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dataloaders_for_value_network(batch_observations, batch_rewards_to_go):
    X = torch.tensor(batch_observations, dtype=torch.float32)
    y = torch.tensor(batch_rewards_to_go, dtype=torch.float32)
    full_dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader

def test(model, criterion, test_dataloader):
    loss_total = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            n_samples += len(y)
            loss_total += criterion(y_pred, y).item() * len(y)
    avg_loss = loss_total / n_samples
    return avg_loss, 0

def train(model, criterion, optimizer, train_dataloader, n_updates):

    n_batches = len(train_dataloader)
    # n_updates = 3
    update_batches = np.linspace(0, n_batches-1, n_updates).astype(int)
    total_loss_between_updates = 0
    total_loss = 0
    n_samples_between_updates = 0
    n_samples = 0

    # if n_updates > 0:
    #     print(f"[batch] / {n_batches} | [avg train loss between updates]")
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        n_samples_between_updates += len(y)
        n_samples += len(y)
        total_loss += loss.item() * len(y)
        total_loss_between_updates += loss.item() * len(y)
        if batch in update_batches:
            avg_loss_between_updates = total_loss_between_updates / n_samples_between_updates
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_between_updates:.5f}")
            n_samples_between_updates = 0
            total_loss_between_updates = 0
    
    avg_train_loss = total_loss / n_samples
    return avg_train_loss

def train_value_function_network(model, criterion, optimizer, train_dataloader, test_dataloader, EPOCHS, verbose=False):


    test_loss, test_acc = test(model, criterion, test_dataloader)
    print(f"\tepoch {0} / {EPOCHS} | avg test loss = {test_loss:.5f}")

    # logs = []
    # logs_flattened = []
    # initial_log = {
    #     "epoch": 0,
    #     "test_loss": test_loss,
    #     "test_acc": test_acc,
    #     "train_loss": None,
    #     "": get_all_param_metrics(model),
    # }
    # logs.append(initial_log)
    # logs_flattened.append(flatten_nested_dict(initial_log, separator='/'))

    for epoch in range(EPOCHS):

        train_loss = train(model, criterion, optimizer, train_dataloader, n_updates=0)

        if epoch % 1 == 0:
            test_loss, test_acc = test(model, criterion, test_dataloader)
            print(f"\tepoch {epoch+1} / {EPOCHS} | avg test loss = {test_loss:.5f}")
            if verbose:
                print(f"\t\ttrain loss {train_loss}")

        # log = {
        #     "epoch": epoch+1,
        #     "test_loss": test_loss,
        #     "test_acc": test_acc,
        #     "train_loss": train_loss,
        #     "": get_all_param_metrics(model),
        # }
        # logs.append(log)
        # logs_flattened.append(flatten_nested_dict(log, separator='/'))


if __name__ == '__main__':
    
    
    print(f"Using {device} device")

    # pass

