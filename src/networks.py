import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits

# what is the right way to perform unit tests?
if __name__ == '__main__':

    num_features = 4
    num_actions = 2
    batch_size = 20

    print(f"Testing PolicyMLP...")
    policy_mlp = PolicyMLP(num_features, 8, num_actions)
    X_batch = torch.rand(batch_size, num_features)
    X_sample = torch.rand(num_features)
    policy_mlp.eval()
    with torch.no_grad():
        logits_batch = policy_mlp(X_batch)
        logits_sample = policy_mlp(X_sample)
    assert logits_batch.shape == (batch_size, num_actions), "Output shape for batch input is incorrect"
    assert logits_sample.shape == (num_actions,), "Output shape for single-sample input is incorrect"
    print(f"PolicyMLP passed the test")
