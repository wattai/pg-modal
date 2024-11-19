import modal.gpu
import torch
import torch.nn as nn
import torch.nn.functional as F
import modal


image = modal.Image.debian_slim().pip_install(
    # "accelerate",
    # "transformers",
    "torch",
    # "datasets",
    # "tensorboard",
)

app = modal.App(name="test-run-simple-attention-model", image=image)


class Attention(nn.Module):
    def __init__(self, key_dim, query_dim, value_dim):
        super(Attention, self).__init__()
        self.scale_factor = torch.sqrt(torch.tensor(key_dim, dtype=torch.float32))

    def forward(self, query, key, value):
        # Calculate the dot product between query and key
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale_factor

        # Apply the softmax to calculate the attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply the attention weights by the value
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class SimpleModel(nn.Module):
    def __init__(
        self, input_dim, key_dim, query_dim, value_dim, hidden_dim, output_dim
    ):
        super(SimpleModel, self).__init__()
        self.attention = Attention(key_dim, query_dim, value_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, query, key, value):
        # Calculate attention
        attention_output, attention_weights = self.attention(query, key, value)

        # Pass the attention output through a fully connected layer
        hidden = F.relu(self.fc1(attention_output))
        output = self.fc2(hidden)

        return output, attention_weights


def run(device="cpu"):
    # Define the dimensions
    input_dim = 10
    key_dim = 5
    query_dim = 5
    value_dim = 10
    hidden_dim = 20
    output_dim = 5

    # Create an instance of SimpleModel
    model = SimpleModel(
        input_dim, key_dim, query_dim, value_dim, hidden_dim, output_dim
    )

    # Generate some random tensors to simulate key, query, and value
    query = torch.randn(1, 3, query_dim)
    key = torch.randn(1, 3, key_dim)
    value = torch.randn(1, 3, value_dim)

    # Run the attention mechanism
    output, attention_weights = model.to(device)(
        query.to(device), key.to(device), value.to(device)
    )

    # Print the output and attention weights
    print("Output: ", output)
    print("Attention Weights: ", attention_weights)


@app.function(
    gpu=modal.gpu.A10G(count=1),
    timeout=7200,
)
def run_on_modal():
    run(device="cuda")


@app.local_entrypoint()
def main():
    run_on_modal.remote()


if __name__ == "__main__":
    run()
