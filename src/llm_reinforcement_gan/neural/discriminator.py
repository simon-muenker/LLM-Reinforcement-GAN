import torch


class Discriminator(torch.nn.Module):
    def __init__(self, size: int, num_layers: int = 1):
        super().__init__()

        self.gru = torch.nn.GRU(size, size, num_layers, batch_first=True, dtype=torch.bfloat16)
        self.linear = torch.nn.Linear(size, 1, dtype=torch.bfloat16)
        self.activation = torch.nn.Sigmoid()

        self.to("cuda")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.gru(batch)[0]
        output = output[:, -1, :]
        output = self.linear(output)
        output = self.activation(output)
        output = output.squeeze()

        return output
