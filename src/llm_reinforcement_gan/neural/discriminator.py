import torch


class Discriminator(torch.nn.Module):
    def __init__(self, size: int, num_layers: int = 4, num_heads: int = 4):
        super().__init__()

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=size, nhead=num_heads, batch_first=True, dtype=torch.bfloat16
            ),
            num_layers=num_layers,
        )
        self.linear = torch.nn.Linear(size, 1, dtype=torch.bfloat16)
        self.activation = torch.nn.Sigmoid()

        self.to("cuda")

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.encoder(batch)
        output = torch.mean(output, dim=1)
        output = self.linear(output)
        output = self.activation(output)
        output = output.squeeze()

        return output
