import typing

import pydantic
import torch


class DiscriminatorArgs(pydantic.BaseModel):
    encoder_num_layers: int = 8
    encoder_layers_config: typing.Dict = dict(nhead=4, batch_first=True, dtype=torch.bfloat16)

    lstm_config: typing.Dict = dict(num_layers=4, batch_first=True, dtype=torch.bfloat16)

    linear_config: typing.Dict = dict(dtype=torch.bfloat16)


class Discriminator(torch.nn.Module):
    def __init__(self, input_size: int, args: DiscriminatorArgs = DiscriminatorArgs()):
        super().__init__()
        self.args: DiscriminatorArgs = args

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=input_size, **self.args.encoder_layers_config
            ),
            num_layers=self.args.encoder_num_layers,
        )
        self.lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=input_size, **self.args.lstm_config
        )
        self.linear = torch.nn.Linear(input_size, 1, **self.args.linear_config)
        self.activation = torch.nn.Sigmoid()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.encoder(batch)
        _, (output, _) = self.lstm(output)
        output = self.linear(output)
        output = self.activation(output)

        return output
