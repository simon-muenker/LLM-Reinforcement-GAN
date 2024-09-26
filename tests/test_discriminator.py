import pytest
import rich
import torch

import llm_reinforcement_gan as rfgan

SIZE: int = 32


class TestDiscriminator:
    @pytest.fixture
    def input(self, device:str) -> torch.Tensor:
        return torch.rand(4, 16, SIZE, dtype=torch.bfloat16, device=torch.device(device))
    
    @pytest.fixture
    def discriminator(self) -> rfgan.neural.Discriminator:
        return rfgan.neural.Discriminator(input_size=SIZE)

    def test__forward(self, discriminator: rfgan.neural.Discriminator, input: torch.Tensor, device:str):
        prediction: torch.Tensor = discriminator(input).to(torch.device(device))

        assert input.size()[0] == prediction.size()[0]

        rich.print(input.size())
        rich.print(prediction.size())
