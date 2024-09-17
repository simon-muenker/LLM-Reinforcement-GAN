import pytest
import rich
import torch

import llm_reinforcement_gan as rfgan

SIZE: int = 32
INPUT: torch.Tensor = torch.rand(4, 16, SIZE, dtype=torch.bfloat16, device="cuda:0")


class TestDiscriminator:
    @pytest.fixture
    def discriminator(self) -> rfgan.neural.Discriminator:
        return rfgan.neural.Discriminator(input_size=SIZE)

    def test__forward(self, discriminator: rfgan.neural.Discriminator):
        prediction: torch.Tensor = discriminator(INPUT)

        assert INPUT.size()[0] == prediction.size()[0]

        rich.print(INPUT.size())
        rich.print(prediction.size())
