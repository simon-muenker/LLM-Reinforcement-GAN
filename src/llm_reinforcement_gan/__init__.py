import pydantic
import torch

from llm_reinforcement_gan import neural
from llm_reinforcement_gan.dataset import Dataset


class Loss(torch.nn.Module):
    pass


class Trainer(pydantic.BaseModel):
    pass


class Pipeline(pydantic.BaseModel):
    dataset: Dataset

    generator: neural.Generator
    discriminator: neural.Discriminator

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __call__(self):
        optimizer = torch.optim.SGD([
                {'params': self.generator.parameters()},
                {'params': self.discriminator.parameters()}
            ], lr=0.001, momentum=0.9
        )

        src = list(self.dataset[:][self.dataset.data_label].values())
        tgt = list(self.dataset[:][self.dataset.target_label].values())

        print(self.generator.generate(src))

        for _ in range(20):
            optimizer.zero_grad()

            synthetics = self.discriminator.forward(self.generator.forward(src, max_new_tokens=16))
            originals = self.discriminator.forward(self.generator.forward(
                [f"{x} {y}" for x, y in zip(src, tgt)],
                max_new_tokens=1
            ))

            generator_loss = torch.nn.L1Loss()(synthetics, torch.ones(synthetics.size(), device="cuda"))
            discriminator_loss = torch.nn.L1Loss()(originals, torch.zeros(originals.size(), device="cuda"))

            generator_loss.backward()
            discriminator_loss.backward()

            print(generator_loss, discriminator_loss)

            optimizer.step()

        print(self.generator.generate(src))

__all__ = ["Dataset", "neural", "Pipeline"]
