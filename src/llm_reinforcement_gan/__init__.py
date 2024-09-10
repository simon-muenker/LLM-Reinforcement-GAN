import pydantic
import torch

from llm_reinforcement_gan.dataset import Dataset


class Generator(torch.nn.Module):
    pass


class Discriminator(torch.nn.Module):
    pass


class Loss(torch.nn.Module):
    pass


class Trainer(pydantic.BaseModel):
    pass


class Pipeline(pydantic.BaseModel):
    pass


__all__ = ["Dataset"]
