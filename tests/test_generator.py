import typing

import cltrier_lib
import pytest
import rich
import torch

import llm_reinforcement_gan as rfgan


class TestGenerator:
    @pytest.fixture
    def chats(
        self, data: typing.List[typing.Dict]
    ) -> typing.List[cltrier_lib.inference.schemas.Chat]:
        return [
            cltrier_lib.inference.schemas.Chat(
                messages=[
                    cltrier_lib.inference.schemas.Message(
                        role="user",
                        content=sample["data"],
                    )
                ]
            )
            for sample in data
        ]

    def test__generate(
        self,
        generator: rfgan.neural.Generator,
        chats: typing.List[cltrier_lib.inference.schemas.Chat],
    ):
        rich.print(generator.generate(chats))

    def test__embed(
        self,
        generator: rfgan.neural.Generator,
        chats: typing.List[cltrier_lib.inference.schemas.Chat],
    ):
        logits, hidden_state = generator.embed(chats)

        rich.print(logits.size())
        rich.print(torch.argmax(logits, dim=-1, keepdim=True).size())
        rich.print(hidden_state.size())
