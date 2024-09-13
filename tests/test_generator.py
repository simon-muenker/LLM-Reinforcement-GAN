import typing

import cltrier_lib
import pytest
import rich

import llm_reinforcement_gan as rfgan
from llm_reinforcement_gan.pipeline.util import create_chats


class TestGenerator:
    @pytest.fixture
    def chats(
        self, data: typing.List[typing.Dict]
    ) -> typing.List[cltrier_lib.inference.schemas.Chat]:
        return create_chats([sample["data"] for sample in data], message_role="user")

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
        rich.print(generator.embed(chats))
