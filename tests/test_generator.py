import typing

import rich

import llm_reinforcement_gan as rfgan


class TestGenerator:
    def test__generate(self, generator: rfgan.neural.Generator, data: typing.List[typing.Dict]):
        rich.print(generator.generate([sample["data"] for sample in data]))

    def test__forward(self, generator: rfgan.neural.Generator, data: typing.List[typing.Dict]):
        rich.print(generator.forward([sample["data"] for sample in data]))
