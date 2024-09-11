import typing

import pandas
import pytest
import rich

import llm_reinforcement_gan as rfgan


class TestPipeline:
    @pytest.fixture
    def pipeline(self, dataset: rfgan.Dataset, generator: rfgan.neural.Generator,) -> rfgan.Pipeline:
        return rfgan.Pipeline(
            dataset=dataset,
            generator=generator,
            discriminator=rfgan.neural.Discriminator(size=generator.hidden_size)
        )

    def test__call(self, pipeline: rfgan.Pipeline):
        pipeline()