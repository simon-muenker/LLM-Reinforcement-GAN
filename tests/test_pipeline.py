import pathlib

import pytest

import llm_reinforcement_gan as rfgan


class TestPipeline:
    @pytest.fixture
    def pipeline(
        self,
        dataset: rfgan.Dataset,
        generator: rfgan.neural.Generator,
    ) -> rfgan.Pipeline:
        return rfgan.Pipeline(
            data_train=dataset,
            data_test=dataset,
            generator=generator,
            discriminator=rfgan.neural.Discriminator(size=generator.hidden_size),
            args=rfgan.PipelineArgs(
                epochs=2, batch_size=2, report_path=pathlib.Path("./tests/_outputs")
            ),
        )

    def test__call(self, pipeline: rfgan.Pipeline):
        pipeline()
