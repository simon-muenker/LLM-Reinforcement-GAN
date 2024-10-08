import pathlib

import pytest
import torch

import llm_reinforcement_gan as rfgan


class TestPipeline:
    @pytest.fixture
    def pipeline(
        self,
        dataset: rfgan.Dataset,
        generator: rfgan.neural.Generator,
        device: str
    ) -> rfgan.Pipeline:
        return rfgan.Pipeline(
            data_train=dataset,
            data_test=dataset,
            generator=generator,
            discriminator=rfgan.neural.Discriminator(input_size=generator.hidden_size).to(torch.device(device)),
            loss_fn=rfgan.neural.Loss(device=torch.device(device)),
            args=rfgan.PipelineArgs(
                epochs=10, batch_size=2, report_path=pathlib.Path("./tests/_outputs")
            ),
        )

    def test__call(self, pipeline: rfgan.Pipeline):
        pipeline()
