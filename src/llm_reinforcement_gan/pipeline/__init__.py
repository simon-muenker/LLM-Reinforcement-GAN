import pathlib
import typing

import cltrier_lib
import pandas
import pydantic
import rich
import rich.progress
import torch

from llm_reinforcement_gan import neural
from llm_reinforcement_gan.dataset import Dataset
from llm_reinforcement_gan.pipeline.tracker import Epoch, Tracker
from llm_reinforcement_gan.pipeline.util import create_chats


class PipelineArgs(pydantic.BaseModel):
    epochs: int = 50
    batch_size: int = 64

    optimizer_config: typing.Dict = dict(
        lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )

    report_path: pathlib.Path = pathlib.Path(".")


class Pipeline(pydantic.BaseModel):
    data_train: Dataset
    data_test: Dataset

    generator: neural.Generator
    discriminator: neural.Discriminator

    loss_fn: neural.Loss = neural.Loss()

    args: PipelineArgs = PipelineArgs()
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __call__(self):
        optimizer_discriminator = torch.optim.AdamW(
            self.discriminator.parameters(), **self.args.optimizer_config
        )
        optimizer_generator = torch.optim.AdamW(
            self.generator.parameters(), **self.args.optimizer_config
        )

        tracker: Tracker = Tracker(report_path=self.args.report_path)

        self._predict(self.data_test).to_csv(
            self.args.report_path / f"predictions.before.csv", index=False
        )

        for n in range(1, self.args.epochs + 1):
            epoch = Epoch(n=n)

            for batch in rich.progress.track(
                self._get_data_loader(self.data_train), "Training ...", transient=True
            ):
                _, originals, synthetics = self._step(batch)

                epoch.add_batch_train(
                    loss_generator=self._step_generator(synthetics, optimizer_generator),
                    loss_discriminator=self._step_discriminator(
                        originals, synthetics, optimizer_discriminator
                    ),
                )

            with torch.no_grad():
                for batch in rich.progress.track(
                    self._get_data_loader(self.data_test), "Testing ...", transient=True
                ):
                    _, originals, synthetics = self._step(batch)

                    epoch.add_batch_test(
                        loss_generator=self._step_generator(synthetics),
                        loss_discriminator=self._step_discriminator(originals, synthetics),
                    )

            tracker.add(epoch)

        self._predict(self.data_test).to_csv(
            self.args.report_path / f"predictions.after.csv", index=False
        )

    def _step(
        self,
        batch: typing.Tuple[
            cltrier_lib.inference.schemas.Chat, cltrier_lib.inference.schemas.Chat
        ],
    ) -> typing.Tuple[
        cltrier_lib.inference.schemas.Chat,
        cltrier_lib.inference.schemas.Chat,
        cltrier_lib.inference.schemas.Chat,
    ]:
        instructions, originals = batch

        synthetics = create_chats(
            self.generator.generate(instructions, max_new_tokens=64), "assistant"
        )

        return instructions, originals, synthetics

    def _step_generator(
        self,
        synthetics: cltrier_lib.inference.schemas.Chat,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        synthetic_embeds = self.generator.embed(synthetics)
        synthetic_preds = self.discriminator.forward(synthetic_embeds)

        loss = self.loss_fn.generator(synthetic_preds)

        if optimizer:
            self._optimize(loss, optimizer)

        return loss.item()

    def _step_discriminator(
        self,
        originals: cltrier_lib.inference.schemas.Chat,
        synthetics: cltrier_lib.inference.schemas.Chat,
        optimizer: torch.optim.Optimizer | None = None,
    ):
        original_embeds = self.generator.embed(originals)
        synthetic_embeds = self.generator.embed(synthetics)

        original_preds = self.discriminator.forward(original_embeds)
        synthetic_preds = self.discriminator.forward(synthetic_embeds)

        loss = self.loss_fn.discriminator(original_preds, synthetic_preds)

        if optimizer:
            self._optimize(loss, optimizer)

        return loss.item()

    def _optimize(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _predict(self, dataset: torch.utils.data.Dataset) -> pandas.DataFrame:
        rows: typing.Dict[str, typing.List] = {
            "instructions": [],
            "originals": [],
            "synthetics": [],
        }

        for batch in rich.progress.track(
            self._get_data_loader(dataset), "Predicting ...", transient=True
        ):
            instructions, originals, synthetics = self._step(batch)

            rows["instructions"].extend([chat.messages[-1].content for chat in instructions])
            rows["originals"].extend([chat.messages[-1].content for chat in originals])
            rows["synthetics"].extend([chat.messages[-1].content for chat in synthetics])

        return pandas.DataFrame.from_dict(rows)

    def _get_data_loader(self, dataset: torch.utils.data.Dataset):
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            collate_fn=self._collate,
        )

    def _collate(
        self, batch: typing.List[typing.Dict]
    ) -> typing.Tuple[cltrier_lib.inference.schemas.Chat, cltrier_lib.inference.schemas.Chat]:
        zipped_batch: typing.Tuple[typing.List[str], typing.List[str]] = tuple(
            zip(
                *[
                    (
                        sample[self.data_train.data_label],
                        sample[self.data_train.target_label],
                    )
                    for sample in batch
                ]
            )
        )

        return create_chats(zipped_batch[0], "user"), create_chats(zipped_batch[1], "assistant")
