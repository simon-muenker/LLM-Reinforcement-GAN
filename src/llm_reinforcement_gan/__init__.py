import typing

import cltrier_lib
import pydantic
import pandas
import rich
import rich.progress
import torch

from llm_reinforcement_gan import neural
from llm_reinforcement_gan.dataset import Dataset


class Epoch(pydantic.BaseModel):
    class LossContainer(pydantic.BaseModel):
        values: typing.List[float] = []

        def add(self, val: float):
            self.values.append(val)

        @pydantic.computed_field
        @property
        def mean(self) -> float:
            return sum(self.values) / len(self.values) if self.values else 0.0

    n: int

    loss_train_discriminator: LossContainer = LossContainer()
    loss_train_generator: LossContainer = LossContainer()

    loss_test_discriminator: LossContainer = LossContainer()
    loss_test_generator: LossContainer = LossContainer()

    def add_batch_train(self, loss_discriminator: float, loss_generator: float):
        self.loss_train_discriminator.add(loss_discriminator)
        self.loss_train_generator.add(loss_generator)

    def add_batch_test(self, loss_discriminator: float, loss_generator: float):
        self.loss_test_discriminator.add(loss_discriminator)
        self.loss_test_generator.add(loss_generator)

    def log(self):
        rich.print(
            f"[{self.n:03d}]\t",
            f"loss(train, disc):{self.loss_train_discriminator.mean:2.4f}\t",
            f"loss(train, gen):{self.loss_train_generator.mean:2.4f}\t",
            f"loss(test, disc):{self.loss_test_discriminator.mean:2.4f}\t",
            f"loss(test, gen):{self.loss_test_generator.mean:2.4f}",
        )


class PipelineArgs(pydantic.BaseModel):
    epochs: int = 50
    batch_size: int = 64

    optimizer_config: typing.Dict = dict(
        lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )


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

        epochs: typing.List[Epoch] = []

        self._predict(self.data_test)

        for n in range(1, self.args.epochs + 1):
            epoch = Epoch(n=n)

            for batch in rich.progress.track(
                self._get_data_loader(self.data_train), "Training ...", transient=True
            ):
                _, originals, synthetics = self._step(batch)

                epoch.add_batch_train(
                    loss_discriminator=self._step_discriminator(
                        originals, synthetics, optimizer_discriminator
                    ),
                    loss_generator=self._step_generator(synthetics, optimizer_generator),
                )

            for batch in rich.progress.track(
                self._get_data_loader(self.data_test), "Testing ...", transient=True
            ):
                _, originals, synthetics = self._step(batch)

                epoch.add_batch_test(
                    loss_discriminator=self._step_discriminator(originals, synthetics),
                    loss_generator=self._step_generator(synthetics),
                )

            epoch.log()
            epochs.append(epoch)

        self._predict(self.data_test)

    def _step(
        self,
        batch: typing.Tuple[
            cltrier_lib.inference.schemas.Chat, cltrier_lib.inference.schemas.Chat
        ],
    ) -> typing.Tuple[cltrier_lib.inference.schemas.Chat, cltrier_lib.inference.schemas.Chat, cltrier_lib.inference.schemas.Chat]:
        instructions, originals = batch

        synthetics = create_chats(
            self.generator.generate(instructions, max_new_tokens=64), "assistant"
        )

        return instructions, originals, synthetics

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

    def _optimize(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _predict(self, dataset: torch.utils.data.Dataset):
        rows = {
            "instructions": [],
            "originals": [],
            "synthetics": []
        }

        for batch in rich.progress.track(
            self._get_data_loader(dataset), "Predicting ...", transient=True
        ):  
            instructions, originals, synthetics = self._step(batch)

            rows["instructions"].extend([chat.messages[-1].content for chat in instructions])
            rows["originals"].extend([chat.messages[-1].content for chat in originals])
            rows["synthetics"].extend([chat.messages[-1].content for chat in synthetics])
        
        rich.print(pandas.DataFrame.from_dict(rows))

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


__all__ = ["Dataset", "neural", "Pipeline", "PipelineArgs"]


def create_chats(
    batch: typing.List[str], message_role: cltrier_lib.inference.schemas.Roles
) -> typing.List[cltrier_lib.inference.schemas.Chat]:
    return [
        cltrier_lib.inference.schemas.Chat(
            messages=[
                cltrier_lib.inference.schemas.Message(
                    role=message_role,
                    content=sample,
                )
            ]
        )
        for sample in batch
    ]
