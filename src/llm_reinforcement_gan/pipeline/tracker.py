import datetime
import pathlib
import typing

import pandas
import pydantic
import rich
import seaborn


class ValueContainer(pydantic.BaseModel):
    values: typing.List[float] = []

    def add(self, val: float):
        self.values.append(val)

    @pydantic.computed_field
    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @pydantic.model_serializer()
    def serialize_model(self) -> float:
        return self.mean


class Epoch(pydantic.BaseModel):
    n: int

    loss_train_discriminator: ValueContainer = ValueContainer()
    loss_test_generator: ValueContainer = ValueContainer()

    loss_train_generator: ValueContainer = ValueContainer()
    loss_test_discriminator: ValueContainer = ValueContainer()

    _start_time: datetime.datetime = pydantic.PrivateAttr(default_factory=datetime.datetime.now)
    _end_time: datetime.datetime | None = pydantic.PrivateAttr(default=None)

    @pydantic.computed_field
    @property
    def duration(self) -> datetime.timedelta:
        if self._end_time:
            return self._end_time - self._start_time

        else:
            return datetime.datetime.now() - self._start_time

    def add_batch_train(self, loss_generator: float, loss_discriminator: float):
        self.loss_train_generator.add(loss_generator)
        self.loss_train_discriminator.add(loss_discriminator)

    def add_batch_test(self, loss_generator: float, loss_discriminator: float):
        self.loss_test_generator.add(loss_generator)
        self.loss_test_discriminator.add(loss_discriminator)

    def end(self):
        self._end_time = datetime.datetime.now()
        self.log()

    def log(self):
        rich.print(
            f"[{self.n:03d}]\t",
            f"loss(gen, train): {self.loss_train_generator.mean:2.4f}\t",
            f"loss(gen, test): {self.loss_test_generator.mean:2.4f}\t",
            f"loss(disc, train): {self.loss_train_discriminator.mean:2.4f}\t",
            f"loss(disc, test): {self.loss_test_discriminator.mean:2.4f}\t",
            f"duration: {self.duration}",
        )


class Tracker(pydantic.BaseModel):
    epochs: typing.List[Epoch] = []

    report_path: pathlib.Path = pathlib.Path(".")
    reporth_name: str = "train.tracking"

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __del__(self):
        if self.epochs:
            self.to_df().to_csv(self.report_path / f"{self.reporth_name}.csv")
            self.plot(self.report_path / f"{self.reporth_name}.pdf")

    def add(self, epoch: Epoch) -> None:
        epoch.end()
        self.epochs.append(epoch)

    def plot(self, path: pathlib.Path):
        (
            seaborn.lineplot(
                self.to_df().reset_index().melt(
                    id_vars=["n"],
                    value_vars=[
                        "loss_train_discriminator",
                        "loss_test_generator",
                        "loss_train_generator",
                        "loss_test_discriminator",
                    ],
                ),
                x="n",
                y="value",
                hue="variable",
            )
            .get_figure()
            .savefig(path, bbox_inches="tight")
        )
    
    def to_df(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.model_dump()["epochs"]).set_index("n")
