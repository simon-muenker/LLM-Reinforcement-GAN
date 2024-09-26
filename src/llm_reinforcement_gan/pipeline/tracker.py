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


class RowContainer(pydantic.BaseModel):
    label: str

    generator: ValueContainer = ValueContainer()
    discriminator: ValueContainer = ValueContainer()

    def add_batch(self, value_generator: float, value_discriminator: float):
        self.generator.add(value_generator)
        self.discriminator.add(value_discriminator)

    def to_df(self, n: int = 1) -> pandas.DataFrame:
        return pandas.DataFrame(
            data={
                (self.label, "generator"): self.generator.values,
                (self.label, "discriminator"): self.discriminator.values,
            },
            index=[
                i + len(self.generator.values) * (n - 1)
                for i in range(1, len(self.generator.values) + 1)
            ],
        ).rename_axis("iteration")


class Epoch(pydantic.BaseModel):
    n: int

    train: RowContainer = RowContainer(label="train")
    test: RowContainer = RowContainer(label="test")

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
        self.train.add_batch(loss_generator, loss_discriminator)

    def add_batch_test(self, loss_generator: float, loss_discriminator: float):
        self.test.add_batch(loss_generator, loss_discriminator)

    def end(self):
        self._end_time = datetime.datetime.now()
        self.log()

    def log(self):
        rich.print(
            f"[{self.n:03d}]\t",
            f"loss(train, gen): {self.train.generator.mean:2.4f}\t",
            f"loss(train, disc): {self.train.discriminator.mean:2.4f}\t",
            f"loss(test, gen): {self.test.generator.mean:2.4f}\t",
            f"loss(test, disc): {self.test.discriminator.mean:2.4f}\t",
            f"duration: {self.duration}",
        )

    def to_df(self) -> pandas.DataFrame:
        return self.train.to_df(self.n).join(self.test.to_df(self.n), how="outer")


class Tracker(pydantic.BaseModel):
    epochs: typing.List[Epoch] = []

    report_path: pathlib.Path = pathlib.Path(".")
    reporth_name: str = "train.tracking"

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __del__(self):
        if self.epochs:
            self.to_df().to_json(
                self.report_path / f"{self.reporth_name}.json", orient="records", indent=4
            )
            self.plot(self.report_path / f"{self.reporth_name}.pdf")

    def add(self, epoch: Epoch) -> None:
        epoch.end()
        self.epochs.append(epoch)

    def plot(self, path: pathlib.Path):
        g = seaborn.FacetGrid(
            (
                pandas.concat([epoch.to_df() for epoch in self.epochs])
                .reset_index()
                .melt(
                    id_vars=[("iteration", "")],
                    value_vars=[
                        ("train", "generator"),
                        ("train", "discriminator"),
                        ("test", "generator"),
                        ("test", "discriminator"),
                    ],
                )
                .rename(
                    columns={
                        "variable_0": "split",
                        "variable_1": "component",
                        ("iteration", ""): "iteration",
                    }
                )
            ),
            col="split",
            hue="component",
            sharex=False,
            aspect=1.6,
        )
        g.map_dataframe(
            seaborn.lineplot,
            x="iteration",
            y="value",
        )
        g.add_legend()
        g.savefig(path, bbox_inches="tight")

    def to_df(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.model_dump()["epochs"])
