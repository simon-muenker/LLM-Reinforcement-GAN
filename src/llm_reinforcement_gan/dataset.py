import typing

import pandas
import pydantic
import torch


class Dataset(pydantic.BaseModel, torch.utils.data.Dataset):
    label: str
    df: pandas.DataFrame

    data_label: str = "data"
    target_label: str = "target"

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        return self.df.iloc[idx].to_dict()

    def __len__(self) -> int:
        return len(self.df)
