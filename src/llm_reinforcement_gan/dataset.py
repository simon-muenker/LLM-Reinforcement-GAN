import typing

import pandas
import pydantic
import torch


class Dataset(pydantic.BaseModel, torch.utils.data.Dataset):
    label: str
    df: pandas.DataFrame
    instruction: str

    data_label: str
    target_label: str

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        return self.df.iloc[idx].to_dict()

    def __len__(self) -> int:
        return len(self.df)
