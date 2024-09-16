import typing

import rich

import llm_reinforcement_gan as rfgan


class TestDataset:
    def test__log_dataset(self, dataset: rfgan.Dataset):
        rich.print(dataset)

    def test__get_dataset_item(self, dataset: rfgan.Dataset, data: typing.List[typing.Dict]):
        assert dataset[0] == data[0]

    def test__get_dataset_len(self, dataset: rfgan.Dataset, data: typing.List[typing.Dict]):
        assert len(dataset) == len(data)
