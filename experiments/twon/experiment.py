import pathlib
import argparse

import torch
import pandas
import transformers

import llm_reinforcement_gan as rfgan

DATASET_PATH: str = "./experiments/twon/data"
REPORT_PATH: str = "./experiments/twon/results"

DATASET_META = dict(
    instruction="You are a social media user. Write a matching post to the given topic.",
    data_label="annotation",
    target_label="full_text",
)

DATASETS = dict(
    prelim=lambda split: f"./{DATASET_PATH}/prelim.{split}.parquet",
    left=lambda split: f"./{DATASET_PATH}/left.{split}.parquet",
    right=lambda split: f"./{DATASET_PATH}/right.{split}.parquet",
)

MODELS = dict(
    qwen_tiny="Qwen/Qwen2-0.5B-Instruct",
    phi_mini="microsoft/Phi-3.5-mini-instruct",
    llama31_small="meta-llama/Meta-Llama-3.1-8B-Instruct",
    llama32_tiny="meta-llama/Llama-3.2-1B-Instruct",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama32_tiny", help="model to optimize")
    parser.add_argument("--dataset", type=str, default="right", help="dataset to use")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--device", type=str, default="cuda:2", help="gpu computation")
    parser.add_argument("--use_lora", type=bool, default=False, help="train lora adapters")

    cfg = parser.parse_args()

    generator = rfgan.neural.Generator(
        tokenizer=transformers.AutoTokenizer.from_pretrained(MODELS[cfg.model]),
        model=transformers.AutoModelForCausalLM.from_pretrained(
            MODELS[cfg.model],
            torch_dtype="auto",
            device_map=cfg.device,
        ),
        args=rfgan.neural.GeneratorArgs(use_lora=cfg.use_lora)
    )

    rfgan.Pipeline(
        data_train=rfgan.Dataset(
            label=f"{cfg.dataset}_train",
            df=pandas.read_parquet(DATASETS[cfg.dataset]("train")),
            **DATASET_META,
        ),
        data_test=rfgan.Dataset(
            label=f"{cfg.dataset}_test",
            df=pandas.read_parquet(DATASETS[cfg.dataset]("test")),
            **DATASET_META,
        ),
        generator=generator,
        discriminator=rfgan.neural.Discriminator(input_size=generator.hidden_size).to(torch.device(cfg.device)),
        loss_fn=rfgan.neural.Loss(device=torch.device(cfg.device)),
        args=rfgan.PipelineArgs(
            epochs=cfg.epochs, batch_size=cfg.batch_size, report_path=pathlib.Path(REPORT_PATH)
        ),
    )()
