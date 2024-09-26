import pathlib

import torch
import pandas
import transformers

import llm_reinforcement_gan as rfgan

DATASET_META = dict(
    instruction="You are a social media user. Write a matching post to the given topic.",
    data_label="annotation",
    target_label="full_text",
)
MODEL_SLUG: str = "Qwen/Qwen2-0.5B-Instruct"  # "microsoft/Phi-3.5-mini-instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"

DEVICE: str = "cuda:2"

generator = rfgan.neural.Generator(
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL_SLUG),
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_SLUG,
        torch_dtype="auto",
        device_map=DEVICE,
    ),
)

rfgan.Pipeline(
    data_train=rfgan.Dataset(
        label="debug_train",
        df=pandas.read_parquet("./experiments/twon/data/left.train.parquet"),
        **DATASET_META,
    ),
    data_test=rfgan.Dataset(
        label="debug_test",
        df=pandas.read_parquet("./experiments/twon/data/left.test.parquet"),
        **DATASET_META,
    ),
    generator=generator,
    discriminator=rfgan.neural.Discriminator(input_size=generator.hidden_size).to(torch.device(DEVICE)),
    loss_fn=rfgan.neural.Loss(device=torch.device(DEVICE)),
    args=rfgan.PipelineArgs(
        epochs=1, batch_size=32, report_path=pathlib.Path("./experiments/twon/results")
    ),
)()
