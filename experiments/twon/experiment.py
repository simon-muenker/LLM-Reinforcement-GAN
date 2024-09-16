import pathlib

import pandas
import transformers

import llm_reinforcement_gan as rfgan

DATASET_META = dict(
    instruction="You are a social media user and react to incoming messages in the form of Twitter-like replies.",
    data_label="post",
    target_label="reply",
)
MODEL_SLUG = "Qwen/Qwen2-0.5B-Instruct"  # "meta-llama/Meta-Llama-3.1-8B-Instruct"

generator = rfgan.neural.Generator(
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL_SLUG),
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_SLUG, torch_dtype="auto", device_map="cuda:0"
    ),
)

rfgan.Pipeline(
    data_train=rfgan.Dataset(
        label="debug_train",
        df=pandas.read_parquet("./experiments/twon/data/prelim.train.parquet"),
        **DATASET_META,
    ),
    data_test=rfgan.Dataset(
        label="debug_test",
        df=pandas.read_parquet("./experiments/twon/data/prelim.test.parquet"),
        **DATASET_META,
    ),
    generator=generator,
    discriminator=rfgan.neural.Discriminator(size=generator.hidden_size),
    args=rfgan.PipelineArgs(
        epochs=5, batch_size=80, report_path=pathlib.Path("./experiments/twon/results")
    ),
)()
