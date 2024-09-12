import typing

import cltrier_lib
import torch
import transformers

from llm_reinforcement_gan.neural import util

transformers.logging.set_verbosity_error()


class Generator(torch.nn.Module):
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        model: transformers.AutoModelForCausalLM,
    ):
        super().__init__()
        self.tokenizer: transformers.AutoTokenizer = tokenizer
        self.model: transformers.AutoModelForCausalLM = model

    def format_chat(
        self, batch: typing.List[cltrier_lib.inference.schemas.Chat]
    ) -> typing.List[str]:
        return [
            self.tokenizer.apply_chat_template(
                chat.model_dump()["messages"], tokenize=False, add_generation_prompt=True
            )
            for chat in batch
        ]

    def tokenize(
        self, batch: typing.List[str]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
            "cuda"
        )

    def prepare(
        self, batch: typing.List[cltrier_lib.inference.schemas.Chat]
    ) -> transformers.tokenization_utils_base.BatchEncoding:
        return self.tokenize(self.format_chat(batch))

    def decode(self, batch: typing.List[torch.Tensor]) -> typing.List[str]:
        return self.tokenizer.batch_decode(batch, skip_special_tokens=True)

    def embed(self, batch: typing.List[cltrier_lib.inference.schemas.Chat]) -> torch.Tensor:
        model_inputs = self.prepare(batch)

        outputs = self.model(**model_inputs, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]

        return outputs

    def generate(
        self, batch: typing.List[cltrier_lib.inference.schemas.Chat], max_new_tokens: int = 12
    ) -> typing.List[str]:
        model_inputs = self.prepare(batch)

        generated_ids = self.model.generate(
            model_inputs.input_ids, max_new_tokens=max_new_tokens
        )

        generated_ids = util.remove_prefixes(generated_ids, model_inputs.input_ids)

        return self.decode(generated_ids)

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size
