# LLM Reinforcement with GANs: Aligning LLMs through Discriminator Reward Functions

General training structure similar to RLGAF [^1].

## Experimental Design

```mermaid
flowchart TD

	A[(Twitter Threads)] --> B{{Agent: Generative LLM}}
	A --> C(Real Human Reply)
	B --> D(Generated Agent Reply)
	C & D --> E{{Discriminator: Encoder+Classifier}}
	E --> F>Agent Loss]
	E --> G>Discriminator Loss]

	classDef model fill:#EEE
	B:::model
	E:::model

	classDef real fill:#d9ead3
	C:::real

	classDef fake fill:#f4cccc
	D:::fake
```

## Roadmap

1. Preparing an instruction-tunable dataset based on TWON dataset
2. 

[^1]: Fine-tuning Language Models with Generative Adversarial Reward Modelling: <https://arxiv.org/abs/2305.06176>