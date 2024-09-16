# LLM Reinforcement with GANs: Aligning LLMs through Discriminator Reward Functions

General training structure similar to RLGAF [^1].

## Experimental Design

```mermaid
flowchart TD

    data[(Sample)]
    instruction(Instruction)

    real_response(Real Response)
    real_embed(Real Embedding)
    real_prediction(Real Prediction)

    synthetic_response(Synthetic Response)
    synthetic_embed(Synthetic Embedding)
    synthetic_prediction(Synthetic Prediction)

    generator{{Generator: Instruction-tuned LLM}}
    generator_link{{Generator: Instruction-tuned LLM}}
    generator o--o generator_link

    discriminator{{Discriminator: Vanilla Transformer Encoder + Classifier}}

    loss_discriminator>Discriminator Loss]
    loss_generator>Generator Loss]


	data --> instruction
    data --> real_response

    instruction --> generator
	
    generator -->|generate| synthetic_response

    subgraph train_generator[generator train]
    
        synthetic_response --> generator_link
        real_response --> generator_link

        generator_link -->|encode| synthetic_embed
        generator_link -->|encode| real_embed

        synthetic_embed & real_embed --> discriminator

        subgraph discriminator train
            discriminator -->|classify| real_prediction & synthetic_prediction

            real_prediction --> loss_generator
            real_prediction & synthetic_prediction --> loss_discriminator

            loss_discriminator -.->|optimize| discriminator

            end
        
        loss_generator -.->|optimize| generator_link
    
    end


	classDef model fill:#EEE
	generator:::model
	discriminator:::model
    generator_link:::model

	classDef real fill:#d9ead3
	real_response:::real
    real_embed:::real
    real_prediction:::real

	classDef fake fill:#f4cccc
	synthetic_response:::fake
    synthetic_embed:::fake
    synthetic_prediction:::fake
```

## Roadmap

1. Preparing an instruction-tunable dataset based on TWON dataset
2. 

[^1]: Fine-tuning Language Models with Generative Adversarial Reward Modelling: <https://arxiv.org/abs/2305.06176>