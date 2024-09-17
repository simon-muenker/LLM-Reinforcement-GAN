# LLM Reinforcement with GANs: Aligning LLMs through Discriminator Reward Functions

General training structure similar to RLGAF [^1].
Discriminator archicture inspired by TransLSTM [^2].
Loss/Objective equal to orginal GAN proposal.

## Experimental Design

```mermaid
flowchart TD

	classDef model fill:#EEE
	classDef real fill:#d9ead3
	classDef fake fill:#f4cccc

    data[(Sample)]
    instruction(Instruction)

    real_response(Real Response):::real
    real_embed(Real Embedding):::real
    real_prediction(Real Prediction):::real

    synthetic_response(Synthetic Response):::fake
    synthetic_embed(Synthetic Embedding):::fake
    synthetic_prediction(Synthetic Prediction):::fake

    generator{{Generator: Instruction-tuned LLM}}:::model
    generator_link{{Generator: Instruction-tuned LLM}}:::model
    generator o--o generator_link

    discriminator{{Discriminator: Vanilla Transformer Encoder + Classifier}}:::model

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

            synthetic_prediction --> loss_generator
            real_prediction & synthetic_prediction --> loss_discriminator

            loss_discriminator -.->|optimize| discriminator

            end
        
        loss_generator -.->|optimize| generator_link
    
    end
```

## Roadmap

1. Preparing an instruction-tunable dataset based on TWON dataset
2. 

[^1]: Fine-tuning Language Models with Generative Adversarial Reward Modelling: <https://arxiv.org/abs/2305.06176>
[^2]: TransLSTM: A hybrid LSTM-Transformer model for fine-grained suggestion mining <https://www.sciencedirect.com/science/article/pii/S2949719124000372  >