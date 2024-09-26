import torch


class Loss(torch.nn.Module):
    def __init__(self, loss_fn: torch.nn.Module = torch.nn.BCELoss(), device: torch.device = torch.device("cuda")):
        super().__init__()

        self.loss_fn = loss_fn
        self.tensor_type = dict(device=device, dtype=torch.bfloat16)

        self.w_g: float = 1.0
        self.w_d: float = 1.0

    def generator(self, synthetic: torch.Tensor) -> torch.Tensor:
        return self.w_g * self.loss_fn(
            synthetic, torch.zeros(synthetic.size(), **self.tensor_type)
        )

    def discriminator(self, original: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
        return (
            self.w_d
            * (
                self.loss_fn(original, torch.zeros(original.size(), **self.tensor_type))
                + self.loss_fn(synthetic, torch.ones(synthetic.size(), **self.tensor_type))
            )
            / 2
        )
