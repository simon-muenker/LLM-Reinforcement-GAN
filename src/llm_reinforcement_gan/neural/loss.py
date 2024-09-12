import torch


class Loss(torch.nn.Module):
    def __init__(
        self, loss_fn: torch.nn.Module = torch.nn.L1Loss(), w_o: float = 1.0, w_s: float = 1.0
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.w_o = w_o
        self.w_s = w_s

    def discriminator(self, original: torch.Tensor, synthetic: torch.Tensor) -> torch.Tensor:
        return self.w_o * self.loss_fn(
            original, torch.zeros(original.size(), device="cuda")
        ) + self.w_s * self.loss_fn(synthetic, torch.ones(synthetic.size(), device="cuda"))

    def generator(self, synthetic: torch.Tensor) -> torch.Tensor:
        return self.w_s * self.loss_fn(synthetic, torch.zeros(synthetic.size(), device="cuda"))
