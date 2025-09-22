from torch import nn
import torch
import constants


class StationaryPopulator(nn.Module):
    """
    Compute the intensity of transition part depending ot population at some temperature
    (i -> j): (exp(-Ej/ kT) - exp(-Ei / kT)) / (sum (exp) )
    """
    def __init__(self, temperature: float = 300.0, device: torch.device = torch.device("cpu")):
        """
        :param temperature: temperature in K
        """
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature, device=device))
        self.to(device)

    def forward(self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor, *args, **kwargs):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        populations = nn.functional.softmax(-constants.unit_converter(energies, "Hz_to_K") / self.temperature, dim=-1)
        indexes = torch.arange(populations.shape[-2], device=energies.device)
        return populations[..., indexes, lvl_down] - populations[..., indexes, lvl_up]
