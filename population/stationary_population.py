from torch import nn
import torch
import constants

class StationaryPopulator:
    def __init__(self, temperature: float = 300.0):
        """
        :param temperature: temperature in K
        """
        self.temperature = temperature

    def __call__(self, energies, lvl_down, lvl_up):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        populations = nn.functional.softmax(-constants.unit_converter(energies) / self.temperature, dim=-1)
        indexes = torch.arange(populations.shape[-2], device=populations.device)
        return populations[..., indexes, lvl_down] - populations[..., indexes, lvl_up]
