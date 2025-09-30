import torch
import torch.nn as nn

import spin_system

### The energy computation is not effective due to the usage of common interface for population computation
### It should be rebuild without expand operation
class Locator(nn.Module):
    def __init__(self, output_full_eigenvector: bool, spin_dim: int, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.output_full_eigenvector = output_full_eigenvector
        self._triu_indices = torch.triu_indices(spin_dim, spin_dim, offset=1, device=device)
        self.device = device

    def _get_resonance_indexes(self, eigen_values: torch.Tensor,
                               freq_low: torch.Tensor, freq_high: torch.Tensor):
        transitions = eigen_values[..., None, :] - eigen_values[..., :, None]

        i_indices, j_indices = self._triu_indices
        transition_freq = transitions[..., i_indices, j_indices]

        valid_mask = (transition_freq >= freq_low[..., None]) & (transition_freq <= freq_high[..., None])
        return valid_mask, i_indices, j_indices, transition_freq

    def forward(self,
                F: torch.Tensor, Gz: torch.Tensor, freq_low: torch.Tensor,
                freq_high: torch.Tensor, resonance_field: torch.Tensor
                ):
        H = F + Gz * resonance_field
        eigen_values, eigen_vectors = torch.linalg.eigh(H)
        valid_mask, i_indices, j_indices, transition_freq = self._get_resonance_indexes(
            eigen_values, freq_low, freq_high
        )
        mask_trans = valid_mask.any(dim=-2)

        lvl_down = i_indices[mask_trans]
        lvl_up = j_indices[mask_trans]
        transition_freq = transition_freq[..., mask_trans]


        vectors_u = eigen_vectors[..., lvl_down].transpose(-2, -1) * valid_mask[..., mask_trans].unsqueeze(-1)
        vectors_v = eigen_vectors[..., lvl_up].transpose(-2, -1) * valid_mask[..., mask_trans].unsqueeze(-1)

        if self.output_full_eigenvector:
            full_eigen_vectors =\
                eigen_vectors.unsqueeze(-3).expand(-1, lvl_down.shape[-1], -1, -1) *\
                valid_mask[..., mask_trans].unsqueeze(-1).unsqueeze(-1)
        else:
            full_eigen_vectors = None

        eigen_values = eigen_values.unsqueeze(-2).expand(-1, lvl_down.shape[-1], -1)
        return (vectors_u, vectors_v), (lvl_down, lvl_up),\
            transition_freq, eigen_values, full_eigen_vectors


class ResFreq(nn.Module):
    def __init__(self,
                 spin_system_dim: int,
                 mesh_size: torch.Size,
                 batch_dims: torch.Size | tuple,
                 output_full_eigenvector: bool = False,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32):
        """
        :param eigen_finder: The eigen solver that should find eigen values and eigen vectors
        """
        super().__init__()
        self.register_buffer('spin_system_dim', torch.tensor(spin_system_dim))
        self.output_full_eigenvector = output_full_eigenvector
        self.mesh_size = mesh_size
        self.batch_dims = batch_dims
        self.locator = Locator(output_full_eigenvector, spin_system_dim, device, dtype=dtype)
        self.device = device

    def forward(self, sample: spin_system.BaseSample,
                 resonance_field: torch.Tensor,
                 freq_low: torch.Tensor, freq_high: torch.Tensor, F: torch.Tensor, Gz: torch.Tensor) ->\
            tuple[
            tuple[torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor],
            torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        :param sample: The sample for which the resonance parameters need to be found
        :param resonance_field: the resonance field. The shape is []
        :param freq_low: low limit of frequency intervals. The shape is [batch_dim]
        :param freq_high: high limit of frequency intervals. The shape is [batch_dim]
        :param F: The magnetic free part of Hamiltonian
        :param Gz: z-part of Zeeman magnetic field term B * Gz
        :return: list of next data:
        - tuple of the eigen vectors of high transition states and of low transition states and Vi and Vj where i>j
        is EPR transition
        - tuple of valid indexes of levels between which transition occurs
        - magnetic frequency of transitions
        - resonance energies
        - vector_full_system | None. The eigen vectors for all energy levels
        """
        config_dims = (*self.batch_dims, *self.mesh_size)
        (vectors_u, vectors_v), (lvl_down, lvl_up), \
            transition_freq, eigen_values, full_eigen_vectors = self.locator(
            F.flatten(0, -3), Gz.flatten(0, -3), freq_low.flatten(0, -1), freq_high.flatten(0, -1), resonance_field
        )

        max_columns = lvl_down.shape[-1]

        vectors_u = vectors_u.view(*config_dims, max_columns, self.spin_system_dim)
        vectors_v = vectors_v.view(*config_dims, max_columns, self.spin_system_dim)

        eigen_values = eigen_values.view(*config_dims, max_columns, self.spin_system_dim)
        transition_freq = transition_freq.view(*config_dims, max_columns)

        if full_eigen_vectors is not None:
            full_eigen_vectors = full_eigen_vectors.view(
                *config_dims, max_columns, self.spin_system_dim, self.spin_system_dim
            )
        return (vectors_u, vectors_v), (lvl_down, lvl_up),\
            transition_freq, eigen_values, full_eigen_vectors