import typing as tp
import sys
import torch
import torch.nn as nn
import spin_system

sys.path.append("..")
from spectra_manager import StationarySpectraCreator, BaseIntensityCalculator,\
    BaseSpectraIntegrator, PostSpectraProcessing, Broadener, IntegrationProcessorPowder
import constants
import mesher
from population import StationaryPopulator


class BatchedStationaryPopulator(nn.Module):
    """
    Compute the intensity of transition part depending ot population at some temperature
    (i -> j): (exp(-Ej/ kT) - exp(-Ei / kT)) / (sum (exp) )
    """

    def __init__(self, temperature: torch.Tensor = torch.tensor([300.0]), device: torch.device = torch.device("cpu")):
        """
        :param temperature: temperature in K
        """
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature, device=device))

    def forward(self, energies: torch.Tensor, lvl_down: torch.Tensor, lvl_up: torch.Tensor):
        """
        :param energies: energies in Hz
        :return: population_differences
        """
        new_shape = self.temperature.shape + (1,) * (energies.dim())
        temperature = self.temperature.reshape(new_shape)
        populations = nn.functional.softmax(-constants.unit_converter(energies, "Hz_to_K") / temperature, dim=-1)
        indexes = torch.arange(populations.shape[-2], device=energies.device)
        return populations[..., indexes, lvl_down] - populations[..., indexes, lvl_up]


class GenerationBroadener(Broadener):
    def add_hamiltonian_straine(self, sample: spin_system.MultiOrientedSample, squared_width):
        hamiltonian_width = sample.build_ham_strain().unsqueeze(-1).square()
        return (squared_width + hamiltonian_width.unsqueeze(1).unsqueeze(1)).sqrt()


class GenerationIntegrationProcessorPowder(IntegrationProcessorPowder):
    def __init__(self,
                 mesh: mesher.BaseMeshPowder,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32,
                 num_points: int = 4_000,
                 spectral_width_part: float = 0.5,
                 min_spectral_width: float = 0.01,
                 ):
        super().__init__(mesh, spectra_integrator, harmonic, post_spectra_processor,
                         chunk_size=chunk_size, device=device, dtype=dtype)
        self.register_buffer("num_points", torch.tensor(num_points, device=device))
        self.register_buffer("spectral_width_part", torch.tensor(spectral_width_part, device=device, dtype=dtype))
        self.register_buffer("min_spectral_width", torch.tensor(min_spectral_width, device=device, dtype=dtype))

        self.to(device)

    def _get_new_field(self, res_fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dims = res_fields.dim()
        batch_dims = tuple(range(max(dims-2, 0), dims))

        min_pos_batch = torch.amax(res_fields, dim=batch_dims)
        max_pos_batch = torch.amax(res_fields, dim=batch_dims)

        nature_spectra_width = torch.max(max_pos_batch - min_pos_batch, self.min_spectral_width)
        min_pos_batch = min_pos_batch * (1.0 - self.spectral_width_part * nature_spectra_width)
        max_pos_batch = max_pos_batch * (1.0 + self.spectral_width_part * nature_spectra_width)

        steps = torch.linspace(0, 1, self.num_points, device=res_fields.device, dtype=res_fields.dtype)
        fields = steps * (max_pos_batch - min_pos_batch).unsqueeze(-1) + min_pos_batch.unsqueeze(-1)
        return fields, min_pos_batch, max_pos_batch

    def forward(self,
                 res_fields: torch.Tensor,
                 intensities: torch.Tensor,
                 width: torch.Tensor,
                 gauss: torch.Tensor,
                 lorentz: torch.Tensor,
                 fields: torch.Tensor
                ):

        res_fields, width, intensities, areas = (
            self._transform_data_to_mesh_format(
                res_fields, intensities, width
            )
        )
        res_fields, width, intensities, areas = self._final_mask(res_fields, width, intensities, areas)
        fields, min_pos_batch, max_pos_batch = self._get_new_field(res_fields)
        spec = self.spectra_integrator(
            res_fields, width, intensities, areas, fields
        )
        return self.post_spectra_processor(gauss, lorentz, fields, spec), (min_pos_batch, max_pos_batch)


class GenerationCreator(StationarySpectraCreator):
    """
    Base clas of spectra creators
    """

    def __init__(self,
                 freq: tp.Union[float, torch.Tensor],
                 sample: tp.Optional[spin_system.MultiOrientedSample] = None,
                 spin_system_dim: tp.Optional[int] = None,
                 batch_dims: tp.Optional[float] = None,
                 mesh: tp.Optional[mesher.BaseMesh] = None,
                 intensity_calculator: tp.Optional[BaseIntensityCalculator] = None,
                 populator: tp.Optional[StationaryPopulator] = None,
                 spectra_integrator: tp.Optional[BaseSpectraIntegrator] = None,
                 harmonic: int = 1,
                 post_spectra_processor: PostSpectraProcessing = PostSpectraProcessing(),
                 temperature: tp.Optional[tp.Union[float, torch.Tensor]] = torch.tensor([293]),
                 recompute_spin_parameters: bool = True,
                 integration_chunk_size: int = 128,
                 device: torch.device = torch.device("cpu"),
                 dtype: torch.dtype = torch.float32
                 ):
        """
        :param freq: Resonance frequency of experiment

        :param sample: MultiOrientedSample.
            It is just an example of spin system to extract meta information (spin_system_dim, batch_dims, mesh)
            If it is None, then spin_system_dim, batch_dims, mesh should be given

        :param spin_system_dim: The size of spin system. Default is None
        :param batch_dims: The number of batch dimensions. Default is None
        :param mesh: Mesh object. Default is None
            If (mesh, batch_dims, spin_system_dim) are None then sample object should be given

        :param intensity_calculator:
            Class that is used to compute intensity of spectra via temperature/ time/ hamiltonian parameters.
            Default is None
            If it is None then it will be initialized as default calculator specific to given spectra_creator

        :param populator:
            Class that is used to compute part intensity due to population of levels. Default is None
            If it is None then it will be initialized as default calculator specific to given intensity_calculator

        :param spectra_integrator:
            Class to integrate the resonance lines to get the spectrum

        :param harmonic: Harmonic of spectra: 1 is derivative, 0 is absorbance
        :param post_spectra_processor:
            Class to post process resulted resonance data (fields, intensities, width):
            integration, mesh mapping and so on. Default post_spectra_processor is powder spectra processor

        :param temperature: The temperature of an experiment. If populator is not None it takes from it
        :param recompute_spin_parameters:
            Recompute spin parameters in __call__ methods. For stationary creator is True, for time resolves is False

        """
        if populator is None:
            populator = BatchedStationaryPopulator(temperature=temperature)
        super().__init__(freq,
                         sample,
                         spin_system_dim,
                         batch_dims,
                         mesh,
                         intensity_calculator,
                         populator=populator,
                         spectra_integrator=spectra_integrator,
                         harmonic=harmonic,
                         post_spectra_processor=post_spectra_processor,
                         temperature=temperature,
                         recompute_spin_parameters=recompute_spin_parameters,
                         integration_chunk_size=integration_chunk_size,
                         device=device,
                         dtype=dtype)
        self.broader = GenerationBroadener(device=device)


    def _init_spectra_processor(self,
                                spectra_integrator: tp.Optional[BaseSpectraIntegrator],
                                harmonic: int,
                                post_spectra_processor: PostSpectraProcessing,
                                chunk_size: int, device: torch.device, dtype: torch.dtype) ->\
            IntegrationProcessorPowder:
        if self.mesh.name == "PowderMesh":
            return GenerationIntegrationProcessorPowder(self.mesh, spectra_integrator, harmonic,
                                                        post_spectra_processor, chunk_size=chunk_size, device=device,
                                                        dtype=dtype)

        elif self.mesh.name == "CrystalMesh":
            raise NotImplementedError

        else:
            return GenerationIntegrationProcessorPowder(self.mesh, spectra_integrator, harmonic,
                                                        post_spectra_processor, chunk_size=chunk_size, device=device,
                                                        dtype=dtype)

    def compute_parameters(self, sample: spin_system.MultiOrientedSample,
                           F: torch.Tensor,
                           Gx: torch.Tensor,
                           Gy: torch.Tensor,
                           Gz: torch.Tensor,
                           vector_down: torch.Tensor, vector_up: torch.Tensor,
                           lvl_down: torch.Tensor, lvl_up: torch.Tensor,
                           res_fields: torch.Tensor,
                           resonance_energies: torch.Tensor,
                           full_system_vectors: tp.Optional[torch.Tensor]) -> \
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tp.Any]]:
        """
        :param sample: The sample which transitions must be found
        :param F: Magnetic free part of spin Hamiltonian H = F + B * G
        :param Gx: x-part of Hamiltonian Zeeman Term
        :param Gy: y-part of Hamiltonian Zeeman Term
        :param Gz: z-part of Hamiltonian Zeeman Term

        :param vector_down:
            Eigenvectors of the lower energy states. The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param vector_up:
            Eigenvectors of the upper energy states.The shape is [...., M, N],
            where M is number of transitions, N is number of levels

        :param lvl_down:
            Energy levels of lower states from which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param lvl_up:
            Energy levels of upper states to which transitions occur.
            Shape: [time, ..., N], where time is the time dimension and
            N is the number of energy levels.

        :param resonance_energies:
            Energies of spin states. The shape is [..., N]

        :param res_fields: Resonance fields. The shape os [..., N]

        :param full_system_vectors: Eigen vector of each level of a spin system. The shape os [..., N, N]

        :return: tuple of the next data
         - Resonance fields
         - Intensities of transitions
         - Width of transition lines
         - extras parameters computed in _compute_additional
        """
        intensities = self.intensity_calculator.compute_intensity(
            Gx, Gy, Gz, vector_down, vector_up, lvl_down, lvl_up, resonance_energies, res_fields, full_system_vectors
        )

        lines_dimension = tuple(range(intensities.ndim - 1))
        intensities_mask = (intensities / intensities.abs().max() > self.threshold).any(dim=lines_dimension)
        intensities = intensities[..., intensities_mask]

        extras = self._mask_additional(vector_down,
                                       vector_up, lvl_down, lvl_up, resonance_energies,
                                       full_system_vectors)

        extras = self._mask_components(intensities_mask, *extras)

        res_fields = res_fields[..., intensities_mask].unsqueeze(0).unsqueeze(0)
        vector_u = vector_down[..., intensities_mask, :].unsqueeze(0).unsqueeze(0)
        vector_v = vector_up[..., intensities_mask, :].unsqueeze(0).unsqueeze(0)

        freq_to_field = self._freq_to_field(vector_u, vector_v, Gz.unsqueeze(0).unsqueeze(0))
        intensities = intensities.unsqueeze(0)
        intensities *= freq_to_field
        intensities = intensities / intensities.abs().max()
        width = self.broader(sample, vector_u, vector_v, res_fields) * freq_to_field

        extras = self._compute_additional(
            sample, F, Gx, Gy, Gz, *extras
        )
        width_size = width.shape[0]
        temp_size = intensities.shape[1]
        common_shape = intensities.shape[2:]
        target_shape = [width_size, temp_size, *common_shape]

        res_fields = res_fields.expand(target_shape)
        intensities = intensities.expand(target_shape)
        width = width.expand(target_shape)
        return res_fields, intensities, width, *extras

