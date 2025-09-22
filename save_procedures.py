import itertools
import math
import os

import typing as tp
import numpy as np
import torch
import scipy
import particles

from spin_system import BaseSample, SpinSystem, Interaction, MultiOrientedSample
from spectra_manager import BaseSpectraCreator, StationarySpectraCreator


class SampleDict:
    def get_dicted_sample(self, sample, format_type: str = 'pytorch'):
        format_handlers = {
            'pytorch': self._get_pytorch,
            'npy': self._get_npy,
            'easyspin': self._get_easyspin
        }

        if format_type not in format_handlers:
            raise ValueError(f"Unsupported format: {format_type}")
        return format_handlers[format_type](sample)

    def _get_easyspin(self, sample: BaseSample):
        out = EasySpinSaverSampleDict().get_dict(sample)
        return out

    def _get_pytorch(self, sample: BaseSample):
        out_dict = self._get_base_dict(sample, serialize_for_torch=True)
        return out_dict

    def _get_npy(self, sample: BaseSample):
        out_dict = self._get_base_dict(sample, serialize_for_torch=False)
        return out_dict

    def _add_interaction_arrays(self, arrays_dict: tp.Dict, interactions, prefix: str):
        for i, interaction in enumerate(interactions):
            components = interaction.components.detach().cpu().numpy()
            arrays_dict[f'{prefix}_{i}_components'] = np.stack([components.real, components.imag])

            if interaction.strain is not None:
                strain = interaction.strain.detach().cpu().numpy()
                arrays_dict[f'{prefix}_{i}_strain'] = strain

    def _add_paired_interaction_arrays(self, arrays_dict: tp.Dict, paired_interactions, prefix: str):
        for idx_1, idx_2, interaction in paired_interactions:
            components = interaction.components.detach().cpu().numpy()
            arrays_dict[f'{prefix}_{idx_1}_{idx_2}_components'] = np.stack([components.real, components.imag])

            if interaction.strain is not None:
                strain = interaction.strain.detach().cpu().numpy()
                arrays_dict[f'{prefix}_{idx_1}_{idx_2}_strain'] = strain

    def _get_base_dict(self, sample: BaseSample, serialize_for_torch: bool = False) -> tp.Dict[str, tp.Any]:
        return {
            'spin_system': self._serialize_spin_system(sample.base_spin_system, serialize_for_torch),
            'gauss': self._convert_tensor(sample.gauss, serialize_for_torch),
            'lorentz': self._convert_tensor(sample.lorentz, serialize_for_torch),
            'base_ham_strain': self._convert_tensor(sample.base_ham_strain, serialize_for_torch),
        }

    def _serialize_spin_system(self, spin_system, serialize_for_torch: bool) -> tp.Dict[str, tp.Any]:
        serialize_interaction = (
            self._serialize_interaction_torch if serialize_for_torch
            else self._serialize_interaction_numpy
        )
        serialize_electron = (
            self._serialize_electron if serialize_for_torch
            else self._serialize_electron
        )
        serialize_nucleus = (
            self._serialize_nucleus if serialize_for_torch
            else self._serialize_nucleus
        )

        result = {
            'electrons': [serialize_electron(e) for e in spin_system.electrons],
            'nuclei': [serialize_nucleus(n) for n in spin_system.nuclei],
            'g_tensors': [serialize_interaction(g) for g in spin_system.g_tensors],
        }

        if serialize_for_torch:
            result.update({
                'electron_nuclei': spin_system.electron_nuclei,
                'electron_electron': spin_system.electron_electron,
                'nuclei_nuclei': spin_system.nuclei_nuclei,
            })
        else:
            result.update({
                'electron_nuclei': [
                    self._serialize_paired_interaction(pair)
                    for pair in spin_system.electron_nuclei
                ],
                'electron_electron': [
                    self._serialize_paired_interaction(pair)
                    for pair in spin_system.electron_electron
                ],
                'nuclei_nuclei': [
                    self._serialize_paired_interaction(pair)
                    for pair in spin_system.nuclei_nuclei
                ],
            })

        return result

    def _convert_tensor(self, tensor, serialize_for_torch: bool):
        """Convert tensor based on serialization format."""
        return tensor if serialize_for_torch else tensor.detach().cpu().numpy()

    def _serialize_paired_interaction(self, pair_interaction: tuple):
        """Serialize a paired interaction."""
        idx_1, idx_2, interaction = pair_interaction
        return {
            'indexes': (idx_1, idx_2),
            'interaction': self._serialize_interaction_numpy(interaction),
        }

    def _serialize_electron(self, particle) -> tp.Dict[str, float]:
        """Serialize electron particle (same for both formats)."""
        return {
            'spin': float(particle.spin),
        }

    def _serialize_nucleus(self, particle) -> tp.Dict[str, tp.Union[float, str]]:
        return {
            'spin': float(particle.spin),
            'g_factor': float(particle.g_factor),
            'name': particle.nucleus_str,
        }

    def _serialize_interaction_torch(self, interaction) -> tp.Dict[str, tp.Any]:
        return {
            'components': interaction.components,
            'strain': interaction.strain,
            'frame': interaction.frame,
        }

    def _serialize_interaction_numpy(self, interaction) -> tp.Dict[str, tp.Any]:
        return {
            'components': interaction.components.detach().cpu().numpy(),
            'strain': interaction.strain.detach().cpu().numpy() if interaction.strain is not None else np.nan,
            'frame': interaction.frame.detach().cpu().numpy() if interaction.frame is not None else np.nan,
        }



class EasySpinSaverSampleDict:
    hz_to_MHz = 1e-6
    T_to_mT = 1e3
    g_easy_spin_strain_converter = 2 * math.log(2)
    def get_dict(self, sample: BaseSample):
        out = self._serialize_sample(sample)
        return out

    def _serialize_sample(self, sample: BaseSample) -> tp.Dict[str, tp.Any]:
        spin_system = sample.base_spin_system

        lorentz = sample.lorentz.detach().cpu().item()
        gauss = sample.gauss.detach().cpu().item()
        ham_strain = self._convert_tensor(sample.base_ham_strain)

        sys_dict = self._serialize_spin_system(spin_system)
        sys_dict["HStrain"] = ham_strain * self.hz_to_MHz
        sys_dict["lw"] = np.array([gauss, lorentz]) * self.T_to_mT

        return sys_dict


    def _serialize_spin_system(self, spin_system: SpinSystem) -> tp.Dict[str, tp.Any]:

        electrons = spin_system.electrons
        g_tensors = spin_system.g_tensors
        nuclei = spin_system.nuclei
        electron_nuclei = spin_system.electron_nuclei
        electron_electron = spin_system.electron_electron
        nuclei_nuclei = spin_system.nuclei_nuclei
        result = {
            **self._serialize_electrons(electrons),
            **self._serialize_nuclei(nuclei),
            **self._serialize_g_tensors(g_tensors),
            **self._serialize_electron_nuclei(electrons, nuclei, electron_nuclei),
            **self._serialize_electron_electron(electrons, electron_electron),
            **self._serialize_nuclei_nuclei(nuclei, nuclei_nuclei),
        }

        return result

    def _serialize_electrons(self, electrons: list[particles.Electron]):
        return {"S": [electron.spin for electron in electrons]}

    def _serialize_nuclei(self, nuclei: list[particles.Nucleus]):
        if nuclei:

            return {"Nucs": ",".join([nucleus.nucleus_str for nucleus in nuclei])}
        else:
            return {}

    def _convert_tensor(self, tensor: torch.Tensor):
        return tensor.detach().cpu().numpy().astype(np.float64)

    def _serialize_g_tensors(self, g_interactions: list[Interaction]):
        g_tensors = []
        g_frames = []
        g_strains = []
        for g_interaction in g_interactions:
            g_tensors.append(self._convert_tensor(g_interaction.components))

            frame = g_interaction.frame
            frame = self._convert_tensor(frame) if frame is not None else np.array([0.0, 0.0, 0.0])
            g_frames.append(frame)

            strain = g_interaction.strain
            g_strain = self._convert_tensor(strain) if strain is not None else np.array([0.0, 0.0, 0.0])
            g_strains.append(g_strain)

        return {"g": np.array(g_tensors),
                "gStrain": np.array(g_strains) / self.g_easy_spin_strain_converter, "gFrame": np.array(g_frames)}

    def _serialize_electron_nuclei(self,
                                   electrons,
                                   nuclei: list[particles.Nucleus],
                                   electron_nuclei: list[tuple[int, int, Interaction]]
    ):

        num_electrons = len(electrons)
        num_nuclei = len(nuclei)

        tensors = np.zeros((num_nuclei, num_electrons * 3), dtype=np.float64)
        strains = np.zeros((num_nuclei, num_electrons * 3), dtype=np.float64)
        frames = np.zeros((num_nuclei, num_electrons * 3), dtype=np.float64)

        if electron_nuclei:

            interaction_dict = {}
            for el_idx, nuc_idx, interaction in electron_nuclei:
                interaction_dict[(el_idx, nuc_idx)] = interaction

            for el_idx in range(num_electrons):
                for nuc_idx in range(num_nuclei):

                    start_pos = el_idx * 3

                    if (el_idx, nuc_idx) in interaction_dict:
                        interaction = interaction_dict[(el_idx, nuc_idx)]

                        strain = interaction.strain
                        strain = self._convert_tensor(strain) if strain else [0, 0, 0]

                        tensors[nuc_idx, start_pos:start_pos + 3] = self._convert_tensor(interaction.components)
                        frames[nuc_idx, start_pos:start_pos + 3] = self._convert_tensor(interaction.frame)
                        strains[nuc_idx, start_pos:start_pos + 3] = strain
            return {"A": np.array(tensors) * self.hz_to_MHz,
                    # "AStrain": np.array(frames),    I didn't understand the AStrain logic in Easyspin. So I disolved it. It should be one or many or what.!!
                    "AFrame": np.array(strains) * self.hz_to_MHz}

        else:
            return {}

    def _serialize_electron_electron(self,
                                     electrons,
                                     electron_electron: list[tuple[int, int, Interaction]]
                                     ):

        zfs_flag = False

        num_electrons = len(electrons)
        J_tensor = np.zeros(int(num_electrons * (num_electrons - 1) / 2), dtype=np.float64)
        dipole_tensor = np.zeros((int(num_electrons * (num_electrons - 1) / 2), 3), dtype=np.float64)
        dipole_frame_tensor = np.zeros((int(num_electrons * (num_electrons - 1) / 2), 3), dtype=np.float64)

        zfs_array = np.zeros((int(num_electrons), 2), dtype=np.float64)
        zfs_frame = np.zeros((int(num_electrons), 3), dtype=np.float64)
        zfz_strain = np.zeros((int(num_electrons), 2), dtype=np.float64)

        coupling_dict = {}
        zero_field = {}
        for el_idx_1, el_idx_2, interaction in electron_electron:
            if el_idx_1 != el_idx_2:
                coupling_dict[(el_idx_1, el_idx_2)] = interaction
            else:
                zero_field[(el_idx_1, el_idx_2)] = interaction

        position_zfs = 0
        position_dip_dip = 0
        for el_idx_1 in range(num_electrons):
            for el_idx_2 in range(el_idx_1, num_electrons):

                if (el_idx_1, el_idx_2) in coupling_dict:
                    interaction = coupling_dict[(el_idx_1, el_idx_2)]

                    tensor = self._convert_tensor(interaction.components)
                    J = np.mean(tensor)
                    dip = tensor - J

                    J_tensor[position_dip_dip] = J
                    dipole_tensor[position_dip_dip] = dip
                    dipole_frame_tensor[position_dip_dip] = self._convert_tensor(interaction.frame)
                    position_dip_dip += 1

                if (el_idx_1, el_idx_2) in zero_field:
                    zfs_flag = True
                    interaction = zero_field[(el_idx_1, el_idx_2)]

                    tensor = self._convert_tensor(interaction.components)
                    frame = self._convert_tensor(interaction.frame)

                    strain = interaction.strain
                    strain = self._convert_tensor(strain) if strain is not None else [0, 0, 0]

                    D = 3 * tensor[-1] / 2
                    E = abs((tensor[0] - tensor[1]) / 2)

                    zfs_array[position_zfs] = np.array([D, E])
                    zfs_frame[position_zfs] = frame

                    D_str = 3 * strain[-1] / 2
                    E_str = abs(((strain[0] - strain[1]) / 2))

                    zfz_strain[position_zfs] = np.array([D_str, E_str])

                    position_zfs += 1

        out_dict = {}
        if zfs_flag:
            out_dict = {"D": np.array(zfs_array) * self.hz_to_MHz, "DFrame": np.array(zfs_frame),
                        "DStrain": np.array(zfz_strain) * self.hz_to_MHz}

        dipole_tensor = np.array(dipole_tensor)

        out_dict["dip"] = np.array(dipole_tensor) * self.hz_to_MHz
        out_dict["J"] = np.array(J_tensor) * self.hz_to_MHz
        out_dict["eeFrame"] = np.array(dipole_frame_tensor)

        return out_dict

    def _serialize_nuclei_nuclei(self, nuclei, nuclei_nuclei: list[tuple[int, int, Interaction]]):

        out_dict = {}
        num_nuclei = len(nuclei)
        if num_nuclei and nuclei_nuclei:
            Q_array = np.zeros(int(num_nuclei * (num_nuclei - 1) / 2), dtype=np.float64)
            frame_array = np.zeros((int(num_nuclei * (num_nuclei - 1) / 2), 3), dtype=np.float64)

            coupling_dict = {}
            for nuc_idx_1, nuc_idx_2, interaction in nuclei_nuclei:
                if nuc_idx_1 != nuc_idx_2:
                    coupling_dict[(nuc_idx_1, nuc_idx_2)] = interaction

            position = 0
            for nuc_idx_1 in range(num_nuclei):
                for nuc_idx_2 in range(nuc_idx_1 + 1, num_nuclei):

                    if (nuc_idx_1, nuc_idx_2) in coupling_dict:
                        interaction = coupling_dict[(nuc_idx_1, nuc_idx_2)]

                        Q_array[position] = self._convert_tensor(interaction.components)
                        frame_array[position] = self._convert_tensor(interaction.frame)

                    position += 1
            out_dict["Q"] = np.array(Q_array) * self.hz_to_MHz
            out_dict["QFrame"] = np.array(frame_array)

        return out_dict


class CreatorDict:
    hz_to_ghz = 1e-9
    def get_dicted_creator(self, creator: BaseSpectraCreator, format_type: str = 'pytorch'):
        format_handlers = {
            'pytorch': self._get_pytorch,
            'npy': self._get_npy,
            'easyspin': self._get_easyspin
        }

        if format_type not in format_handlers:
            raise ValueError(f"Unsupported format: {format_type}")
        return format_handlers[format_type](creator)

    def _get_easyspin(self, creator: BaseSpectraCreator):
        temperature = creator.intensity_calculator.temperature
        temperature = temperature if temperature is None else np.array(temperature).astype(np.float64)
        frequency = creator.resonance_parameter.detach().cpu().numpy().astype(np.float64)

        out_dict = {"Temperature": temperature, "mwFreq": frequency * self.hz_to_ghz}
        return out_dict

    def _get_pytorch(self, creator: BaseSpectraCreator):
        temperature = creator.intensity_calculator.temperature
        frequency = creator.resonance_parameter
        out_dict = {"temperature": temperature, "res_freq": frequency}
        return out_dict

    def _get_npy(self, creator: BaseSpectraCreator):
        temperature = creator.intensity_calculator.temperature
        temperature = temperature if temperature is None else np.array(temperature).astype(np.float64)
        frequency = creator.resonance_parameter.detach().cpu().numpy().astype(np.float64)

        out_dict = {"temperature": temperature, "res_freq": frequency}
        return out_dict


def save(
        filepath: str,
        sample: tp.Optional[BaseSample] = None,
        spectra_creator: tp.Optional[BaseSpectraCreator] = None,
        field: tp.Optional[tp.Union[torch.Tensor, np.ndarray]] = None,
        format_type="torch"
):
    """
    Save experimental and sample parameters
    :param filepath: The file path where data should be saved. Should include the desired
        file extension or directory path depending on format_type.
    :param sample: BaseSample instance. Default is None
    :param spectra_creator: SpectraCreate instance. Default is None
    :param field: The magnetic field torch/numpy array. Should be in Tesla units

    :param format_type: {'pytorch', 'npy', 'easyspin'}, optional
        The output format for saved data:

        - 'pytorch': Saves data as PyTorch tensors (.pt files)
        - 'npy': Saves data as NumPy arrays (.npy files)
        - 'easyspin': Creates EasySpin-compatible Sys and Exp parameter files

        Default is 'pytorch'.t
    :return: None
    """

    if format_type is None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pt' or ext == '.pth':
            format_type = 'pytorch'
        elif ext == '.npy':
            format_type = 'npy'
        elif ext == '.mat' or ext == '.npz':
            format_type = 'easyspin'
        else:
            raise ValueError(f"Cannot infer format from extension {ext}. Please specify format_type.")

    format_handlers = {
        'pytorch': save_torch,
        'npy': save_npy,
        'easyspin': save_npz
    }

    if format_type not in format_handlers:
        raise ValueError(f"Unsupported format: {format_type}")

    sample_dict = SampleDict().get_dicted_sample(sample, format_type) if sample is not None else {}
    creator_dict = CreatorDict().get_dicted_creator(spectra_creator, format_type) if spectra_creator is not None else {}

    return format_handlers[format_type](filepath, sample_dict, creator_dict, field)


def save_npy(filepath: str, sample_dict: dict[str, tp.Any], creator_dict: dict[str, tp.Any],
             field: tp.Optional[tp.Union[torch.Tensor, np.ndarray]] = None):
    out = {"sample": sample_dict}
    out["creator_dict"] = creator_dict
    out["field"] = parse_field(field)
    np.save(filepath, out)


def save_torch(filepath: str, sample_dict: dict, creator_dict: dict[str, tp.Any],
               field: tp.Optional[tp.Union[torch.Tensor, np.ndarray]] = None):
    out = {"sample": sample_dict}
    out["creator_dict"] = creator_dict
    out["field"] = parse_field(field)
    torch.save(out, filepath)


def save_npz(filepath: str, sample_dict: dict, creator_dict: dict[str, tp.Any],
             field: tp.Optional[tp.Union[torch.Tensor, np.ndarray]] = None):
    T_to_mT = 1e3
    out = {"Sys": sample_dict}
    out["Exp"] = creator_dict
    field_dict = parse_field(field)
    if field_dict:
        out["Exp"]["Range"] = np.array([field_dict["min_field"], field_dict["max_field"]], dtype=np.float64) * T_to_mT
        out["Exp"]["nPoints"] = field_dict["field_num"]
    scipy.io.savemat(filepath, out, oned_as='row')


def parse_field(field: tp.Optional[tp.Union[torch.Tensor, np.ndarray]] = None):
    if field is None:
        return {}
    out = {}
    out["max_field"] = max(field)
    out["min_field"] = min(field)
    out["field_num"] = len(field)
    return out


class SampleLoader:
    g_easy_spin_strain_converter = 2 * math.log(2)

    def load_sample_from_dict(self, sample_dict: dict, format_type: str = 'pytorch') -> BaseSample:
        format_handlers = {
            'pytorch': self._load_pytorch_sample,
            'npy': self._load_npy_sample,
            'easyspin': self._load_easyspin_sample
        }

        if format_type not in format_handlers:
            raise ValueError(f"Unsupported format: {format_type}")

        return format_handlers[format_type](sample_dict)

    def _load_pytorch_sample(self, sample_dict: dict) -> BaseSample:
        spin_system = self._deserialize_spin_system(sample_dict['spin_system'], torch_format=True)

        sample = MultiOrientedSample(
            spin_system=spin_system,
            gauss=sample_dict['gauss'],
            lorentz=sample_dict['lorentz'],
            ham_strain=sample_dict['ham_strain']
        )
        return sample

    def _load_npy_sample(self, sample_dict: dict) -> BaseSample:
        spin_system = self._deserialize_spin_system(sample_dict['spin_system'], torch_format=False)

        sample = MultiOrientedSample(
            spin_system=spin_system,
            gauss=torch.tensor(sample_dict['gauss'], dtype=torch.float32),
            lorentz=torch.tensor(sample_dict['lorentz'], dtype=torch.float32),
            ham_strain=sample_dict['ham_strain']
        )

        return sample

    def _load_easyspin_sample(self, sample_dict: dict) -> BaseSample:
        MHz_to_hz = 1e6
        mT_to_T = 1e-3

        lorentz_conversion = math.sqrt(2 * math.log(2))
        gauss_conversion = math.sqrt(3)

        if "lw" in sample_dict:
            lw = sample_dict.get('lw', np.array([[0.0, 0.0]]))
            gauss = torch.tensor(lw[0][0] * mT_to_T, dtype=torch.float32)
            lorentz = torch.tensor(lw[0][1] * mT_to_T, dtype=torch.float32)

        elif "lwpp" in sample_dict:
            lwpp = sample_dict.get('lwpp', np.array([[0.0, 0.0]]))
            gauss = torch.tensor(lwpp[0][0] * mT_to_T, dtype=torch.float32) * gauss_conversion
            lorentz = torch.tensor(lwpp[0][1] * mT_to_T, dtype=torch.float32) * lorentz_conversion


        spin_system = self._deserialize_easyspin_spin_system(sample_dict)

        if 'HStrain' in sample_dict:
            ham_strain = torch.tensor(
                sample_dict['HStrain'] * MHz_to_hz, dtype=torch.float32
            )[0]
        else:
            ham_strain = None

        sample = MultiOrientedSample(
            spin_system=spin_system,
            gauss=gauss,
            lorentz=lorentz,
            ham_strain=ham_strain
        )

        return sample

    def _deserialize_spin_system(self, sys_dict: dict, torch_format: bool) -> SpinSystem:
        """Deserialize spin system from pytorch/numpy format."""
        # Reconstruct particles
        electrons = [particles.Electron(spin=e['spin']) for e in sys_dict['electrons']]

        nuclei = [particles.Nucleus(nucleus_str=n['name']) for n in sys_dict['nuclei']]

        g_tensors = []
        for g_data in sys_dict['g_tensors']:
            g_tensors.append(self._deserialize_interaction(g_data, torch_format))

        spin_system = SpinSystem(electrons=electrons, nuclei=nuclei, g_tensors=g_tensors)

        if torch_format:
            spin_system.electron_nuclei = sys_dict.get('electron_nuclei', [])
            spin_system.electron_electron = sys_dict.get('electron_electron', [])
            spin_system.nuclei_nuclei = sys_dict.get('nuclei_nuclei', [])
        else:
            # In numpy format, these need to be reconstructed
            spin_system.electron_nuclei = [
                (pair['indexes'][0], pair['indexes'][1],
                 self._deserialize_interaction(pair['interaction'], torch_format))
                for pair in sys_dict.get('electron_nuclei', [])
            ]
            spin_system.electron_electron = [
                (pair['indexes'][0], pair['indexes'][1],
                 self._deserialize_interaction(pair['interaction'], torch_format))
                for pair in sys_dict.get('electron_electron', [])
            ]
            spin_system.nuclei_nuclei = [
                (pair['indexes'][0], pair['indexes'][1],
                 self._deserialize_interaction(pair['interaction'], torch_format))
                for pair in sys_dict.get('nuclei_nuclei', [])
            ]

        return spin_system

    def _deserialize_easyspin_spin_system(self, sys_dict: dict) -> SpinSystem:
        MHz_to_hz = 1e6

        S_list = sys_dict.get('S', [[]])
        electrons = [particles.Electron(spin=s) for s in itertools.chain(*S_list)]

        nuclei = []
        if 'Nucs' in sys_dict:

            for nuc_str in sys_dict['Nucs'].tolist()[0].split(","):
                nucleus = particles.Nucleus(nuc_str)  # Assuming this method exists
                nuclei.append(nucleus)

        g_tensors = []
        if 'g' in sys_dict:
            g_components = sys_dict['g']
            g_strains = sys_dict.get('gStrain', np.zeros_like(g_components)) * self.g_easy_spin_strain_converter
            g_frames = sys_dict.get('gFrame', np.zeros_like(g_components))

            for i in range(len(g_components)):
                components = torch.tensor(g_components[i], dtype=torch.float32)
                strain = torch.tensor(g_strains[i], dtype=torch.float32) if np.any(g_strains[i]) else None
                frame = torch.tensor(g_frames[i], dtype=torch.float32) if np.any(g_frames[i]) else None

                g_tensors.append(Interaction(components=components, strain=strain, frame=frame))

        spin_system = SpinSystem(electrons=electrons, nuclei=nuclei, g_tensors=g_tensors)

        if 'A' in sys_dict:
            A_tensor = sys_dict['A'] * MHz_to_hz
            A_strains = sys_dict.get('AStrain', np.zeros_like(A_tensor))
            A_frames = sys_dict.get('AFrame', np.zeros_like(A_tensor))

            electron_nuclei = []
            for el_idx in range(len(electrons)):
                for nuc_idx in range(len(nuclei)):
                    start_pos = nuc_idx * 3
                    components = A_tensor[nuc_idx, start_pos:start_pos + 3]

                    if np.any(components):  # Only add if non-zero
                        strain_vals = A_strains[nuc_idx, start_pos:start_pos + 3] * MHz_to_hz
                        frame_vals = A_frames[nuc_idx, start_pos:start_pos + 3]

                        strain = torch.tensor(strain_vals, dtype=torch.float32) if np.any(strain_vals) else None
                        frame = torch.tensor(frame_vals, dtype=torch.float32) if np.any(frame_vals) else None

                        interaction = Interaction(
                            components=torch.tensor(components, dtype=torch.float32),
                            strain=strain,
                            frame=frame
                        )
                        electron_nuclei.append((el_idx, nuc_idx, interaction))

            spin_system.electron_nuclei = electron_nuclei

        electron_electron = []
        if 'J' in sys_dict or 'dip' in sys_dict or 'D' in sys_dict:
            self._add_easyspin_electron_electron(sys_dict, spin_system, electron_electron)

        if 'Q' in sys_dict:
            Q_tensor = sys_dict['Q'] * MHz_to_hz
            Q_frames = sys_dict.get('QFrame', np.zeros((len(Q_tensor), 3)))

            nuclei_nuclei = []
            position = 0
            for nuc_idx_1 in range(len(nuclei)):
                for nuc_idx_2 in range(nuc_idx_1 + 1, len(nuclei)):
                    if position < len(Q_tensor) and Q_tensor[position] != 0:
                        components = torch.tensor(Q_tensor[position], dtype=torch.float32)
                        frame = torch.tensor(Q_frames[position], dtype=torch.float32) if np.any(
                            Q_frames[position]) else None

                        interaction = Interaction(components=components, frame=frame)
                        nuclei_nuclei.append((nuc_idx_1, nuc_idx_2, interaction))
                    position += 1

            spin_system.nuclei_nuclei = nuclei_nuclei

        return spin_system

    def _add_easyspin_electron_electron(self, sys_dict: dict[str, tp.Any],
                                        spin_system: SpinSystem, electron_electron: list[tuple[int, int, Interaction]]):
        """Helper to add electron-electron interactions from EasySpin format."""
        MHz_to_hz = 1e6
        num_electrons = len(spin_system.electrons)

        J_tensor = sys_dict.get('J', 0.0)
        dip_tensor = sys_dict.get('dip', np.zeros((len(J_tensor), 3)) if len(J_tensor) > 0 else np.array([]))
        ee_frames = sys_dict.get('eeFrame', np.zeros_like(dip_tensor))

        D_tensor = sys_dict.get('D', np.array([]))
        D_frames = sys_dict.get('DFrame', np.zeros((len(D_tensor), 3)) if len(D_tensor) > 0 else np.array([]))
        D_strains = sys_dict.get('DStrain', np.zeros_like(D_tensor))

        position = 0
        for el_idx_1 in range(num_electrons):
            for el_idx_2 in range(el_idx_1 + 1, num_electrons):
                if position < len(J_tensor):
                    J = J_tensor[position] * MHz_to_hz
                    dip = dip_tensor[position] * MHz_to_hz if position < len(dip_tensor) else np.zeros(3)
                    frame = ee_frames[position] if position < len(ee_frames) else np.zeros(3)

                    components = torch.tensor(J + dip, dtype=torch.float32)
                    frame_tensor = torch.tensor(frame, dtype=torch.float32) if np.any(frame) else None
                    interaction = Interaction(components=components, frame=frame_tensor)
                    electron_electron.append((el_idx_1, el_idx_2, interaction))

                if position < len(D_tensor) and np.any(D_tensor[position]):
                    D_val = D_tensor[position] * MHz_to_hz
                    D_frame = D_frames[position] if position < len(D_frames) else np.zeros(3)
                    D_strain = D_strains[position] * MHz_to_hz if position < len(D_strains) else np.zeros(3)

                    components = torch.tensor(D_val, dtype=torch.float32)
                    frame_tensor = torch.tensor(D_frame, dtype=torch.float32) if np.any(D_frame) else None
                    strain_tensor = torch.tensor(D_strain, dtype=torch.float32) if np.any(D_strain) else None

                    interaction = Interaction(components=components, frame=frame_tensor, strain=strain_tensor)
                    electron_electron.append((el_idx_1, el_idx_1, interaction))
                position += 1
        spin_system.electron_electron = electron_electron

    def _deserialize_interaction(self, interaction_data: dict, torch_format: bool) -> Interaction:
        if torch_format:
            return Interaction(
                components=interaction_data['components'],
                strain=interaction_data['strain'],
                frame=interaction_data['frame']
            )
        else:
            components = torch.tensor(interaction_data['components'], dtype=torch.float32)

            strain = interaction_data['strain']
            if isinstance(strain, np.ndarray) and not np.isnan(strain).all():
                strain = torch.tensor(strain, dtype=torch.float32)
            else:
                strain = None

            frame = interaction_data['frame']
            if isinstance(frame, np.ndarray) and not np.isnan(frame).all():
                frame = torch.tensor(frame, dtype=torch.float32)
            else:
                frame = None

            return Interaction(components=components, strain=strain, frame=frame)


class CreatorLoader:
    """Reconstructs BaseSpectraCreator objects from dictionary representations."""

    def load_creator_from_dict(self, sample: MultiOrientedSample, creator_dict: dict, format_type: str = 'pytorch') ->\
            BaseSpectraCreator:
        """Load a BaseSpectraCreator from a dictionary based on format type."""
        format_handlers = {
            'pytorch': self._load_pytorch_creator,
            'npy': self._load_npy_creator,
            'easyspin': self._load_easyspin_creator
        }


        if format_type not in format_handlers:
            raise ValueError(f"Unsupported format: {format_type}")

        return format_handlers[format_type](sample, creator_dict)

    def _load_pytorch_creator(self, sample: MultiOrientedSample, creator_dict: dict) -> BaseSpectraCreator:
        """Load creator from pytorch format."""
        return StationarySpectraCreator(
            sample=sample,
            freq=creator_dict['res_freq'],
            temperature=creator_dict['temperature'],
        )

    def _load_npy_creator(self, sample: MultiOrientedSample, creator_dict: dict) -> BaseSpectraCreator:
        """Load creator from numpy format."""
        temperature = creator_dict['temperature']
        res_freq = torch.tensor(creator_dict['res_freq'], dtype=torch.float32)

        return StationarySpectraCreator(sample=sample, temperature=temperature, freq=res_freq)

    def _load_easyspin_creator(self, sample: MultiOrientedSample, creator_dict: dict) -> BaseSpectraCreator:
        """Load creator from EasySpin format."""
        ghz_to_hz = 1e9

        temperature = creator_dict.get('Temperature')
        frequency = creator_dict['mwFreq'] * ghz_to_hz

        return StationarySpectraCreator(
            sample=sample,
            temperature=temperature,
            freq=torch.tensor(frequency, dtype=torch.float32)
        )


def load(filepath: str, format_type: str = None) -> tp.Dict[str, tp.Any]:
    """
    :param filepath: load data from filepath
    :param format_type: {'pytorch', 'npy', 'easyspin'}, optional
        The output format for saved data:

        - 'pytorch': Load data from .pt files
        - 'npy': Load data from .npy files
        - 'easyspin': Load EasySpine data. EasySpin out should contain Sys and Exp structures
        Default is 'pytorch'.
    :return: dict of loaded data: sample, spectracreator, field
    """

    if format_type is None:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.pt' or ext == '.pth':
            format_type = 'pytorch'
        elif ext == '.npy':
            format_type = 'npy'
        elif ext == '.mat' or ext == '.npz':
            format_type = 'easyspin'
        else:
            raise ValueError(f"Cannot infer format from extension {ext}. Please specify format_type.")

    format_handlers = {
        'pytorch': load_torch,
        'npy': load_npy,
        'easyspin': load_mat
    }

    if format_type not in format_handlers:
        raise ValueError(f"Unsupported format: {format_type}")

    return format_handlers[format_type](filepath)


def load_torch(filepath: str) -> tp.Dict[str, tp.Any]:
    data = torch.load(filepath, map_location='cpu')

    result = {}

    if 'sample' in data and data['sample']:
        sample_loader = SampleLoader()
        result['sample'] = sample_loader.load_sample_from_dict(data['sample'], 'pytorch')
    else:
        raise KeyError("Can not instance System Creator without sample example")

    if 'creator_dict' in data and data['creator_dict']:
        creator_loader = CreatorLoader()
        result['creator'] = creator_loader.load_creator_from_dict(result['sample'], data['creator_dict'], 'pytorch')

    if 'field' in data and data['field']:
        result['field'] = reconstruct_field(data['field'])

    return result


def load_npy(filepath: str) -> tp.Dict[str, tp.Any]:
    data = np.load(filepath, allow_pickle=True).item()

    result = {}

    if 'sample' in data and data['sample']:
        sample_loader = SampleLoader()
        result['sample'] = sample_loader.load_sample_from_dict(data['sample'], 'npy')

    if 'creator_dict' in data and data['creator_dict']:
        creator_loader = CreatorLoader()
        result['creator'] = creator_loader.load_creator_from_dict(result['sample'], data['creator_dict'], 'npy')

    if 'field' in data and data['field']:
        result['field'] = reconstruct_field(data['field'])

    return result


def load_mat_file(filepath):
    """Load and properly parse MATLAB .mat file with structured arrays"""
    data = scipy.io.loadmat(filepath)
    clean_data = {k: v for k, v in data.items() if not k.startswith('__')}
    result = {}
    for key, value in clean_data.items():
        if hasattr(value, 'dtype') and value.dtype.names:
            structured_dict = {}
            if value.size > 0:
                first_element = value.flat[0]

                for field_name in value.dtype.names:
                    field_data = first_element[field_name]
                    if isinstance(field_data, np.ndarray):
                        structured_dict[field_name] = field_data
                    else:
                        structured_dict[field_name] = field_data
            result[key] = structured_dict
        else:
            result[key] = value
    return result


def load_mat(filepath: str) -> tp.Dict[str, tp.Any]:
    data = load_mat_file(filepath)

    result = {}

    if 'Sys' in data:
        sample_loader = SampleLoader()
        result['sample'] = sample_loader.load_sample_from_dict(data['Sys'], 'easyspin')

    if 'Exp' in data:
        exp_data = data['Exp']

        if 'Range' in exp_data and 'nPoints' in exp_data:
            mT_to_T = 1e-3
            field_range = exp_data['Range'] * mT_to_T
            n_points = exp_data['nPoints'][0][0]
            field = np.linspace(field_range[0][0], field_range[0][1], n_points)
            result['field'] = field

        creator_loader = CreatorLoader()
        result['creator'] = creator_loader.load_creator_from_dict(result['sample'], exp_data, 'easyspin')
    return result


def reconstruct_field(field_dict: dict) -> np.ndarray | None:
    if not field_dict:
        return None

    min_field = field_dict['min_field']
    max_field = field_dict['max_field']
    field_num = field_dict['field_num']

    return np.linspace(min_field, max_field, field_num)