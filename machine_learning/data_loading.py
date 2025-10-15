import typing as tp
import pathlib
import sys
import pickle
import random
import math
sys.path.append("..")

import torch
import safetensors
import torch.nn as nn

import spin_system
import constants
import particles
import spectra_manager
import tqdm

from .data_generation import SpinSystemStructure

from .transforms import ComponentsAnglesTransform, SpectraModifier,\
    SpecTransformField, BroadTransform, SpecTransformSpecIntensity, SpinTranform


class SampleGraphData:
    """
    Convert Sample to Graph Data.
    1. Each interaction is Node.
    2. Each interaction is characterized by components and angles. For Zeeman interaction the components is multiplied
    by MAGNETRON / Plank
    3. Each node has type: ELECTRON_TYPE, NUCLEI_TYPE, INTRA_TYPE
    """
    ELECTRON_TYPE = 0
    NUCLEI_TYPE = 1
    INTRA_TYPE = 2
    def _parse_el_el(self, base_spin_system: spin_system.SpinSystem):
        indices = base_spin_system.ee_indices
        interactions = base_spin_system.electron_electron_interactions
        destination_shift = 0
        source_shift = len(base_spin_system.electrons) + len(base_spin_system.nuclei)
        return self._parse_inter_interaction(source_shift, destination_shift, destination_shift, interactions, indices)

    def _parse_el_nuc(self, base_spin_system: spin_system.SpinSystem):
        indices = base_spin_system.en_indices
        interactions = base_spin_system.electron_nuclei_interactions
        idx_1_shift = 0
        idx_2_shift = len(base_spin_system.nuclei)
        source_shift = len(base_spin_system.electrons) + len(base_spin_system.nuclei) + len(
            base_spin_system._ee_indices)
        return self._parse_inter_interaction(source_shift, idx_1_shift, idx_2_shift, interactions, indices)

    def _parse_nuc_nuc(self, base_spin_system: spin_system.SpinSystem):
        indices = base_spin_system.nn_indices
        interactions = base_spin_system.nuclei_nuclei_interactions
        destination_shift = len(base_spin_system.nuclei)
        source_shift = len(base_spin_system.electrons) + len(base_spin_system.nuclei) + len(
            base_spin_system.ee_indices) + len(base_spin_system.en_indices)
        return self._parse_inter_interaction(source_shift, destination_shift, destination_shift, interactions, indices)

    def _parse_inter_interaction(self, source_shift: int, idx_1_shift: int,
                                 idx_2_shift: int, inter_interactions: nn.ModuleList, indices: list[tuple[int, int]]):
        components = []
        angles = []
        destination = []
        source = []
        for shift, ((idx_1, idx_2), inter) in enumerate(zip(indices, inter_interactions)):
            components.append(inter.components)
            angles.append(inter.frame)
            if idx_1 == idx_2:
                source.append(shift + source_shift)
                destination.append(idx_1 + idx_1_shift)
            else:
                source.append(shift + source_shift)
                destination.append(idx_1 + idx_1_shift)
                source.append(shift + source_shift)
                destination.append(idx_2 + idx_2_shift)
        return components, angles, destination, source

    def _parse_electrons(self, base_spin_system: spin_system.SpinSystem):
        g_tensors = base_spin_system.g_tensors
        electrons = base_spin_system.electrons
        components = []
        angles = []
        spins = []
        for electron, g_interaction in zip(electrons, g_tensors):
            components.append(g_interaction.components * constants.BOHR / constants.PLANCK)
            angles.append(g_interaction.frame)
            spins.append(electron.spin)
        return components, angles, spins

    def _parse_nuclei(self, base_spin_system: spin_system.SpinSystem):
        nuclei = base_spin_system.nuclei
        components = []
        spins = []
        for nucelus in nuclei:
            components.append(torch.full((*base_spin_system.config_shape, 3),
                                         nucelus.g_factor * constants.NUCLEAR_MAGNETRON / constants.PLANCK))
            spins.append(nucelus.spin)
        angles = [torch.zeros((*base_spin_system.config_shape, 3), dtype=torch.float32,
                              device=base_spin_system.device)] * len(nuclei)
        return components, angles, spins

    def _parse_base_spin_system(self, base_spin_system: spin_system.SpinSystem):
        components_el, angles_el, spins_el = self._parse_electrons(base_spin_system)
        components_nuc, angles_nuc, spins_nuc = self._parse_nuclei(base_spin_system)
        spins = torch.tensor([*spins_el, *spins_nuc])

        components_el_el, angles_el_el, destination_el_el, source_el_el = self._parse_el_el(base_spin_system)
        components_el_nuc, angles_el_nuc, destination_el_nuc, source_el_nuc = self._parse_el_nuc(base_spin_system)
        components_nuc_nuc, angles_nuc_nuc, destination_nuc_nuc, source_nuc_nuc = self._parse_nuc_nuc(base_spin_system)

        components = torch.stack(
            (*components_el, *components_nuc, *components_el_el, *components_el_nuc, *components_nuc_nuc), dim=-0)
        angles = torch.stack((*angles_el, *angles_nuc, *angles_el_el, *angles_el_nuc, *angles_nuc_nuc), dim=-0)
        destinations = destination_el_el + destination_el_nuc + destination_nuc_nuc
        source = source_el_el + source_el_nuc + source_nuc_nuc

        types = [self.ELECTRON_TYPE] * len(base_spin_system.electrons) + [self.NUCLEI_TYPE] * len(
            base_spin_system.nuclei) + \
                [self.INTRA_TYPE] * (len(base_spin_system.en_indices) + len(base_spin_system.ee_indices) + len(
            base_spin_system.nn_indices))
        types = torch.tensor(types, base_spin_system.device)
        spins = spins.view(-1, *[1] * len(base_spin_system.config_shape)).expand(-1, *base_spin_system.config_shape)
        types = types.view(-1, *[1] * len(base_spin_system.config_shape)).expand(-1, *base_spin_system.config_shape)
        return components, angles, destinations, source, spins, types

    def __call__(self, sample: spin_system.BaseSample, temperature: torch.Tensor = torch.tensor([[300], [300]])):
        hum_strain = sample.base_ham_strain
        lorentz = sample.lorentz
        gauss = sample.gauss

        node_data = torch.cat((hum_strain, lorentz, gauss, temperature), dim=-1)
        base_spin_system = sample.base_spin_system
        components, angles, destinations, source, spins, types = self._parse_base_spin_system(base_spin_system)
        return {
            "components": components,
            "angles": angles,
            "destinations": destinations,
            "source": source,
            "spins": spins,
            "types": types,
            "node_data": node_data
        }


class FileParser:
    ELECTRON_TYPE = 0
    NUCLEI_TYPE = 1
    INTRA_TYPE = 2

    def open_structure(self, path_structure: tp.Union[str, pathlib.Path]):
        path_structure = pathlib.Path(path_structure)
        with open(path_structure / "generator_summary.pkl", 'rb') as f:
            generator_summary = pickle.load(f)
        with open(path_structure / "structure.pkl", 'rb') as f:
            structure = pickle.load(f)
        return structure, generator_summary

    def _parse_electrons(self, structure: SpinSystemStructure, generation_summary: dict[str, tp.Any],
                               sample_meta: dict[str, tp.Any], generation_data: dict[str, torch.Tensor]):
        num_temperature_points = generation_summary["num_temperature_points"]
        num_hamiltonian_strains = generation_summary["num_hamiltonian_strains"]
        batch_size = generation_summary["batch_size"]

        electrons_spins = structure.electrons_spins

        g_tensor_components = generation_data["g_tensor_components"]

        g_tensor_components = generation_data["g_tensor_components"].expand(
            num_hamiltonian_strains, num_temperature_points, *g_tensor_components.shape,
        )

        g_tensor_orientations = generation_data["g_tensor_orientations"]
        g_tensor_orientations = generation_data["g_tensor_orientations"].expand(
            num_hamiltonian_strains, num_temperature_points, *g_tensor_orientations.shape,
        )

        return electrons_spins, g_tensor_components, g_tensor_orientations

    def _parse_nuclei(self, structure: SpinSystemStructure, generation_summary: dict[str, tp.Any],
                               sample_meta: dict[str, tp.Any], generation_data: dict[str, torch.Tensor]):
        nuclei = structure.nuclei
        return nuclei

    def _parse_electron_electron(self, structure: SpinSystemStructure, generation_summary: dict[str, tp.Any],
                               sample_meta: dict[str, tp.Any], generation_data: dict[str, torch.Tensor]):
        num_temperature_points = generation_summary["num_temperature_points"]
        num_hamiltonian_strains = generation_summary["num_hamiltonian_strains"]
        batch_size = generation_summary["batch_size"]

        electron_electron_indexes = structure.electron_electron

        exchange_dipolar_pairs = sample_meta["exchange_dipolar_pairs"]
        dipolar_components = generation_data["dipolar_components"]
        dipolar_components = dipolar_components.expand(
            num_hamiltonian_strains, num_temperature_points, *dipolar_components.shape,
        )

        zfs_pairs = sample_meta["zfs_pairs"]
        zfs_components = generation_data["zfs_components"]
        zfs_components = zfs_components.expand(
            num_hamiltonian_strains, num_temperature_points, *zfs_components.shape,
        )

        electron_angles = generation_data["electron_electron_orientations"]
        electron_angles = electron_angles.expand(
            num_hamiltonian_strains, num_temperature_points, *electron_angles.shape,
        )

        electron_electron_components = torch.cat((dipolar_components, zfs_components), dim=-2)
        return electron_electron_components, electron_angles, exchange_dipolar_pairs + zfs_pairs

    def _parse_electron_nuclei(self, structure: SpinSystemStructure, generation_summary: dict[str, tp.Any],
                               sample_meta: dict[str, tp.Any],
                               generation_data: dict[str, torch.Tensor]):
        num_temperature_points = generation_summary["num_temperature_points"]
        num_hamiltonian_strains = generation_summary["num_hamiltonian_strains"]
        batch_size = generation_summary["batch_size"]

        electron_nucleus_pairs = sample_meta["electron_nucleus_pairs"]

        hyperfine_coupling_orientations = generation_data["hyperfine_coupling_orientations"]
        hyperfine_coupling_orientations = hyperfine_coupling_orientations.expand(
            num_hamiltonian_strains, num_temperature_points, *hyperfine_coupling_orientations.shape,
        )

        hyperfine_coupling_components = generation_data["hyperfine_coupling_components"]
        hyperfine_coupling_components = hyperfine_coupling_components.expand(
            num_hamiltonian_strains, num_temperature_points, *hyperfine_coupling_components.shape,
        )

        return hyperfine_coupling_components, hyperfine_coupling_orientations, electron_nucleus_pairs

    def _parse_file(
            self, structure: SpinSystemStructure,
            generation_summary: dict[str, tp.Any], path: tp.Union[str, pathlib.Path]
    ):
        path = pathlib.Path(path)
        with open(path / "sample_meta.pkl", 'rb') as f:
            sample_meta = pickle.load(f)
        generation_data = {}
        with safetensors.safe_open(path / "generation_data.safetensors", framework="pt") as f:
            for k in f.keys():
                generation_data[k] = f.get_tensor(k)

        num_temperature_points = generation_summary["num_temperature_points"]
        num_hamiltonian_strains = generation_summary["num_hamiltonian_strains"]
        batch_size = generation_summary["batch_size"]

        spec = generation_data["out"]
        max_field_pos = generation_data["max_field_pos"]
        min_field_pos = generation_data["min_field_pos"]

        #steps = torch.linspace(0, 1, spec.shape[-1])
        #fields = steps * (max_field_pos - min_field_pos).unsqueeze(-1) + min_field_pos.unsqueeze(-1)

        ham_strain = generation_data["ham_strain"].expand(
            num_hamiltonian_strains, num_temperature_points, batch_size, 3
        )

        temperatures = generation_data["temperatures"].expand(
            num_hamiltonian_strains, num_temperature_points, batch_size, 1
        )
        freq = generation_data["freq"]

        electrons_spins, g_tensor_components, g_tensor_orientations = self._parse_electrons(structure,
                                                                                            generation_summary,
                                                                                            sample_meta,
                                                                                            generation_data)
        nuclei = self._parse_nuclei(structure, generation_summary, sample_meta, generation_data)
        electron_electron_components, electron_angles, electron_pairs = self._parse_electron_electron(structure,
                                                                                                      generation_summary,
                                                                                                      sample_meta,
                                                                                                      generation_data)
        hyperfine_coupling_components, hyperfine_coupling_orientations, hyperfine_pairs = self._parse_electron_nuclei(
            structure, generation_summary, sample_meta, generation_data)

        return ((min_field_pos, max_field_pos), spec, freq, ham_strain, temperatures), (
        electrons_spins, g_tensor_components, g_tensor_orientations), nuclei, (
        electron_electron_components, electron_angles, electron_pairs), (
            hyperfine_coupling_components, hyperfine_coupling_orientations, hyperfine_pairs)

    def to_sample_data(self, structure: SpinSystemStructure, generation_summary: dict[str, tp.Any],
                       path: tp.Union[str, pathlib.Path],
                       lorentz: tp.Optional[torch.Tensor], gauss: tp.Optional[torch.Tensor]):

        ((min_field_pos, max_field_pos), spec, freq, ham_strain, temperatures), (
            electrons_spins, g_tensor_components, g_tensor_orientations), nuclei, (
            electron_electron_components, electron_angles, electron_pairs), (
            hyperfine_coupling_components, hyperfine_coupling_orientations, hyperfine_pairs)\
            = self._parse_file(structure, generation_summary, path)

        g_tensors = []
        for components, orientations in zip(
                g_tensor_components.movedim(-2, 0), g_tensor_orientations.movedim(-2, 0)
        ):
            g_tensors.append(spin_system.Interaction(components=components, frame=orientations))
        el_el = []
        for pairs, components, orientations in zip(
                electron_pairs, electron_electron_components.movedim(-2, 0), electron_angles.movedim(-2, 0)
        ):
            el_el.append((pairs[0], pairs[1], spin_system.Interaction(components=components, frame=orientations)))

        el_nuc = []
        for pairs, components, orientations in zip(
                hyperfine_pairs, hyperfine_coupling_components.movedim(-2, 0),
                hyperfine_coupling_orientations.movedim(-2, 0)
        ):
            el_nuc.append((pairs[0], pairs[1], spin_system.Interaction(components=components, frame=orientations)))

        base_spin_system = spin_system.SpinSystem(
            electrons=electrons_spins,
            g_tensors=g_tensors,
            nuclei=nuclei,
            electron_nuclei=el_nuc,
            electron_electron=el_el,
        )
        sample = spin_system.MultiOrientedSample(spin_system=base_spin_system, ham_strain=ham_strain,
                                                 lorentz=lorentz, gauss=gauss)

        return {
            "sample": sample,
            "min_field_pos": min_field_pos,
            "max_field_pos": max_field_pos,
            "temperatures": temperatures,
            "spec": spec,
            "freq": freq
        }

    def to_graph_data(self, structure: SpinSystemStructure, generation_summary: dict[str, tp.Any],
                       path: tp.Union[str, pathlib.Path],
                       lorentz: tp.Optional[torch.Tensor], gauss: tp.Optional[torch.Tensor]):

        ((min_field_pos, max_field_pos), spec, freq, ham_strain, temperatures), (
            electrons_spins, g_tensor_components, g_tensor_orientations), nuclei, (
            electron_electron_components, electron_angles, electron_pairs), (
            hyperfine_coupling_components, hyperfine_coupling_orientations, hyperfine_pairs)\
            = self._parse_file(structure, generation_summary, path)

        num_electrons = len(electrons_spins)
        num_nuclei = len(nuclei)
        num_ee_interactions = len(electron_pairs)
        num_en_interactions = len(hyperfine_pairs)
        num_nn_interactions = 0

        components_list = []
        angles_list = []

        for g_comp, g_orient in zip(g_tensor_components.movedim(-2, 0), g_tensor_orientations.movedim(-2, 0)):
            components_list.append(g_comp * constants.BOHR / constants.PLANCK)
            angles_list.append(g_orient)

        config_shape = ham_strain.shape[:-1]  # Get config shape from hamiltonian_strain
        device = ham_strain.device
        nuclei = [particles.Nucleus(nucleus) for nucleus in nuclei]

        for nucleus in nuclei:
            components_list.append(
                torch.full((*config_shape, 3), nucleus.g_factor * constants.NUCLEAR_MAGNETRON / constants.PLANCK))
            angles_list.append(torch.zeros((*config_shape, 3), dtype=torch.float32, device=device))

        for comp, angle in zip(
                electron_electron_components.movedim(-2, 0), electron_angles.movedim(-2, 0)
        ):
            components_list.append(comp)
            angles_list.append(angle)

        for comp, angle in zip(
                hyperfine_coupling_components.movedim(-2, 0), hyperfine_coupling_orientations.movedim(-2, 0)
        ):
            components_list.append(comp)
            angles_list.append(angle)

        components = torch.stack(components_list, dim=0)
        angles = torch.stack(angles_list, dim=0)

        destinations = []
        sources = []

        # Electron-electron interactions
        for i, (idx1, idx2) in enumerate(electron_pairs):
            source_idx = i + num_electrons + num_nuclei
            if idx1 == idx2:
                sources.append(source_idx)
                destinations.append(idx1)
            else:
                sources.append(source_idx)
                destinations.append(idx1)
                sources.append(source_idx)
                destinations.append(idx2)

        for i, (idx1, idx2) in enumerate(hyperfine_pairs):
            source_idx = i + num_electrons + num_nuclei + num_ee_interactions
            sources.append(source_idx)
            destinations.append(idx1)  # electron index
            sources.append(source_idx)
            destinations.append(idx2 + num_electrons)  # nuclei index (shifted)

        electron_spins = [spin for spin in electrons_spins]
        nuclei_spins = [nucleus.spin for nucleus in nuclei]
        spins = torch.tensor([*electron_spins, *nuclei_spins])

        types = ([self.ELECTRON_TYPE] * num_electrons +
                 [self.NUCLEI_TYPE] * num_nuclei +
                 [self.INTRA_TYPE] * (num_ee_interactions + num_en_interactions + num_nn_interactions))
        types = torch.tensor(types)

        spins = spins.view(-1, *[1] * len(config_shape)).expand(-1, *config_shape)
        types = types.view(-1, *[1] * len(config_shape)).expand(-1, *config_shape)

        return {
            "components": components,
            "angles": angles,
            "destinations": destinations,
            "source": sources,
            "spins": spins,
            "types": types,
            "ham_strain": ham_strain,
            "lorentz": lorentz,
            "gauss": gauss,
            "temperatures": temperatures,
            "min_field_pos": min_field_pos,
            "max_field_pos": max_field_pos,
            "spec": spec,
            "freq": freq
        }


class EPRDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, rng_generator=random.Random(None)):
        self.components_angles_transform = ComponentsAnglesTransform()
        self.spectra_modifier = SpectraModifier(rng_generator)
        self.file_parser = FileParser()
        self.root_dir = pathlib.Path(root_dir)

        self._structure_info = self._scan_directory()
        self._structure_data = self._get_samples(self._structure_info)

        self._spectra_modifier = SpectraModifier(rng_generator=rng_generator)
        self._spectra_g_transform = SpecTransformField()
        self._broad_transform = BroadTransform()
        self._spectra_transform_intensity = SpecTransformSpecIntensity()

    def _get_samples(self, _structure_info: list[dict[str, tp.Any]]):
        self.cache = []
        for idx in tqdm.tqdm(range(len(_structure_info))):
            self.cache.append(self._load_sample(idx))

    def _scan_directory(self) -> list[tp.Dict[str, str]]:
        _samples_data = []
        structure_dirs = sorted(self.root_dir.glob("structure_*"))
        structure_data = []
        num_nodes = 0
        for structure_dir in structure_dirs:
            structure_id = structure_dir.name
            structure, generator_summary = self.file_parser.open_structure(
                str(structure_dir)
            )
            mean_dirs = sorted(structure_dir.glob("mean_*"))

            mean_data = []
            for mean_dir in mean_dirs:
                mean_id = mean_dir.name
                sample_dirs = sorted(mean_dir.glob("sample_*"))

                sample_data = []
                for sample_dir in sample_dirs:
                    sample_id = sample_dir.name
                    num_nodes = structure.num_electrons + structure.num_nuclei + \
                                len(structure.electron_nuclei) + len(structure.electron_electron) + len(
                        structure.nuclei_nuclei)

                    sample_data.append({
                        "sample_path": str(sample_dir),
                        "mean_id": mean_id,
                        "sample_id": sample_id,
                    })
                mean_data.append(sample_data)

            structure_data.append({
                "structure_path": str(structure_dir),
                "structure_id": structure_id,
                "num_nodes": num_nodes,
                "mean_data": mean_data
            }
            )

            return structure_data

    def __len__(self) -> int:
        return len(self._structure_info)

    def _prepare_out(self, structure_out):
        ham_strain = structure_out.pop("ham_strain")

        spec_out = self._spectra_modifier(
            structure_out.pop("min_field_pos"), structure_out.pop("max_field_pos"), structure_out.pop("spec"),
            ham_strain
        )
        g_feature, freq_feature = self._spectra_g_transform(spec_out["field"], structure_out["freq"])
        spec = self._spectra_transform_intensity(spec_out["spec"])
        spec_distorted = self._spectra_transform_intensity(spec_out["spec_distorted"])
        broad_features = self._broad_transform(ham_strain, spec_out.pop("lorentz"), spec_out.pop("gauss"))

        return structure_out, broad_features, g_feature, freq_feature, spec, spec_distorted

    def __getitem__(self, idx):
        structure_out = self.cache[idx]
        structure_out, broad_features, g_feature, freq_feature, spec, spec_distorted = self._prepare_out(structure_out)

    def _load_sample(self, idx: int) -> tp.Dict[str, tp.Any]:
        """Load a single sample"""
        structure_info = self._structure_info[idx]
        samples_number = 0

        structure, generator_summary = self.file_parser.open_structure(
            structure_info["structure_path"]
        )
        structure_out = {}
        tensor_embedings = []
        min_field_pos = []
        max_field_pos = []
        ham_strain = []
        spec = []
        freq = []
        for mean_data in structure_info["mean_data"]:
            for sample_info in mean_data:
                try:
                    out = self.file_parser.to_graph_data(
                        structure,
                        generator_summary,
                        sample_info["sample_path"],
                        None,
                        None
                    )
                    components = out.pop("components")
                    angles = out.pop("angles")
                    temperature = out.pop("temperatures")
                    spins = out.pop("spins")
                    types = out["types"]

                    tensor_embeding = self.components_angles_transform(components, temperature, angles, types, spins)
                    tensor_embedings.append(
                        torch.flatten(tensor_embeding, start_dim=1, end_dim=-2)
                    )

                    min_field_pos.append(
                        torch.flatten(out.pop("min_field_pos"))
                    )
                    max_field_pos.append(
                        torch.flatten(out.pop("max_field_pos"))
                    )

                    ham_strain.append(
                        torch.flatten(out.pop("ham_strain"), end_dim=-2)
                    )
                    freq.append(
                        torch.flatten(out.pop("freq").expand(ham_strain[0].shape[0]))
                    )
                    spec.append(
                        torch.flatten(out.pop("spec"), end_dim=-2)
                    )
                except FileNotFoundError:
                    continue
        if tensor_embedings:
            tensor_embedings = torch.cat(tensor_embedings, dim=-2)
            spec = torch.cat(spec, dim=-2)
            freq = torch.cat(freq, dim=-1)
            max_field_pos = torch.cat(max_field_pos)
            min_field_pos = torch.cat(min_field_pos)
            ham_strain = torch.cat(ham_strain, dim=-2)

            source = out["source"]
            destinations = out["destinations"]

            types = out["types"]

            structure_out["types"] = types
            structure_out["destinations"] = destinations
            structure_out["source"] = source

            structure_out["tensor_embedings"] = tensor_embedings
            structure_out["spec"] = spec
            structure_out["freq"] = freq

            structure_out["max_field_pos"] = max_field_pos
            structure_out["min_field_pos"] = min_field_pos
            structure_out["num_nudes"] = structure_info["num_nodes"]
            structure_out["ham_strain"] = ham_strain

            return structure_out