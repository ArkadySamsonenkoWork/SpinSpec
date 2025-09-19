import sys
sys.path.append("..")

import torch
import torch.nn as nn

import spin_system
import constants


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