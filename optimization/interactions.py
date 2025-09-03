import sys
import typing as tp
import copy

import torch

import spin_system

sys.path.append("..")
from spin_system import BaseSample
from .fitter import ParamSpec, ParameterSpace
import particles


class VaryInteraction:
    def __init__(self, vary_components: tp.Union[torch.Tensor, tp.Sequence, float] = None,
                 vary_frame: tp.Optional[tp.Union[torch.Tensor, tp.Sequence]] = None,
                 vary_strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]] = None,
                 device=torch.device("cpu"), dtype=torch.float32):
        """
        :param vary_components:
        torch.Tensor | Sequence[float] | float
            The tensor components, provided in one of the following forms:
              - A scalar (for the same varying for all components).
              - A sequence of two values (axial and z components).
              - A sequence of three values (principal components).
        The possible units are [T, Hz, dimensionless]

        :param vary_frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the tensor. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        :param vary_strain:
        torch.Tensor| Sequence[float] | float, optional
            Parameters describing interaction broadening or distribution.
            Default is `None`.

        :param device:

        :param dtype:
        """
        self.vary_components = vary_components
        self.vary_frame = vary_frame
        self.vary_strain = vary_strain


class VaryDEInteraction:
    def __init__(self, vary_components: tp.Union[torch.Tensor, tp.Sequence, float] = None,
                 vary_frame: tp.Optional[tp.Union[torch.Tensor, tp.Sequence]] = None,
                 vary_strain: tp.Optional[tp.Union[torch.Tensor, tp.Sequence, float]] = None,
                 device=torch.device("cpu"), dtype=torch.float32):
        """
        :param vary_components:
        torch.Tensor | Sequence[float] | float
            The tensor components, provided in one of the following forms:
              - A scalar (varying only D component).
              - A sequence of two values (D and E).
        The possible units are [T, Hz, dimensionless]

        :param vary_frame:
        torch.Tensor | Sequence[float] optional
            Orientation of the tensor. Can be provided as:
              - A 1D tensor of shape (3,) representing Euler angles in ZYZ' convention.
              - A 2D tensor of shape (3, 3) representing a rotation matrix.
            Default is `None`, meaning lab frame.

        :param vary_strain:
        torch.Tensor| Sequence[float] | float, optional
            Parameters describing interaction broadening or distribution.
            Default is `None`.

        :param device:

        :param dtype:
        """
        self.vary_components = vary_components
        self.vary_frame = vary_frame
        self.vary_strain = vary_strain


def make_bounds_around(default: float, half_width: float, positive: bool = False) -> tp.Tuple[float, float]:
    if positive:
        lo = max(default - abs(float(half_width)), 0)
    else:
        lo = default - abs(float(half_width))
    hi = default + abs(float(half_width))
    return (float(lo), float(hi))


def vector_to_DE(components: torch.Tensor):
    D = (2/3) * components[2]
    E = (components[0] - components[1]) / 2
    return D, E


def DE_to_vector(D: float, E: float) -> torch.Tensor:
    z_comp = (2/3) * D
    x_comp = -D/3 + E
    y_comp = -D/3 - E
    return torch.tensor([x_comp, y_comp, z_comp], dtype=torch.float32)


class ComponentsParser:
    def _parse_components(
            self, spin_system_components,
            vary_components,
            vary_parameters: list[ParamSpec],
            fixed_parameters: dict[str, float],
            base_name: str,
            de_mode: bool = False,
            positive: bool = False,
    ):
        if de_mode:
            self._parse_components_de(
            spin_system_components, vary_components,
            vary_parameters, fixed_parameters, base_name, positive=positive
        )
        else:
            self._parse_components_normal(
                spin_system_components, vary_components,
                vary_parameters, fixed_parameters, base_name, positive=positive
            )

    def _parse_components_de(
            self, spin_system_components, vary_components,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], base_name: str, positive: bool
    ):
        if vary_components is not None:
            if isinstance(vary_components, float):
                self._parse_single_value_de(spin_system_components, vary_components, base_name, vary_parameters,
                                         fixed_parameters, positive)

            elif isinstance(vary_components, tp.Sequence) and len(vary_components) == 2:
                self._parse_two_value_de(spin_system_components, vary_components, base_name, vary_parameters,
                                  fixed_parameters, positive)

        else:
            if spin_system_components is not None:
                fixed_parameters[f"{base_name}_D"], fixed_parameters[f"{base_name}_E"] =\
                    vector_to_DE(spin_system_components)

    def _parse_components_normal(
            self, spin_system_components, vary_components,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], base_name: str, positive: bool
    ):
        if vary_components is not None:
            if isinstance(vary_components, float):
                self._parse_single_value_normal(
                    spin_system_components, vary_components, base_name, vary_parameters, fixed_parameters, positive)

            elif isinstance(vary_components, tp.Sequence) and len(vary_components) == 2:
                self._parse_axial_normal(spin_system_components, vary_components, base_name, vary_parameters,
                                         fixed_parameters, positive)

            elif isinstance(vary_components, tp.Sequence) and len(vary_components) == 3:
                self._parse_three_components(spin_system_components, vary_components, base_name, vary_parameters,
                                         fixed_parameters, positive)
        else:
            if spin_system_components is not None:
                fixed_parameters[f"{base_name}_x"] = spin_system_components[0]
                fixed_parameters[f"{base_name}_y"] = spin_system_components[1]
                fixed_parameters[f"{base_name}_z"] = spin_system_components[2]

    def _parse_two_value_de(
            self, spin_system_components, vary_components, base_name: str,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], positive: bool
    ):
        if spin_system_components is not None:
            D, E = vector_to_DE(spin_system_components)
            D = D.item()
            E = E.item()
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_D", default=D, bounds=make_bounds_around(D, vary_components[0],
                                                                                      positive=positive))
            )
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_D", default=E, bounds=make_bounds_around(E, vary_components[1],
                                                                                      positive=positive))
            )

        else:
            default_D, default_E = 0.0, 0.0
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_D", default=default_D,
                          bounds=make_bounds_around(default_D, vary_components[0])
                          )
            )
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_E", default=default_E,
                          bounds=make_bounds_around(default_E, vary_components[1])
                          )
            )

    def _parse_single_value_de(
            self, spin_system_components, vary_components, base_name: str,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], positive: bool
    ):
        if spin_system_components is not None:
            D, E = vector_to_DE(spin_system_components)
            default = D.item()
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_D", default=default, bounds=make_bounds_around(
                    default, vary_components, positive=positive))
            )
            fixed_parameters[f"{base_name}_E"] = E.item()

        else:
            default = 0.0
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_D", default=default, bounds=make_bounds_around(
                    default, vary_components, positive=positive))
            )

    def _parse_single_value_normal(
            self, spin_system_components: tp.Optional[torch.Tensor], vary_components, base_name: str,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], positive: bool
    ):
        if spin_system_components is not None:
            default = spin_system_components.mean().item()
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_iso", default=default, bounds=make_bounds_around(
                    default, vary_components, positive=positive))
            )
            if torch.all(spin_system_components == spin_system_components[0]):
                pass
            elif self._check_two_equality(spin_system_components):
                suffix, result = self._get_axial_anisotropy(spin_system_components)

                fixed_parameters[f"{base_name}{suffix}"] = result
            else:
                fixed_parameters[f"{base_name}_z_x_anisotropy"] =\
                    (spin_system_components[2] - spin_system_components[0]).item()
                fixed_parameters[f"{base_name}_z_y_anisotropy"] = (
                            spin_system_components[2] - spin_system_components[1]).item()

        else:
            default = 0.0
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_iso", default=default, bounds=make_bounds_around(
                    default, vary_components, positive=positive))
            )

    def _check_two_equality(self, components: torch.Tensor):
        check_1 = (components[1] == components[0])
        check_2 = (components[1] == components[2])
        check_3 = (components[0] == components[2])
        return check_1 or check_2 or check_3

    def _get_axial_anisotropy(self, components: torch.Tensor):
        check_1 = (components[1] == components[0])
        check_2 = (components[1] == components[2])
        check_3 = (components[0] == components[2])

        if check_1:
            return "_anisotropy_z_axial", (components[2] - components[0]).item()

        elif check_2:
            return "_anisotropy_x_axial", (components[0] - components[1]).item()

        elif check_3:
            return "_anisotropy_y_axial", (components[1] - components[2]).item()

    def _parse_axial_normal(
            self, spin_system_components: tp.Optional[torch.Tensor], vary_components, base_name: str,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], positive: bool
    ):
        if spin_system_components is not None:
            default_axial = spin_system_components[:-1].mean().item()
            default_z = spin_system_components[-1].mean().item()
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_axial", default=default_axial, bounds=make_bounds_around(
                    default_axial, vary_components[0], positive=positive))
            )
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_z", default=default_z, bounds=make_bounds_around(
                    default_z, vary_components[1], positive=positive))
            )
            if torch.all(spin_system_components == spin_system_components[0]):
                pass
            elif (
                    (spin_system_components[0] == spin_system_components[1]) and
                    (spin_system_components[2] != spin_system_components[0])
            ):
                pass
            else:
                fixed_parameters[f"{base_name}_z_axial_anisotropy"] =\
                    (spin_system_components[2] - spin_system_components[0]).item()
        else:
            default_axial = 0.0
            default_z = 0.0
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_axial", default=default_axial, bounds=make_bounds_around(
                    default_axial, vary_components[0], positive=positive))
            )
            vary_parameters.append(
                ParamSpec(name=f"{base_name}_z", default=default_z, bounds=make_bounds_around(
                    default_z, vary_components[1], positive=positive))
            )


    def _parse_three_components(
            self, spin_system_components, vary_components, base_name: str,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], positive: bool
    ):
        if spin_system_components is not None:
            default_x = spin_system_components[0].item()
            default_y = spin_system_components[1].item()
            default_z = spin_system_components[2].item()
        else:
            default_x = 0.0
            default_y = 0.0
            default_z = 0.0

        vary_parameters.append(
            ParamSpec(name=f"{base_name}_x", default=default_x,
                      bounds=make_bounds_around(default_x, vary_components[0]))
        )
        vary_parameters.append(
            ParamSpec(name=f"{base_name}_y", default=default_y,
                      bounds=make_bounds_around(default_y, vary_components[1], positive=positive))
        )
        vary_parameters.append(
            ParamSpec(name=f"{base_name}_z", default=default_z,
                      bounds=make_bounds_around(default_z, vary_components[2], positive=positive))
        )


class SampleUpdator:
    def __init__(self, sample: spin_system.BaseSample):

        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self._base_sample = sample
        self.electrons = sample.base_spin_system.electrons
        self.nuclei = sample.base_spin_system.nuclei

    def update(self, params: dict[str, float], *args) -> spin_system.BaseSample:
        lorentz = None
        gauss = None
        if "lorentz" in params:
            lorentz = torch.tensor(params["lorentz"], dtype=self.dtype)
        if "gauss" in params:
            gauss = torch.tensor(params["gauss"], dtype=self.dtype)
        ham_strain = self._get_ham_strain(params)

        g_tensors, el_el, el_nuc, nuc_nuc = self._parse_interactions(params)
        self._base_sample.update(lorentz=lorentz, gauss=gauss,
                                 ham_strain=ham_strain, g_tensors=g_tensors,
                                 electron_electron=el_el, electron_nuclei=el_nuc, nuclei_nuclei=nuc_nuc
                                 )

        return self._base_sample

    def __call__(self, params: dict[str, float], *args) -> spin_system.BaseSample:
        return self.update(params)

    def _copy_sample(self, sample):
        return copy.deepcopy(sample)

    def _get_ham_strain(self, params: dict[str, float]):
        if "ham_strain" in params:
            ham_strain = torch.tensor(params["ham_strain"], dtype=self.dtype)
        elif all(f"ham_str_{axis}" in params for axis in "xyz"):
            strain = torch.tensor([
                params["ham_strain_x"],
                params["ham_strain_y"],
                params["ham_strain_z"]
            ], dtype=self.dtype)
            ham_strain = strain
        else:
            ham_strain = None
        return ham_strain

    def _parse_interactions(self, params: dict[str, float]) -> tp.Tuple[tp.List, tp.List, tp.List, tp.List]:
        g_tensors = self._parse_g_tensors(params)
        el_el = self._parse_interaction_type(params, "el_el")
        el_nuc = self._parse_interaction_type(params, "el_nuc")
        nuc_nuc = self._parse_interaction_type(params, "nuc_nuc")

        return g_tensors, el_el, el_nuc, nuc_nuc

    def _parse_g_tensors(self, params: dict[str, float]) -> tp.List:
        """Parse g-tensor parameters."""
        g_tensors = []

        g_indices = set()
        for key in params.keys():
            if key.startswith("g_") and not key.endswith(("_strain", "_frame")):
                parts = key.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    g_indices.add(int(parts[1]))

        for idx in sorted(g_indices):
            base_name = f"g_{idx}"
            interaction = self._create_interaction_from_params(params, base_name)
            if interaction is not None:
                g_tensors.append(interaction)

        return g_tensors

    def _parse_interaction_type(self, params: dict[str, float], interaction_type: str) -> tp.List:
        interactions = []
        interaction_pairs = set()
        for key in params.keys():
            if key.startswith(f"{interaction_type}_") and not key.endswith(("_strain", "_frame")):
                parts = key.split("_")
                if len(parts) >= 4 and parts[2].isdigit() and parts[3].isdigit():
                    i, j = int(parts[2]), int(parts[3])
                    interaction_pairs.add((i, j))

        for i, j in sorted(interaction_pairs):
            base_name = f"{interaction_type}_{i}_{j}"
            interaction = self._create_interaction_from_params(params, base_name)
            if interaction is not None:
                interactions.append((i, j, interaction))

        return interactions

    def _create_interaction_from_params(self, params: dict[str, float], base_name: str) -> tp.Optional:
        has_de = any(key.startswith(f"{base_name}_D") or key.startswith(f"{base_name}_E")
                     for key in params.keys())

        if has_de:
            components = self._reconstruct_de_components(params, base_name)
        else:
            components = self._reconstruct_normal_components(params, base_name)

        if components is None:
            return None

        strain = self._reconstruct_strain(params, base_name)

        frame = self._reconstruct_frame(params, base_name)
        return spin_system.Interaction(
            components=components,
            strain=strain,
            frame=frame,
            device=self.device,
            dtype=self.dtype
        )

    def _reconstruct_de_components(self, params: dict[str, float], base_name: str) -> tp.Optional[torch.Tensor]:
        d_key = f"{base_name}_D"
        e_key = f"{base_name}_E"

        if d_key in params and e_key in params:
            return DE_to_vector(params[d_key], params[e_key])
        elif d_key in params:
            return DE_to_vector(params[d_key], 0.0)

        return None

    def _reconstruct_normal_components(self, params: dict[str, float], base_name: str) -> tp.Optional[torch.Tensor]:
        """Reconstruct tensor components from normal parameterization."""
        if all(f"{base_name}_{axis}" in params for axis in "xyz"):
            return torch.tensor([
                params[f"{base_name}_x"],
                params[f"{base_name}_y"],
                params[f"{base_name}_z"]
            ], dtype=self.dtype)

        if f"{base_name}_axial" in params and f"{base_name}_z" in params:
            axial_val = params[f"{base_name}_axial"]
            z_val = params[f"{base_name}_z"]

            x_val = axial_val
            y_val = axial_val

            if f"{base_name}_z_axial_anisotropy" in params:
                aniso = params[f"{base_name}_z_axial_anisotropy"]
                pass

            return torch.tensor([x_val, y_val, z_val], dtype=self.dtype)

        if f"{base_name}_iso" in params:
            iso_val = params[f"{base_name}_iso"]
            x_val = y_val = z_val = iso_val

            if f"{base_name}_anisotropy_z_axial" in params:
                aniso = params[f"{base_name}_anisotropy_z_axial"]
                z_val = iso_val + aniso
            elif f"{base_name}_anisotropy_x_axial" in params:
                aniso = params[f"{base_name}_anisotropy_x_axial"]
                x_val = iso_val + aniso
            elif f"{base_name}_anisotropy_y_axial" in params:
                aniso = params[f"{base_name}_anisotropy_y_axial"]
                y_val = iso_val + aniso

            if f"{base_name}_z_x_anisotropy" in params:
                aniso = params[f"{base_name}_z_x_anisotropy"]
                z_val = x_val + aniso
            if f"{base_name}_z_y_anisotropy" in params:
                aniso = params[f"{base_name}_z_y_anisotropy"]
                z_val = y_val + aniso

            return torch.tensor([x_val, y_val, z_val], dtype=self.dtype)

        return None

    def _reconstruct_strain(self, params: dict[str, float], base_name: str) -> tp.Optional[torch.Tensor]:
        """Reconstruct strain tensor from parameters."""
        strain_base = f"{base_name}_strain"

        if all(f"{strain_base}_{axis}" in params for axis in "xyz"):
            return torch.tensor([
                params[f"{strain_base}_x"],
                params[f"{strain_base}_y"],
                params[f"{strain_base}_z"]
            ], dtype=self.dtype)

        if f"{strain_base}_D" in params:
            d_val = params[f"{strain_base}_D"]
            e_val = params.get(f"{strain_base}_E", 0.0)
            return DE_to_vector(d_val, e_val)

        if f"{strain_base}_iso" in params:
            iso_val = params[f"{strain_base}_iso"]
            return torch.tensor([iso_val, iso_val, iso_val], dtype=self.dtype)

        return None

    def _reconstruct_frame(self, params: dict[str, float], base_name: str) -> torch.Tensor:
        frame_base = f"{base_name}_frame"

        alpha = params.get(f"{frame_base}_alpha", 0.0)
        beta = params.get(f"{frame_base}_beta", 0.0)
        gamma = params.get(f"{frame_base}_gamma", 0.0)

        return torch.tensor([alpha, beta, gamma], dtype=self.dtype)

    def _create_spin_system(self, g_tensors: tp.List, el_el: tp.List, el_nuc: tp.List, nuc_nuc: tp.List):
        """Create a new spin system with the reconstructed interactions."""
        return spin_system.SpinSystem(
            electrons=self.electrons,
            nuclei=self.nuclei,
            g_tensors=g_tensors,
            electron_electron=el_el,
            electron_nuclei=el_nuc,
            nuclei_nuclei=nuc_nuc
        )


class SampleVary:
    def __init__(self):
        self._base_sample = None
        self._param_map: tp.Dict[str, tp.Any] = {}
        self._param_space: tp.Optional[ParameterSpace] = None

        self.electrons = None
        self.nuclei = None
        self._components_parser = ComponentsParser()
        self._interaction_kinds: dict[str, str] = {}



    def vary(self,
             sample,
             g_tensors: tp.Optional[list[tuple[int, VaryInteraction]]] = None,
             electron_nuclei: tp.Optional[list[tuple[int, int, VaryInteraction]]] = None,
             electron_electron: tp.Optional[list[tuple[int, int, tp.Union[VaryInteraction, VaryDEInteraction]]]] = None,
             nuclei_nuclei: tp.Optional[list[tuple[int, int, tp.Union[VaryInteraction, VaryDEInteraction]]]] = None,
             ham_strain: tp.Optional[tp.Union[tuple[float, float, float], float]] = None,
             lorentz: tp.Optional[float] = None, gauss: tp.Optional[float] = None
    ) -> tuple[ParameterSpace, SampleUpdator]:
        """
        Build ParameterSpace for the provided sample and vary descriptors.

        Returns:
            ParameterSpace instance describing the varying parameters. Also saves
            internal metadata to allow _serialize_spin_system(...) to apply values.
        """
        vary_parameters: tp.List[ParamSpec] = []
        fixed_parameters: tp.Dict[str, float] = {}


        if lorentz is not None:
            default = getattr(sample, "lorentz")
            default_val = float(default.item())
            lorentz_param = ParamSpec(
                name="lorentz", default=default_val, bounds=(max(default_val - lorentz, 0.0), default_val + lorentz)
            )
            vary_parameters.append(lorentz_param)

        else:
            default = getattr(sample, "lorentz")
            fixed_parameters["lorentz"] = float(default.item())

        if gauss is not None:
            default = getattr(sample, "gauss")
            default_val = float(default.item())
            lorentz_param = ParamSpec(
                name="gauss", default=default_val, bounds=(max(default_val - gauss, 0.0), default_val + gauss)
            )
            vary_parameters.append(lorentz_param)

        else:
            default = getattr(sample, "gauss")
            fixed_parameters["gauss"] = float(default.item())

        self._parse_ham_strain(ham_strain, sample, vary_parameters, fixed_parameters)
        base_spin_system = sample.base_spin_system
        self._parse_spin_system(
            base_spin_system,
            g_tensors,
            electron_nuclei,
            electron_electron,
            nuclei_nuclei,
            vary_parameters, fixed_parameters
        )

        self._sample_updator = SampleUpdator(sample=sample)
        return ParameterSpace(vary_parameters, fixed_parameters), self._sample_updator

    def _parse_ham_strain(
            self, ham_strain: tp.Optional[tp.Union[tuple[float, float, float], float]],
            sample: BaseSample, vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]
    ):
        if ham_strain is not None:
            if isinstance(ham_strain, tp.Sequence):
                base = getattr(sample, "base_ham_strain", None)
                if base is None:
                    base_vals = [0.0, 0.0, 0.0]
                base_vals = list(base)

                for i, halfw in enumerate(ham_strain):
                    name = f"ham_strain_{'xyz'[i]}"
                    default_val = float(base_vals[i])
                    bounds = make_bounds_around(default_val, float(halfw), positive=True)
                    vary_parameters.append(ParamSpec(name=name, default=default_val, bounds=bounds))
            elif isinstance(ham_strain, float):
                base = getattr(sample, "base_ham_strain", 0.0)
                default_val = float(base) if not hasattr(base, "item") else float(base.item())
                bounds = make_bounds_around(default_val, float(ham_strain), positive=True)
                vary_parameters.append(ParamSpec(name="ham_strain", default=default_val, bounds=bounds))
            else:
                raise ValueError("ham_strain must be Sequence or float")
        else:
            base = getattr(sample, "base_ham_strain", None)
            if base is None:
                fixed_parameters["ham_strain"] = 0.0
            else:
                vals = base.tolist()
                if len(vals) == 3:
                    fixed_parameters["ham_strain_x"] = float(vals[0])
                    fixed_parameters["ham_strain_y"] = float(vals[1])
                    fixed_parameters["ham_strain_z"] = float(vals[2])
                else:
                    fixed_parameters["ham_strain"] = float(vals[0])

    def _parse_spin_system(
            self,
            base_spin_system,
            g_tensors,
            electron_nuclei,
            electron_electron,
            nuclei_nuclei,
            vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]
    ):
        self.electrons = base_spin_system.electrons
        self.nuclei = base_spin_system.nuclei

        spin_system_g_tensors = base_spin_system.g_tensors
        spin_system_el_el = base_spin_system.electron_electron
        spin_system_el_nuc = base_spin_system.electron_nuclei
        spin_system_nuc_nuc = base_spin_system.nuclei_nuclei

        if (g_tensors is not None) or (spin_system_g_tensors is not None):
            self._process_g_tensor(spin_system_g_tensors, g_tensors, vary_parameters, fixed_parameters)

        if (electron_electron is not None) or (spin_system_el_el is not None):
            self._process_el_el(spin_system_el_el, electron_electron, vary_parameters, fixed_parameters)

        if (electron_nuclei is not None) or (spin_system_el_nuc is not None):
            self._process_el_nuc(spin_system_el_nuc, electron_nuclei, vary_parameters, fixed_parameters)

        if (nuclei_nuclei is not None) or (spin_system_el_nuc is not None):
            self._process_nuc_nuc(spin_system_nuc_nuc, nuclei_nuclei, vary_parameters, fixed_parameters)


    def _parse_frame(self, spin_system_frame, vary_frame,
                     vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float],
                     base_name: str):

        if vary_frame is not None:
            default_alpha = 0.0
            default_beta = 0.0
            default_gamma = 0.0

            if spin_system_frame is not None:
                default_alpha = float(spin_system_frame[0].item())
                default_beta = float(spin_system_frame[1].item())
                default_gamma = float(spin_system_frame[2].item())

            if isinstance(vary_frame, (float, int)):
                half = float(vary_frame)
                vary_parameters.append(
                    ParamSpec(
                        name=f"{base_name}_alpha", default=default_alpha, bounds=make_bounds_around(default_alpha, half)
                    )
                )
                vary_parameters.append(
                    ParamSpec(
                        name=f"{base_name}_beta", default=default_beta, bounds=make_bounds_around(default_beta, half)
                    )
                )
                vary_parameters.append(
                    ParamSpec(
                        name=f"{base_name}_gamma", default=default_gamma, bounds=make_bounds_around(default_gamma, half)
                    )
                )
            elif isinstance(vary_frame, tp.Sequence) and len(vary_frame) == 3:
                vary_parameters.append(
                    ParamSpec(
                        name=f"{base_name}_alpha",
                        default=default_alpha, bounds=make_bounds_around(default_alpha, vary_frame[0]))
                )
                vary_parameters.append(
                    ParamSpec(
                        name=f"{base_name}_beta",
                        default=default_beta, bounds=make_bounds_around(default_beta, vary_frame[1]))
                )
                vary_parameters.append(
                    ParamSpec(
                        name=f"{base_name}_gamma",
                        default=default_gamma, bounds=make_bounds_around(default_gamma, vary_frame[2]))
                )
            else:
                fixed_parameters[f"{base_name}_alpha"] = float(default_alpha)
                fixed_parameters[f"{base_name}_beta"] = float(default_beta)
                fixed_parameters[f"{base_name}_gamma"] = float(default_gamma)
        else:
            if (spin_system_frame == 0.0).all():
                pass
            else:
                fixed_parameters[f"{base_name}_alpha"] = float(spin_system_frame[0].item())
                fixed_parameters[f"{base_name}_beta"] = float(spin_system_frame[1].item())
                fixed_parameters[f"{base_name}_gamma"] = float(spin_system_frame[2].item())

    def _parse_interaction(self, spin_system_interaction: spin_system.Interaction,
                           vary_interaction: tp.Union[VaryDEInteraction, VaryInteraction],
                           vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float], base_name: str):
        de_mode = isinstance(vary_interaction, VaryDEInteraction)

        self._components_parser._parse_components(
                getattr(spin_system_interaction, "components", None), getattr(vary_interaction, "vary_components", None),
                vary_parameters, fixed_parameters, f"{base_name}", de_mode, positive=False
            )

        self._components_parser._parse_components(
                getattr(spin_system_interaction, "strain", None), getattr(vary_interaction, "vary_strain", None),
                vary_parameters, fixed_parameters, f"{base_name}_strain", de_mode, positive=True
            )
        self._parse_frame(
            getattr(spin_system_interaction, "frame", None), getattr(vary_interaction, "vary_frame", None),
            vary_parameters, fixed_parameters, f"{base_name}_frame")

    def _parse_not_vary_interaction(self, spin_system_interaction, fixed_parameters: dict[str, float], base_name):
        components = spin_system_interaction.components
        strain = spin_system_interaction.strain
        frame = spin_system_interaction.frame

        local_name = f"{base_name}"
        fixed_parameters[f"{local_name}_x"],\
            fixed_parameters[f"{local_name}_y"],\
            fixed_parameters[f"{local_name}_z"] = components[0].item(), components[1].item(), components[2].item()

        if strain is not None:
            local_name = f"{base_name}_strain"
            fixed_parameters[f"{local_name}_x"],\
                fixed_parameters[f"{local_name}_y"],\
                fixed_parameters[f"{local_name}_z"] = strain[0].item(), strain[1].item(), strain[2].item()

        if not (frame == 0).all():
            local_name = f"{base_name}_frame"
            fixed_parameters[f"{local_name}_alpha"],\
                fixed_parameters[f"{local_name}_beta"],\
                fixed_parameters[f"{local_name}_gamma"] = frame[0].item(), frame[1].item(), frame[2].item()

    def _process_g_tensor(self, spin_system_g_tensors, vary_g_tensor,
                          vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]
                          ):
        dicted_vary = {element[0]: element[1] for element in vary_g_tensor}
        base_name = "g"
        for idx, entry in enumerate(spin_system_g_tensors):
            g_tensor = entry
            base_name_local = f"{base_name}_{idx}"
            if idx in dicted_vary:
                self._parse_interaction(g_tensor, dicted_vary[idx], vary_parameters, fixed_parameters, base_name_local)
            else:
                self._parse_not_vary_interaction(
                    g_tensor,  fixed_parameters, base_name_local
                )

    def _process_interaction_type(self, spin_system_data, vary_data, interaction_type,
                                  vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]):
        dicted_vary = {}
        if vary_data is not None:
            for entry in vary_data:
                key = (entry[0], entry[1])
                dicted_vary[key] = entry[2]

        for entry in spin_system_data:
            i, j, interaction = entry[0], entry[1], entry[2]

            base_name_local = f"{interaction_type}_{i}_{j}" if j is not None else f"{interaction_type}_{i}"
            vary_key = (i, j)
            alt_key = (j, i) if j is not None else None

            if vary_data is not None and vary_key in dicted_vary:
                self._parse_interaction(interaction, dicted_vary[vary_key], vary_parameters,
                                        fixed_parameters, base_name_local)
            elif (vary_data is not None and alt_key is not None and
                  alt_key in dicted_vary):
                self._parse_interaction(interaction, dicted_vary[alt_key], vary_parameters,
                                        fixed_parameters, base_name_local)
            else:
                self._parse_not_vary_interaction(interaction, fixed_parameters, base_name_local)

    def _process_el_el(self, spin_system_el_el, vary_el_el,
                       vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]):
        self._process_interaction_type(spin_system_el_el, vary_el_el, 'el_el',
                                       vary_parameters, fixed_parameters)

    def _process_el_nuc(self, spin_system_el_nuc, vary_el_nuc,
                        vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]):
        self._process_interaction_type(spin_system_el_nuc, vary_el_nuc, 'el_nuc',
                                       vary_parameters, fixed_parameters)

    def _process_nuc_nuc(self, spin_system_nuc_nuc, vary_nuc_nuc,
                         vary_parameters: list[ParamSpec], fixed_parameters: dict[str, float]):
        self._process_interaction_type(spin_system_nuc_nuc, vary_nuc_nuc, 'nuc_nuc',
                                       vary_parameters, fixed_parameters)


