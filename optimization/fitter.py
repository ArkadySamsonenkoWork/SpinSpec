import copy
from dataclasses import dataclass
import typing as tp
import sys
import math

import nevergrad as ng
import numpy as np
import torch
import optuna
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import spin_system

sys.path.append("..")
from spectra_processing import normalize_spectrum
from optimization import objectives


class TrialResult(tp.TypedDict):
    trial_number: int
    params: dict[str, float]
    delta: dict[str, float]
    loss: float
    distance: float


def print_trial_results(results: tp.Union[TrialResult, list[TrialResult]], max_params=None, precision=6) -> None:
    """
    Print trial results.

    Args:
        results: Single trial dict or list of trial dicts
        max_params: Maximum number of parameters to display (None for all)
        precision: Number of decimal places for numeric values
    """

    if isinstance(results, dict):
        results = [results]

    for i, trial in enumerate(results):
        if i > 0:
            print("\n" + "=" * 80 + "\n")

        # Print header
        print(f"TRIAL #{trial['trial_number']}")
        print(f"index: {i}")
        print("-" * 40)
        print(f"Loss:     {trial['loss']:.{precision}f}")
        print(f"Distance: {trial['distance']:.{precision}f}")
        print()

        params = trial["params"]
        deltas = trial["delta"]
        param_names = list(params.keys())

        if max_params and len(param_names) > max_params:
            param_names = param_names[:max_params]
            truncated = True
        else:
            truncated = False

        print("PARAMETERS:")
        print("-" * 40)
        max_name_len = max(len(name) for name in param_names) if param_names else 0
        for param_name in param_names:
            value = params[param_name]
            delta = deltas.get(param_name, None)

            if isinstance(value, float):
                if abs(value) > 1000000:
                    value_str = f"{value:.{precision - 2}e}"
                else:
                    value_str = f"{value:.{precision}f}"
            else:
                value_str = str(value)

            if isinstance(delta, float):
                if abs(delta) > 1000000:
                    delta_str = f"{delta:+.{precision - 2}e}"
                else:
                    delta_str = f"{delta:+.{precision}f}"
            else:
                delta_str = str(delta)

            print(f"  {param_name:<{max_name_len}} = {value_str:>15} (Δ {delta_str})")
        if truncated:
            remaining = len(params) - max_params
            print(f"  ... and {remaining} more parameters")


def print_params(params: dict[str, float], max_params=None, precision=6) -> None:
    """
    :param params: the dict of parameter names and their values
    :param max_params: maximum number of parameters
    :param precision: Number of decimal places for numeric values
    :return: None
    """

    param_names = list(params.keys())
    if max_params and len(param_names) > max_params:
        param_names = param_names[:max_params]
        truncated = True
    else:
        truncated = False

    print("PARAMETERS:")
    print("-" * 40)
    max_name_len = max(len(name) for name in param_names) if param_names else 0

    for param_name, in param_names:
        value = params[param_name]

        if isinstance(value, float):
            if abs(value) > 1000000:
                value_str = f"{value:.{precision - 2}e}"
            else:
                value_str = f"{value:.{precision}f}"
        else:
            value_str = str(value)

        print(f"  {param_name:<{max_name_len}} = {value_str:>15} ")

    if truncated:
        remaining = len(params) - max_params
        print(f"  ... and {remaining} more parameters")


@dataclass
class FitResult:
    best_params: tp.Dict[str, float]
    best_loss: float
    best_spectrum: tp.Optional[torch.Tensor]
    optimizer_info: tp.Dict


@dataclass
class ExperementalParameters:
    best_params: tp.Dict[str, float]
    best_loss: float
    best_spectrum: tp.Optional[torch.Tensor]
    optimizer_info: tp.Dict


@dataclass
class NevergradTrial:
    params: tp.Dict[str, float]
    value: float
    _trial_id: int

    def __repr__(self):
        return f"_trial_id: {self._trial_id}, loss: {self.value}"

    def __str__(self):
        return f"_trial_id: {self._trial_id}, loss: {self.value}"


class TrialsTracker:
    def __init__(self):
        self.trials = []
        self.losses = []
        self.step = 0

    def __call__(self, optimizer: ng.optimization.Optimizer,
                 candidate: ng.p.Instrumentation, loss: float):
        """Callback function called after each evaluation"""
        self.trials.append(candidate.value[0])
        self.losses.append(loss)
        self.step += 1

        # Optional: print progress
        if self.step % 10 == 0:
            print(f"Step {self.step}: Loss = {loss:.6f}")

    def get_best_trial(self):
        """Get the trial with the lowest loss"""
        best_idx = np.argmin(self.losses)
        return {
            '_trial_id': best_idx + 1,
            'params': self.trials[best_idx],
            'value': self.losses[best_idx]
        }

    def get_all_trials(self):
        """Get all trials as a list of dictionaries"""
        return [
            {
                '_trial_id': i + 1,
                'params': trial,
                'value': loss
            }
            for i, (trial, loss) in enumerate(zip(self.trials, self.losses))
        ]


class LogTransform:
    def __call__(self, x: float) -> float:
        return math.pow(10, x)

    def inverse(self, y: float) -> float:
        return math.log(y)


@dataclass
class ParamSpec:
    """Specification for a single scalar parameter.

    Attributes:
        name: parameter name

        bounds: (low, high) bounds for optimizer search (floats)

        default: optional default value to use for initialization

        transform: optional callable applied to a raw optimizer value to map
                   it to the physical parameter (for example, log-scales)

        vary: bool: Whether the parameter should vary or not.
        In the latter case, this is equivalent to specifying the parameter in fixed_parameters.
        If you don't plan to vary the parameter, then the more correct way is to specify it in fixed_parameters.

    """
    name: str
    bounds: tp.Tuple[float, float]
    default: tp.Optional[float] = None
    transform: tp.Optional[tp.Callable[[float], float]] = None
    vary: bool = True

    def clip(self, x: float) -> float:
        lo, hi = self.bounds
        return float(min(max(x, lo), hi))

    def apply(self, x: float) -> float:
        x = self.clip(x)
        return self.transform(x) if self.transform is not None else x

    def set_bounds(self, bounds: tp.Tuple[float, float]):
        """Update the bounds for this parameter spec."""
        self.bounds = bounds


class ParameterSpace:
    print_precision: int = 4
    def __init__(self, specs: tp.Sequence[ParamSpec],
                 fixed_params: tp.Optional[tp.Dict[str, float]] = None):
        """
        :param specs: The sequence of ParamSpec instances.
        The list include parameters that should be varied (if spec.vary = True. For more details
        see ParamSpec documentation)
        :param fixed_params: The parameters that are fixed during fit.
        """
        self.specs = list(specs)

        self.fixed_params: tp.Dict[str, float] = {} if fixed_params is None else dict(fixed_params)
        self.fixed_params.update({s.name: s.default for s in self.specs if not getattr(s, "vary")})

        self._varying_specs = [s for s in self.specs if getattr(s, "vary", True)]
        self.varying_names = [s.name for s in self._varying_specs]
        self.varying_params = {s.name: s.default for s in self._varying_specs}

        for name in list(self.fixed_params.keys()):
            if name in self.varying_names:
                idx = next(i for i, s in enumerate(self._varying_specs) if s.name == name)
                del self._varying_specs[idx]
                self.varying_names.remove(name)

    def __deepcopy__(self, memo):
        new_obj = type(self).__new__(type(self))

        new_obj.specs = copy.deepcopy(self.specs, memo)
        new_obj.fixed_params = copy.deepcopy(self.fixed_params, memo)
        new_obj._varying_specs = copy.deepcopy(self._varying_specs, memo)
        new_obj.varying_names = copy.deepcopy(self.varying_names, memo)
        new_obj.varying_params = copy.deepcopy(self.varying_params, memo)
        new_obj.print_precision = self.print_precision
        return new_obj

    def __getitem__(self, key: str):
        try:
            return self.fixed_params[key]
        except KeyError:
            try:
                return self.varying_params[key]
            except KeyError:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def __setitem__(self, key: str, value: float):
        if key in self.fixed_params:
            self.fixed_params[key] = value
        elif key in self.varying_names:
            for spec in self._varying_specs:
                if spec.name == key:
                    spec.default = value
                    self.varying_params[key] = value
        else:
            raise KeyError(f"Key '{key}' not found in fixed_params or varying_params")

    def __dict__(self) -> dict[str, float]:
        return {**self.varying_params, **self.fixed_params}

    def __iter__(self):
        return iter(self.__dict__().items())

    def __repr__(self) -> str:
        """
        Print parameters space
        """

        text = ""

        text += f"____Fixed parameters_____ \n"
        text += "-" * 40 + "\n"
        param_names = list(self.fixed_params.keys())
        for key, value in self.fixed_params.items():
            max_name_len = max(len(name) for name in param_names) if param_names else 0
            if isinstance(value, float):
                if abs(value) > 1000:
                    value_str = f"{value:.{self.print_precision - 2}e}"
                else:
                    value_str = f"{value:.{self.print_precision}f}"
            else:
                value_str = str(value)
            text += f"  {key:<{max_name_len}} = {value_str:>15}\n"

        text += "\n\n"
        text += f"______Varying parameters_____ \n"
        text += "-" * 40 + "\n"

        param_names = list(self.varying_names)
        for spec in self._varying_specs:
            max_name_len = max(len(name) for name in param_names) if param_names else 0

            value = spec.default
            name = spec.name
            (low, up) = spec.bounds

            if isinstance(value, float):
                if abs(value) > 1000:
                    value_str = f"{value:.{self.print_precision - 2}e}"
                else:
                    value_str = f"{value:.{self.print_precision}f}"
            else:
                value_str = str(value)

            if isinstance(low, float):
                if abs(low) > 1000000:
                    low = f"{low:+.{self.print_precision - 2}e}"
                else:
                    low = f"{low:+.{self.print_precision}f}"
            else:
                low = str(low)

            if isinstance(up, float):
                if abs(up) > 1000:
                    up = f"{up:+.{self.print_precision - 2}e}"
                else:
                    up = f"{up:+.{self.print_precision}f}"
            else:
                up = str(up)
            text += f"  {name:<{max_name_len}} = {value_str:>15}   (low:   {low}  up:   {up})\n"

        return text

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self, name: str, value: tp.Optional[float] = None):
        """
        Freeze a parameter by name. If value is provided, use it; otherwise
        use its default (or current) value.
        """
        if name not in self.varying_names:
            raise KeyError(name)
        spec = next(s for s in self.specs if s.name == name)

        if value is None:
            if spec.default is not None:
                value = float(spec.default)
            else:
                lo, hi = spec.bounds
                value = 0.5 * (lo + hi)
        self.fixed_params[name] = float(value)

        self._varying_specs = [s for s in self._varying_specs if s.name != name]
        self.varying_names = [s.name for s in self._varying_specs]
        self.varying_params = {s.name: s.default for s in self._varying_specs}

    def unfreeze(self, name: str):
        """Unfreeze a parameter previously frozen with `freeze` or fixed_params."""
        if name in self.fixed_params:
            del self.fixed_params[name]

        for s in self.specs:
            if s.name == name and s not in self._varying_specs and getattr(s, 'vary', True):
                self._varying_specs.append(s)
                self.varying_names.append(s.name)
        self.varying_params = {s.name: s.default for s in self._varying_specs}

    def vector_to_dict(self, vec: tp.Sequence[float]) -> tp.Dict[str, float]:
        """Convert an optimizer vector (ordered only over *varying* params)
        into a full parameter dict that includes fixed parameters.
        """
        if len(vec) != len(self._varying_specs):
            raise ValueError(f"Expected vector of length {len(self._varying_specs)}, got {len(vec)}")
        out = dict(self.fixed_params)  # start with fixed
        for s, v in zip(self._varying_specs, vec):
            out[s.name] = s.apply(float(v))
        return out

    def varying_vector_to_dict(self, vec: tp.Sequence[float]) -> tp.Dict[str, float]:
        """Convert an optimizer vector (ordered only over *varying* params)
        into a full parameter dict that includes fixed parameters.
        """
        if len(vec) != len(self._varying_specs):
            raise ValueError(f"Expected vector of length {len(self._varying_specs)}, got {len(vec)}")
        out = {}
        for s, v in zip(self._varying_specs, vec):
            out[s.name] = s.apply(float(v))
        return out

    def dict_to_vector(self, params: tp.Dict[str, float]) -> np.ndarray:
        return np.array([params[n] for n in self.varying_names], dtype=float)

    def defaults_vector(self) -> np.ndarray:
        vals = []
        for s in self._varying_specs:
            if s.default is not None:
                vals.append(float(s.default))
            else:
                lo, hi = s.bounds
                vals.append(0.5 * (lo + hi))
        return np.array(vals, dtype=float)

    def _set_single_bounds(self, param_name: str, bounds: tp.Tuple[float, float]):
        """
        :param param_name: name of the varying parameter
        :param bounds: new bounds of the parameter
        :return: None
        """
        if param_name not in self.varying_names:
            raise KeyError(f"Parameter {param_name} not found in varying parameter space")

        low, high = bounds
        if low >= high:
            raise ValueError(f"Invalid bounds: low ({low}) must be less than high ({high})")

        for spec in self.specs:
            if spec.name == param_name:
                spec.set_bounds(bounds)
                break

    def set_default(self, params: dict[str, float]):
        """
        :param params: the dict of parameters. Set default value for parameters given in params
        :return:
        """
        for key, value in params.items():
            if key in self.fixed_params:
                self.fixed_params[key] = value
            elif key in self.varying_params:
                for spec in self._varying_specs:
                    if spec.name == key:
                        spec.default = value
                        self.varying_params[spec.name] = value
                        break
            else:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def reduce_bounds(self, names: tp.Optional[str] = None, alpha: float = 0.2):
        """
        Reduces bounds of varying parameters. If bounds was (a, b) and default value c than the new bounds are:
        delta = (b-a)*alpha
        new_bounds = (c - delta, c+delta)
        :param names: names of parameters to reduce
        :param alpha: reducing coefficient
        :return: None
        """
        if names is None:
            names = self.varying_names
        for name in names:
            for spec in self._varying_specs:
                if spec.name == name:
                    default = spec.default
                    low, up = spec.bounds
                    delta = (up - low) * alpha
                    spec.bounds = (default - delta, default + delta)
                    break

    def set_bounds(self, bounds_dict: tp.Dict[str, tp.Tuple[float, float]]):
        """
        :param bounds_dict: the dict with names of parameters and their new bounds
        :return: None
        """
        for param_name, bounds in bounds_dict:
            self._set_single_bounds(param_name, bounds)

    def suggest_optuna(self, trial) -> tp.Dict[str, float]:
        out = dict(self.fixed_params)  # start with fixed
        for s in self._varying_specs:
            lo, hi = s.bounds
            val = trial.suggest_float(s.name, lo, hi)
            out[s.name] = s.apply(val)
        return out

    def instrument_nevergrad(self) -> ng.p.Instrumentation:
        params = []
        for s in self._varying_specs:
            lo, hi = s.bounds
            params.append(ng.p.Scalar(lower=lo, upper=hi))
        return ng.p.Instrumentation(*params)


class CWSpectraSimulator:
    """
    Example of CW spectra simulator.
    """
    def __init__(self,
                 sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                 spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
        """
        :param sample_updator: Callable object that updates sample
        :param spectra_creator: Callable object that creates spectra
        :param args:
        """
        self.sample_updator = sample_updator
        self.spectra_creator = spectra_creator
        self.args = args

    def __call__(self, fields: torch.Tensor, params: dict[str, float]):
        """
        :param fields: magnetic fields in Tesla units
        :param params: parameters of param space
        :return:
        """
        sample = self.sample_updator(params, *self.args)
        return self.spectra_creator(sample, fields)


class SpectrumFitter:
    """
    General fitter for spectra.
    The user must provide either a `simulate_spectrum_callable` that maps a
    parameter dict -> torch.Tensor (spectrum on the same B-grid), or override
    the `simulate_spectrum` method in a subclass.

    Typical usage:
      - construct with B grid, experimental spectrum (np or torch), device
      - provide parameter specs
      - call fit(method='optuna'|'nevergrad')
    """
    __available_optimizer__ = {"nevergrad": sorted(ng.optimizers.registry.keys()),
                               "optuna": [optuna.integration.BoTorchSampler,
                                          optuna.samplers.RandomSampler,
                                          optuna.samplers.TPESampler,
                                          optuna.samplers.BruteForceSampler,
                                          optuna.samplers.GridSampler,
                                          optuna.samplers.CmaEsSampler,
                                          optuna.samplers.NSGAIISampler,
                                          optuna.samplers.NSGAIIISampler,
                                          ]
                               }

    def __init__(
        self,
        x_exp: tp.Union[np.ndarray, torch.Tensor] | list[tp.Union[np.ndarray, torch.Tensor]],
        y_exp: tp.Union[np.ndarray, torch.Tensor] | list[tp.Union[np.ndarray, torch.Tensor]],
        param_space: ParameterSpace,
        spectra_simulator: tp.Callable[
            [list[torch.Tensor] | torch.Tensor, tp.Dict[str, float], tp.Dict],
             torch.Tensor | list[torch.Tensor]
        ],
        norm_mode: str = "integral",
        objective=objectives.MSEObjective(),
        weights: list[float] = None,
        device: tp.Optional[torch.device] = None,
    ):
        """
        :param x_exp: Experimental x-axis data. It can be magnetic field (T), time (s), or 2d-array (for time-resolved).
            It is possible to pass a list for multi-object fit
        :param y_exp: Experimental y-axis data.
        :param param_space: The object of ParameterSpace class where all varying parameters are included
        :param spectra_simulator: Any callable object that takes x_data and parameters and returns simulated
            spectra or list of simulated spectra.
            It is highly recommended for all new parameters to use update methods:
            sample.update(new_params) or spec_creator.update_config(config)

            Example:
            class CWSpectraSimulator:
                def __init__(self,
                             sample_updator: tp.Callable[[dict[str, float], tp.Any], tp.Any],
                             spectra_creator: tp.Callable[[tp.Any, torch.Tensor], torch.Tensor], *args):
                    self.sample_updator = sample_updator
                    self.spectra_creator = spectra_creator
                    self.args = args

                def __call__(self, fields: torch.Tensor, params: dict[str, float]):
                    sample = self.sample_updator(params, *self.args)
                    return self.spectra_creator(sample, fields)

        :param norm_mode: Norm mode to fit data. 'integral' / 'max'
        :param device: Device for computation
        :param objective: Used objective function. It should be an inheritor of objectives.BaseObjective
        :param weights: The weights for multi-data fit. Default is None
        """
        self.device = torch.device("cpu") if device is None else device
        self.norm_mode = norm_mode
        self._simulate_callable = spectra_simulator

        self.x_exp, self.y_exp, self.multisample = self._set_experemental(x_exp, y_exp)

        if self.multisample and (weights is None):
            self.weights = torch.ones(len(self.x_exp), dtype=torch.float32, device=self.device)
        elif weights is None:
            self.weights = None
        else:
            self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        self.param_space = param_space
        self._objective = objective
        self._loss_normalization = self._get_loss_norm()

    def _set_experemental(self, B: tp.Union[np.ndarray, torch.Tensor] | list[tp.Union[np.ndarray, torch.Tensor]],
                                y_exp: tp.Union[np.ndarray, torch.Tensor] | list[tp.Union[np.ndarray, torch.Tensor]]):
        if isinstance(B, list):
            if len(B) != len(y_exp):
                raise ValueError("The number of fields array and experimental arrays must be the same")
            else:
                B = [torch.tensor(b, dtype=torch.float32, device=self.device) for b in B]
                y_exp = [torch.tensor(y, dtype=torch.float32, device=self.device) for y in y_exp]
                for idx, b in enumerate(B):
                    y_exp[idx] = normalize_spectrum(b, y_exp[idx], mode=self.norm_mode)
                multisample = True
        else:
            B = torch.tensor(B, dtype=torch.float32, device=self.device)
            y_exp = torch.tensor(y_exp, dtype=torch.float32, device=self.device)
            y_exp = normalize_spectrum(B, y_exp, mode=self.norm_mode)
            multisample = False

        return B, y_exp, multisample

    def _get_loss_norm(self):
        if self.multisample:
            return [self._objective(torch.zeros_like(y), y).reciprocal() for y in self.y_exp]
        else:
            return self._objective(torch.zeros_like(self.y_exp), self.y_exp).reciprocal()

    def _simulate_single_spectrum(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        return normalize_spectrum(self.x_exp, self._simulate_callable(self.x_exp, params, **kwargs), mode=self.norm_mode)

    def _simulate_spectral_set(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor]:
        models = self._simulate_callable(self.x_exp, params, **kwargs)
        for idx in range(len(models)):
            models[idx] = normalize_spectrum(self.x_exp[idx], models[idx], mode=self.norm_mode)
        return models

    def simulate_spectroscopic_data(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor] | torch.Tensor:
        """
        :param params: fict of parameter names: parameter values.
        The names of parameters are names from param_space.
        Example:
        fitter.simulate_spectroscopic_data(dict(param_space))
        :param kwargs:
        :return: Simulated spectra - list or single spectra
        """
        if self.multisample:
            model = self._simulate_spectral_set(params, **kwargs)
        else:
            model = self._simulate_single_spectrum(params, **kwargs)
        return model

    def simulate_spectra_from_trial_params(self, trial_params: tp.Dict[str, float], **kwargs) ->\
            list[torch.Tensor] | torch.Tensor:
        """
        :param trial_params: Simulate spectra from parameters given as trial_params (only varied parameters)
        As fixed_parameters the parameters from self.param_space are used
        :param kwargs:
        :return: Simulated spectra - list or single spectra
        """
        return self.simulate_spectroscopic_data({**self.param_space.fixed_params, **trial_params}, **kwargs)

    def _loss_from_params(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        """Compute model - experiment residuals as a torch.Tensor."""
        with torch.no_grad():
            if self.multisample:
                models = self._simulate_spectral_set(params, **kwargs)
                loss = sum(self.weights[idx] * self._loss_normalization[idx] * self._objective(
                    models[idx], self.y_exp[idx]) for idx in range(len(models))) / len(models)
            else:
                model = self._simulate_single_spectrum(params, **kwargs)
                loss = self._loss_normalization * self._objective(model, self.y_exp)
            return loss

    def _tracker_to_trials(self, trials_tracker: TrialsTracker) -> list[NevergradTrial]:
        trials_all_results = trials_tracker.get_all_trials()
        ng_trials = [
            NevergradTrial(params=self.param_space.varying_vector_to_dict(trial["params"]),
                           _trial_id=trial["_trial_id"],
                           value=trial["value"]
                           ) for trial in trials_all_results
        ]
        return ng_trials


    def fit_optuna(
        self,
        show_progress: bool,
        seed: tp.Optional[int],
        return_best_spectrum: bool,

        n_trials: int = 300,
        timeout: tp.Optional[float] = None,
        n_jobs: int = 1,
        sampler: tp.Optional[optuna.samplers.BaseSampler] = None,
        study_name: tp.Optional[str] = None,
        **kwargs,
    ) -> FitResult:
        """Fit using Optuna.

        Requires optuna to be installed.
        """
        def loss_function(trial):
            p = self.param_space.suggest_optuna(trial)
            loss = self._loss_from_params(p, **kwargs)
            return loss

        if sampler is None:
            sampler = optuna.samplers.CmaEsSampler(
                seed=seed,
                n_startup_trials=50)
            #sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)


        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name, load_if_exists=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            loss_function, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress)

        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data({**self.param_space.fixed_params, **best_params}, **kwargs)
        return FitResult(best_params, float(study.best_value), best_spec, {"backend": "optuna", "study": study})

    def fit_nevergrad(
        self,
        show_progress: bool,
        seed: tp.Optional[int],
        return_best_spectrum: bool,

        budget: int = 200,
        optimizer_name: str = "TwoPointsDE",
        track_trials: bool = True,

        **kwargs,
    ) -> FitResult:
        """Fit using Nevergrad (if installed)."""
        if ng is None:
            raise RuntimeError("Nevergrad is required for fit_nevergrad but not installed")

        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)
        opt = ng.optimizers.registry[optimizer_name](parametrization=instr, budget=budget)

        def _loss_from_tuple(*args):
            params = self.param_space.vector_to_dict(args)
            return self._loss_from_params(params).item()

        if show_progress:
            progress_bar = ng.callbacks.ProgressBar()
            progress_bar.update_frequency = 25
            opt.register_callback("tell", progress_bar)

        trials_tracker = None
        if track_trials:
            trials_tracker = TrialsTracker()
            opt.register_callback("tell", trials_tracker)

        recommendation = opt.minimize(_loss_from_tuple)
        x = recommendation.value
        best_params = self.param_space.varying_vector_to_dict(x[0])
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectroscopic_data({**self.param_space.fixed_params, **best_params})

        trials = None
        if track_trials:
            trials = self._tracker_to_trials(trials_tracker)

        return FitResult(
            best_params, self._loss_from_params({**self.param_space.fixed_params, **best_params}), best_spec,
            {"backend": "nevergrad", "optimizer": optimizer_name, "trials": trials}
        )

    def fit(
        self,
        backend: str = "optuna",
        seed: tp.Optional[int] = None,
        show_progress: bool = True,
        return_best_spectrum: bool = True,

        **backend_kwargs,
    ) -> FitResult:
        """
        All fitting methods can be viewed in SpectrumFitter.__available_optimizer__

        :param backend: optuna / nevergrad. Sets which library should be used to fit data.
            Optuna supports not as many methods as nevergrad but they are quite powerful. Default fitting method is TPE.
            TPE has quite high exploration abilities and not as dramatic speed of work as Bayesian models.
            After the initial fitting process it is recommended to reduce
            the bounds and continue fitting with any method of convex optimization from Nevergrad:
            For example, with COBYLA.

        :param backend_kwargs: The kwargs of fit settings described in optuna / nevergrad library
            NOTE! Optuna and Nevergrad have different backend parameters.
            We have saved the initial naming from these libraries

            Key differences:
                                        optuna                                   nevergrad
            ----------------------------------------------------------------------------------------------------
            method type          optuna.samplers.BaseSampler                    str object
            ----------------------------------------------------------------------------------------------------
            number of iterations      n_trials                                    budget
            ----------------------------------------------------------------------------------------------------

        :return: None
        """
        method = backend.lower()
        if method == "optuna":
            return self.fit_optuna(seed=seed, show_progress=show_progress,
                                   return_best_spectrum=return_best_spectrum, **backend_kwargs
                                   )
        if method in ("nevergrad", "ng"):
            return self.fit_nevergrad(seed=seed, show_progress=show_progress,
                                      return_best_spectrum=return_best_spectrum, **backend_kwargs
                                      )
        raise ValueError(f"Unknown fit method: {method}")


class SpaceSearcher:
    """
    For some cases not only the best fitting parameters are useful but all 'good' parameters.
    Space searcher try to catch 'good' parameters that are far from the best fit parameters.
    """
    def __init__(
        self,
        loss_rel_tol: float = 1.0,
        top_k: int = 5,
        distance_fraction: float = 0.2,
    ):
        """
        :param loss_rel_tol: loss_trial / loss_best: cutoff parameter
            that sets the acceptable loss of trial. Default is 1
        :param top_k: Returns only top_k lowest-loss trials.
        :param distance_fraction: Among all 'good' trials with low loss it
            accepts only trials with Euclidean distance in scaled (-1, 1) parameters < distance_fraction * max_distance

            To compute distance the parameters are scaled to (-1, 1)
        """
        self.loss_rel_tol = float(loss_rel_tol)
        self.top_k = int(top_k)
        self.distance_fraction = float(distance_fraction)

    def _parse_trials(self, trials: list[NevergradTrial | optuna.Trial], param_names: list[str]):
        param_rows = []
        losses = []
        trial_ids = []
        for t in trials:
            if t.value is None:
                continue
            vals = []
            for name in param_names:
                if name not in t.params:
                    vals = None
                    break
                vals.append(float(t.params[name]))
            if vals is None:
                continue
            param_rows.append(vals)
            losses.append(float(t.value))
            trial_ids.append(t._trial_id)
        if len(param_rows) == 0:
            return np.zeros((0, 0)), np.array([]), []
        P = np.asarray(param_rows, dtype=float)
        L = np.asarray(losses, dtype=float)
        return P, L, np.asarray(trial_ids, dtype=np.int32)

    def _extract_trials_from_fit(self, fit_result: FitResult,
                                   param_names: list[str] | None = None):
        """
        Return arrays: (param_matrix, losses, trial_indices)
        param_matrix shape: (n_trials, n_varying_params)
        losses: array of length n_trials (float)
        trial_indices: list of optuna trial numbers corresponding to rows
        """
        backend = fit_result.optimizer_info["backend"]

        if backend == "nevergrad":
            trials = fit_result.optimizer_info["trials"]
        elif backend == "optuna":
            trials = [t for t in fit_result.optimizer_info["study"].trials if t.state.is_finished()]
        else:
            raise KeyError("Unknown fit result")

        if len(trials) == 0:
            return np.zeros((0, 0)), np.array([]), []

        if param_names is None:
            param_names = list(fit_result.best_params.keys())
        return trials, param_names

    def __call__(self, fit_result: FitResult, param_names: list[str] | None = None):
        """
        :param fit_result: The output of fitter.
        :param param_names: The names of parameters that should be included in search procedure.
        Default value is None means that all spec (varying) parameters should be included.
        :return:
        """
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, trial_numbers = self._parse_trials(trials, param_names)
        best_params = fit_result.best_params

        if P.size == 0 or L.size == 0:
            return []

        scaler = StandardScaler()
        P_scaled = scaler.fit_transform(P)

        best_loss = float(L.min())
        loss_cutoff = best_loss * (1.0 + self.loss_rel_tol)
        good_mask = L <= loss_cutoff
        if not np.any(good_mask):
            return []

        P_good = P_scaled[good_mask]
        L_good = L[good_mask]
        trials_good = trial_numbers[good_mask]

        best_idx_in_good = int(np.argmin(L_good))
        best_vector = P_good[best_idx_in_good].reshape(1, -1)

        distances = cdist(best_vector, P_good, metric="euclidean").flatten()

        sorted_idx = np.argsort(distances)
        sorted_idx = sorted_idx[sorted_idx != best_idx_in_good][::-1]

        max_dist = max(distances)
        if self.distance_fraction > 0:
            thresh = self.distance_fraction * max_dist
            within_thresh = [i for i in sorted_idx if distances[i] <= thresh]
            if within_thresh:
                chosen_idx = within_thresh[: self.top_k]
            else:
                chosen_idx = sorted_idx[: self.top_k]
        else:
            chosen_idx = sorted_idx[: self.top_k]

        results: tp.List[tp.Dict[str, tp.Any]] = []

        trial_map = {getattr(t, "number", getattr(t, "_trial_id", None)): t for t in trials}

        for idx in chosen_idx:
            tn = int(trials_good[idx])
            t_obj = trial_map.get(tn)
            params = getattr(t_obj, "params", {}) if t_obj is not None else {}

            delta = {}
            for key, value in params.items():
                value_best = best_params.get(key, None)
                delta_value = value - value_best
                delta[key] = delta_value

            results.append(
                {
                    "trial_number": tn,
                    "params": params,
                    "delta": delta,
                    "loss": float(L_good[idx]),
                    "distance": float(distances[idx]),
                }
            )
        return results