from dataclasses import dataclass
import typing as tp
import sys

import nevergrad as ng
import numpy as np
import torch
import optuna
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


sys.path.append("..")
from spectra_processing import normalize_spectrum
from optimiation import objectives


@dataclass
class ParamSpec:
    """Specification for a single scalar parameter.

    Attributes:
        name: parameter name
        bounds: (low, high) bounds for optimizer search (floats)
        default: optional default value to use for initialization
        transform: optional callable applied to a raw optimizer value to map
                   it to the physical parameter (useful for log-scales)
    """
    name: str
    bounds: tp.Tuple[float, float]
    default: tp.Optional[float] = None
    transform: tp.Optional[tp.Callable[[float], float]] = None

    def clip(self, x: float) -> float:
        lo, hi = self.bounds
        return float(min(max(x, lo), hi))

    def apply(self, x: float) -> float:
        x = self.clip(x)
        return self.transform(x) if self.transform is not None else x


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
class MinimaCluster:
    center_params: tp.Dict[str, float]
    best_loss: float
    members: tp.List[tp.Dict[str, float]]


@dataclass
class NevergradTrial:
    params: tp.Dict[str, float]
    value: float
    _trial_id: int


class ParameterSpace:
    """Helper to manage a set of ParamSpec and conversions.
    Stores parameters in a fixed order. Supports parameters that are
    fixed (not varied during optimization) and variable parameters. Fixed
    parameters can be provided either by using ParamSpec(vary=False) or
    via the `fixed_params` mapping.
    """

    def __init__(self, specs: tp.Sequence[ParamSpec],
                 fixed_params: tp.Optional[tp.Dict[str, float]] = None):
        self.specs = list(specs)
        self.names = [s.name for s in self.specs]

        self.fixed_params: tp.Dict[str, float] = {} if fixed_params is None else dict(fixed_params)
        self._varying_specs = [s for s in self.specs if getattr(s, 'vary', True)]
        self.varying_names = [s.name for s in self._varying_specs]

        self.varying_params = {s.name: s.default for s in self._varying_specs}

        for name in list(self.fixed_params.keys()):
            if name in self.varying_names:
                idx = next(i for i, s in enumerate(self._varying_specs) if s.name == name)
                del self._varying_specs[idx]
                self.varying_names.remove(name)

    def __getitem__(self, key: str):
        try:
            return self.fixed_params[key]
        except KeyError:
            try:
                return self.varying_params[key]
            except KeyError:
                raise KeyError(f"Key '{key}' not found in fixed_params or _varying_specs")

    def freeze(self, name: str, value: tp.Optional[float] = None):
        """
        Freeze a parameter by name. If value is provided, use it; otherwise
        use its default (or current) value.
        """
        if name not in self.names:
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

    def set(self, params: dict[str, float]):
        for key, value in params.items():
            if key not in self.names:
                raise KeyError(f"Parameter '{key}' not found in parameter space")
            if key in self.fixed_params:
                self.fixed_params[key] = value
            elif key in self.varying_names:
                for spec in self._varying_specs:
                    if spec.name == key:
                        spec.default = value
                        break

    def __dict__(self) -> dict[str, float]:
        return {**self.varying_params, **self.fixed_params}

    def __iter__(self):
        return iter(self.__dict__().items())

    def suggest_optuna(self, trial) -> tp.Dict[str, float]:
        out = dict(self.fixed_params)  # start with fixed
        for s in self._varying_specs:
            lo, hi = s.bounds
            val = trial.suggest_float(s.name, lo, hi)
            out[s.name] = s.apply(val)
        return out

    # Nevergrad instrumentation helper - only instrument varying parameters
    def instrument_nevergrad(self) -> ng.p.Instrumentation:
        params = []
        for s in self._varying_specs:
            lo, hi = s.bounds
            params.append(ng.p.Scalar(lower=lo, upper=hi))
        return ng.p.Instrumentation(*params)


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
        B: tp.Union[np.ndarray, torch.Tensor] | list[tp.Union[np.ndarray, torch.Tensor]],
        y_exp: tp.Union[np.ndarray, torch.Tensor] | list[tp.Union[np.ndarray, torch.Tensor]],
        param_space: ParameterSpace,
        simulate_spectrum_callable: tp.Callable[
            [list[torch.Tensor] | torch.Tensor, tp.Dict[str, float], tp.Dict],
             torch.Tensor | list[torch.Tensor]
        ],
        norm_mode: str = "integral",
        device: tp.Optional[torch.device] = None,
        objective=objectives.MSEObjective(),
        weights: list[float] = None
    ):
        self.device = torch.device("cpu") if device is None else device
        self.norm_mode = norm_mode
        self._simulate_callable = simulate_spectrum_callable

        self.B, self.y_exp, self.multisample = self._set_experemental(B, y_exp)

        if self.multisample and (weights is None):
            self.weights = torch.ones(len(self.B), dtype=torch.float32, device=self.device)
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

    def simulate_single_spectrum(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        return normalize_spectrum(self.B, self._simulate_callable(self.B, params, **kwargs), mode=self.norm_mode)

    def simulate_spectral_set(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor]:
        models = self._simulate_callable(self.B, params, **kwargs)
        for idx in range(len(models)):
            models[idx] = normalize_spectrum(self.B[idx], models[idx], mode=self.norm_mode)
        return models

    def simulate_spectroscopic_data(self, params: tp.Dict[str, float], **kwargs) -> list[torch.Tensor] | torch.Tensor:
        if self.multisample:
            model = self.simulate_spectral_set(params, **kwargs)
        else:
            model = self.simulate_single_spectrum(params, **kwargs)
        return model

    def simulate_spectra_from_trial_params(self, trial_params: tp.Dict[str, float], **kwargs) ->\
            list[torch.Tensor] | torch.Tensor:
        return self.simulate_spectroscopic_data({**self.param_space.fixed_params, **trial_params}, **kwargs)

    def loss_from_params(self, params: tp.Dict[str, float], **kwargs) -> torch.Tensor:
        """Compute model - experiment residuals as a torch.Tensor."""
        with torch.no_grad():
            if self.multisample:
                models = self.simulate_spectral_set(params, **kwargs)
                loss = sum(self.weights[idx] * self._loss_normalization[idx] * self._objective(
                    models[idx], self.y_exp[idx]) for idx in range(len(models))) / len(models)
            else:
                model = self.simulate_single_spectrum(params, **kwargs)
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
        n_trials: int = 100,
        timeout: tp.Optional[float] = None,
        n_jobs: int = 1,
        sampler: tp.Optional[optuna.samplers.BaseSampler] = None,
        study_name: tp.Optional[str] = None,
        show_progress: bool = True,
        seed: tp.Optional[int] = None,
        return_best_spectrum: bool = True,
            **kwargs,
    ) -> FitResult:
        """Fit using Optuna.

        Requires optuna to be installed.
        """
        def loss_function(trial):
            p = self.param_space.suggest_optuna(trial)
            loss = self.loss_from_params(p, **kwargs)
            return loss

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
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
        budget: int = 200,
        optimizer_name: str = "TwoPointsDE",
        seed: tp.Optional[int] = None,
        show_progress: bool = True,
        return_best_spectrum: bool = True,
        track_trials = True,
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
            return self.loss_from_params(params).item()

        if show_progress:
            opt.register_callback("tell", ng.callbacks.ProgressBar())

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

        if track_trials:
            trials = self._tracker_to_trials(trials_tracker)

        return FitResult(
            best_params, self.loss_from_params({**self.param_space.fixed_params, **best_params}), best_spec,
            {"backend": "nevergrad", "optimizer": optimizer_name, "trials": trials}
        )

    def fit(
        self,
        method: str = "optuna",
        **kwargs,
    ) -> FitResult:
        method = method.lower()
        if method == "optuna":
            return self.fit_optuna(**kwargs)
        if method in ("nevergrad", "ng"):
            return self.fit_nevergrad(**kwargs)
        raise ValueError(f"Unknown fit method: {method}")


class SpaceSearcher:
    def __init__(
        self,
        loss_rel_tol: float = 0.5,
        top_k: int = 5,
        distance_fraction: float = 0.2,
    ):
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
        If top_k is given, returns only top_k lowest-loss trials.
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
        trials, param_names = self._extract_trials_from_fit(fit_result, param_names)
        P, L, trial_numbers = self._parse_trials(trials, param_names)

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
            results.append(
                {
                    "trial_number": tn,
                    "params": params,
                    "loss": float(L_good[idx]),
                    "distance": float(distances[idx]),
                }
            )
        return results