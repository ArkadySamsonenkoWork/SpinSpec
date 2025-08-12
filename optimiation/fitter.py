import dataclasses
from dataclasses import dataclass
import typing as tp
import math
import time

import nevergrad as ng
import numpy as np
import torch
import optuna
from sklearn.cluster import DBSCAN


from ..spectra_processing import normalize_spectrum
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
class MinimaCluster:
    center_params: tp.Dict[str, float]
    best_loss: float
    members: tp.List[tp.Dict[str, float]]


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

        for name in list(self.fixed_params.keys()):
            if name in self.varying_names:
                idx = next(i for i, s in enumerate(self._varying_specs) if s.name == name)
                del self._varying_specs[idx]
                self.varying_names.remove(name)

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

    def unfreeze(self, name: str):
        """Unfreeze a parameter previously frozen with `freeze` or fixed_params."""
        if name in self.fixed_params:
            del self.fixed_params[name]

        for s in self.specs:
            if s.name == name and s not in self._varying_specs and getattr(s, 'vary', True):
                self._varying_specs.append(s)
                self.varying_names.append(s.name)

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

    def suggest_optuna(self, trial) -> tp.Dict[str, float]:
        out = dict(self.fixed_params)  # start with fixed
        for s in self._varying_specs:
            lo, hi = s.bounds
            val = trial.suggest_float(s.name, lo, hi)
            out[s.name] = s.apply(val)
        return out

    # Nevergrad instrumentation helper - only instrument varying parameters
    def instrument_nevergrad(self) -> "ng.p.instrumentation.Instrumentation":
        instr = ng.p.Instrumentation()
        for s in self._varying_specs:
            lo, hi = s.bounds
            instr *= ng.p.Scalar(lower=lo, upper=hi)
        return instr


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

    def __init__(
        self,
        B: tp.Union[np.ndarray, torch.Tensor],
        y_exp: tp.Union[np.ndarray, torch.Tensor],
        param_space: ParameterSpace,
        simulate_spectrum_callable: tp.Optional[tp.Callable[[tp.Dict[str, float]], torch.Tensor]] = None,
        norm_mode: str = "integral",
        device: tp.Optional[torch.device] = None,
        objective=objectives.MSEObjective(),
    ):
        self.device = torch.device("cpu") if device is None else device
        self.B = torch.tensor(B, dtype=torch.float32, device=self.device)
        self.y_exp = torch.tensor(y_exp, dtype=torch.float32, device=self.device)
        self.norm_mode = norm_mode
        self.y_exp = normalize_spectrum(self.B, self.y_exp, mode=norm_mode)
        self.param_space = param_space
        self._objective = objective
        self._simulate_callable = simulate_spectrum_callable

    def simulate_spectrum(self, params: tp.Dict[str, float]) -> torch.Tensor:
        return normalize_spectrum(self.B, self._simulate_callable(params), mode=self.norm_mode)

    def loss_from_params(self, params: tp.Dict[str, float]) -> torch.Tensor:
        """Compute model - experiment residuals as a torch.Tensor."""
        with torch.no_grad():
            model = self.simulate_spectrum(params)
        return self._objective(model, self.y_exp)

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
    ) -> FitResult:
        """Fit using Optuna.

        Requires optuna to be installed.
        """
        def objective(trial):
            p = self.param_space.suggest_optuna(trial)
            loss = self.loss_from_params(p)
            return loss

        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler, study_name=study_name, load_if_exists=True)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=show_progress)

        best_params = {k: float(v) for k, v in study.best_params.items()}
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectrum(best_params)
        return FitResult(best_params, float(study.best_value), best_spec, {"backend": "optuna", "study": study})


    def fit_nevergrad(
        self,
        budget: int = 200,
        optimizer_name: str = "TwoPointsDE",
        seed: tp.Optional[int] = None,
        return_best_spectrum: bool = True,
    ) -> FitResult:
        """Fit using Nevergrad (if installed)."""
        if ng is None:
            raise RuntimeError("Nevergrad is required for fit_nevergrad but not installed")

        instr = self.param_space.instrument_nevergrad()
        if seed is not None:
            ng.optimizers.registry.seed(seed)
        opt = ng.optimizers.registry[optimizer_name](instrumentation=instr, budget=budget)

        def _loss_from_tuple(*args):
            params = self.param_space.vector_to_dict(args)
            return self.loss_from_params(params)

        recommendation = opt.minimize(_loss_from_tuple)
        x = recommendation.value
        if not isinstance(x, (list, tuple, np.ndarray)):
            x = (x,)
        best_params = self.param_space.vector_to_dict(x)
        best_spec = None
        if return_best_spectrum:
            best_spec = self.simulate_spectrum(best_params)
        return FitResult(
            best_params, self.loss_from_params(best_params), best_spec,
            {"backend": "nevergrad", "optimizer": optimizer_name}
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


class SpaceSearcher(SpectrumFitter):
    def search(self, n_trials=200, eps=0.05, min_samples=3, **kwargs) -> tp.List[MinimaCluster]:
        """Explore space and find multiple local minima."""
        fit_results = self.fit_optuna(n_trials=n_trials, **kwargs)
        study = fit_results.optimizer_info["study"]

        X = np.array([self.param_space.dict_to_vector(t.params) for t in study.trials])
        losses = np.array([t.value for t in study.trials])

        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        clusters = []
        for label in set(clustering.labels_):
            if label == -1:
                continue
            indices = np.where(clustering.labels_ == label)[0]
            cluster_losses = losses[indices]
            cluster_params = [study.trials[i].params for i in indices]
            best_idx = indices[np.argmin(cluster_losses)]
            clusters.append(
                MinimaCluster(
                    center_params=study.trials[best_idx].params,
                    best_loss=float(np.min(cluster_losses)),
                    members=cluster_params
                )
            )
        return clusters