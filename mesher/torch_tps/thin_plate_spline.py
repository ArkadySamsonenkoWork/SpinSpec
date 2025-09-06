from __future__ import annotations

import torch
import torch.linalg

from sklearn.metrics import pairwise_distances
import numpy as np

from .polynomial_transform import PolynomialFeatures


class Metric:
    def __call__(self, *args):
        point_1 = args[0]
        point_2 = args[1]
        delta_phi = abs(point_2[..., 1] - point_1[..., 1])
        if delta_phi > np.pi:
            delta_phi = 2 * np.pi - delta_phi

        return ((point_2[0] - point_1[0]) ** 2 + delta_phi ** 2) ** 0.5


class ThinPlateSpline:
    """Solve the Thin Plate Spline interpolation.

    In fact, this class supports the more general multi-dimensional polyharmonic spline interpolation.

    It learns a non-linear elastic mapping function f: R^d -> R given n control points (x_i, y_i)_i
    where x_i \\in R^d and y_i \\in R.

    The problem is formally stated as:
    f = min_f E_{fit}(f) + \\alpha E_{reg}(f)      (1)

    where E_{fit}(f) = \\sum_{i=1}^n (y_i - f(x_i))^2  (Least square objectif)
    and E_{reg}(f) = \\int \\|\\nabla^{order} f \\|_2^2 dx (Regularization)
    which penalizes the norm of the order-th derivatives of the learnt function.

    When the regularization alpha is 0, then this becomes equivalent to minimizing E_{reg}
    under the constraints f(x_i) = y_i.

    NOTE: In the classical 2 dimensionnal TPS, the spline order is 2.

    Using Euler-Lagrange, one can show that the function solutions are of the form:
    f(x) = P(x) + \\sum_i w_i G(\\|x - x_i\\|_2), with P a polynom of degree = order - 1
    and G are radial basis functions (RBF).

    It can be shown that the RBF follows:

    G(r) = r**(2 * order - d) log(r) for even dimension d
    or
    G(r) = r**(2 * order - d) for odd dimension d

    For TPS, this sets G(r) = r**2 log(r)
    As RBF, are not always defined in r = 0 (if 2 * order \\le d), we decided to use the common
    (but incorrect) TPS kernel instead of the mathematical one.

    Fitting the spline, amounts at finding the polynoms coefficients and the weights (w_i).

    Vectorized solution:
    --------------------

    Let X = (x_i) \\in R^{n x d} and Y = (y_i) \\in R^{n x v} be the control points and associated values.
    And X' = (x'_i) \\in R^{n' x d} be any set of points where we want to interpolate f.

    Note that we now have several values to interpolate for each position, which simply amounts at finding
    a mapping f_j for each value and concatenating them.

    X and X' can expended as its polynomial features X_p \\in R^{n x d_p} and X'_p \\in R^{n' x d_p}.
    For instance, TPS, the polynom is of degree one and the polynomial extension is simply the homogenous
    coordinates: X_p = (1, X). See PolynomialFeatures for more information.

    Let's call K(X') \\in R^{n' x n} the matrix of RBF values for any X'. K(X') = G(cdist(X', X)): it is the
    RBF applied to the radial distances to the control points.

    The polynoms coefficients can be vectorized into a matrix C \\in R^{d_p x v} (one polynom for each value)
    and finally we can also vectorize the RBF weights into W \\in R^{n x v} (One weight for each control point
    and each value).

    The function f can now be vectorized as:
    f(X') = X'_p C + K(X') W

    Learning the parameters C and W is done by solving the following system with the control points X:
            A      .   P   =   B
                          <=>
    |  K(X) , X_p |  | W |   |Y |
    |             |  |   | = |  |       (2)
    | X_p^T ,  0  |  | C |   |0 |

    The first row enforces f(X) = Y and the second adds orthogonality constraints.
    The system can be relaxed when \\alpha != 0, then one can show that same system can be solved
    by simply replacing K(X) by K(X) + alpha I.

    To transform X', one can simply apply the learned parameters P = (W, C) with
    Y' = f(X') = X'_p C + K(X') W

    In our notations, we have:
    A \\in R^{(n + d_p) x (n + d_p)}
    P \\in R^{(n + d_p) x v}
    Y \\in R^{(n + d_p) x v}

    Attrs:
        alpha (float): Regularization parameter
            Default: 0.0 (The mapping will enforce f(X) = Y)
        order (int): Order of the spline (minimizes the squared norm of the order-th derivatives)
            Default: 2 (Suited for TPS interpolation)
        enforce_tps_kernel (bool): Always use the RBF G(r) = r**2 log(r).
            This should be sub-optimal, but it yields good results in practice.
            Default: False
        device (torch.device): Torch device to run on (cuda for nvidia gpu)
            Default: cpu
        parameters (torch.Tensor): All the parameters P = (W, C). Shape: (n + d_p, v)
        control_points (torch.Tensor): Control points fitted (the last X given to fit). Shape: (n, d)
    """

    eps = 0  # Relaxed orthogonality constraints by setting it to 1e-6 ?

    def __init__(self, init_vertices, extended_vertices, alpha=0.2, order=1, enforce_tps_kernel=False, device="cpu") ->\
            None:
        self._fitted = False
        self._polynomial_features = PolynomialFeatures(order - 1)

        self.init_vertices = self._to_lat_long(init_vertices)
        self.extended_vertices = self._to_lat_long(extended_vertices)
        # theta, phi
        self.alpha = alpha
        self.order = order
        self.enforce_tps_kernel = enforce_tps_kernel
        self.device = torch.device(device)

        self.parameters = torch.tensor([], dtype=torch.float32)
        self.control_points = torch.tensor([], dtype=torch.float32)

        self._precompute(self.init_vertices, self.extended_vertices)

    def _to_lat_long(self, array: list[tuple[float, float]]):
        array = np.array(array).copy()[:, ::-1]
        array = np.array(array)
        array[:, 0] = np.pi / 2 - array[:, 0]
        array[:, 1] = array[:, 1] - np.pi
        return array

    def _precompute(self, init_vertices: np.ndarray, extended_vertices: np.ndarray):
        n, _ = init_vertices.shape  # (n, d)
        # Compute radial distances
        phi = torch.from_numpy(self._radial_distance(init_vertices)).to(dtype=torch.float32)  # (n, n) (phi = K(X))
        init_vertices = torch.tensor(init_vertices.copy(), dtype=torch.float32)

        # Polynomial expansion
        X_p = self._polynomial_features.fit(init_vertices).transform(init_vertices)  # (n, d_p)

        # Assemble system A P = B
        # A = |K + alpha I, X_p|
        #     |     X_p.T ,  0 |
        A = torch.vstack(  # ((n + d_p), (n + d_p))
            [
                torch.hstack(  # (n, (n + d_p))
                    [phi + self.alpha * torch.eye(n, dtype=init_vertices.dtype, device=self.device), X_p]
                ),
                torch.hstack(  # (d_p, (n + d_p))
                    [X_p.T, self.eps * torch.eye(X_p.shape[1], dtype=init_vertices.dtype, device=self.device)]
                ),
            ]
        )

        self.A = A
        self.X_p_shape = X_p.shape
        #self.L = torch.linalg.inv(A)
        # Compute radial distances

        phi_epxand = torch.from_numpy(self._radial_distance(extended_vertices)).to(dtype=torch.float32)  # (n', n)
        extended_vertices = torch.tensor(extended_vertices.copy(), dtype=torch.float32)

        # Polynomial expansion
        X_p_expand = self._polynomial_features.transform(extended_vertices)  # (n', d_p)

        # Compute f(X)
        X_aug_expand = torch.hstack([phi_epxand, X_p_expand])  # (n', (n + d_p))\
        self.X_aug_expand = X_aug_expand
        self.L = torch.linalg.pinv(self.A)

    def __call__(self, Y: torch.Tensor) -> torch.Tensor:
        Y = Y.transpose(-1, -2)

        B = torch.vstack(  # ((n + d_p), v)
            [Y, torch.zeros((self.X_p_shape[1], Y.shape[1]), dtype=Y.dtype, device=self.device)]
        )
        self.parameters = torch.linalg.solve(self.A, B)  # pylint: disable=not-callable
        #self.parameters = self.L @ B

        return self.X_aug_expand @ self.parameters  # (n', v)

    def _radial_distance(self, X: np.ndarray) -> torch.Tensor:
        """Compute the pairwise RBF values for the given points to the control points

        Args:
            X (torch.Tensor): Points to be interpolated
                Shape: (n', d)

        Returns:
            torch.Tensor: The RBF evaluated for each point from a control point (K(X))
                Shape: (n', n)
        """
        dist = pairwise_distances(X, self.init_vertices, metric="haversine")
        # Don't use mm for euclid dist, lots of imprecision comes from it (Will be a bit slower)
        #dist = torch.cdist(X, self.control_points, compute_mode="donot_use_mm_for_euclid_dist")

        power = self.order * 2 - self.control_points.shape[-1]

        # As negatif power leads to ill-defined RBF at r = 0
        # We use the TPS RBF r^2 log r, though it is not optimal
        if power <= 0 or self.enforce_tps_kernel:
            power = 2  # Use r^2 log r (TPS kernel)

        if power % 2:  # Odd
            return dist**power

        # Even
        dist[dist == 0] = 1  # phi(r) = r^power log(r) ->  (phi(0) = 0)
        return dist**power * np.log(dist + 1e-8)


def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure that tensor is a 2d tensor

    In case of 1d tensor, let's expand the last dim
    """
    assert tensor.ndim in (1, 2)

    # Expand last dim in order to interpret this as (n, 1) points
    if tensor.ndim == 1:
        tensor = tensor[:, None]

    return tensor
