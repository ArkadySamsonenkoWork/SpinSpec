

# ===== 1. Build your (phi, theta) grid and f values =====
def create_spherical_grid(K):
    """
    Generate phi and theta grids based on the given parameterization.

    Parameters:
    K (int): Grid resolution parameter.

    Returns:
    phi (torch.Tensor): 1D tensor of phi values.
    theta (torch.Tensor): 1D tensor of theta values.
    grid_indices (list): List of (q, k) pairs for triangular mesh.
    """
    phi_list = []
    theta_list = []
    grid_indices = []

    for k in range(K):
        theta_k = torch.pi / 2 * k / (K - 1) if K > 1 else 0.0
        for q in range(k + 1):
            phi_qk = torch.pi / 2 * q / k if k > 0 else 0.0
            phi_list.append(phi_qk)
            theta_list.append(theta_k)
            grid_indices.append((q, k))

    phi = torch.tensor(phi_list, dtype=torch.float32)
    theta = torch.tensor(theta_list, dtype=torch.float32)
    return phi, theta, grid_indices


def create_bivariate_spline(phi, theta, f, K):
    """
    Create a bivariate tensor product spline with reflection boundary conditions.

    Parameters:
    phi (torch.Tensor): 1D tensor of phi angles.
    theta (torch.Tensor): 1D tensor of theta angles.
    f (torch.Tensor): 2D tensor of function values f(phi, theta).
    K (int): Grid resolution parameter.

    Returns:
    spline_phi: NaturalCubicSpline for phi direction.
    spline_theta: NaturalCubicSpline for theta direction.
    """
    # Sort phi and theta to ensure monotonicity
    phi, phi_indices = torch.sort(phi)
    theta, theta_indices = torch.sort(theta)

    # Since f is 2D, we need to align it with sorted indices (assuming f matches phi, theta order)
    # For simplicity, assume f is provided in the order of grid_indices

    # Apply reflection boundary conditions
    phi_reflected = torch.cat([phi, 2 * phi[-1] - phi.flip(0)])
    theta_reflected = torch.cat([theta, 2 * theta[-1] - theta.flip(0)])

    # Extend f for reflection (assuming f is a 2D tensor matching the grid)
    f_reflected_phi = torch.cat([f, f.flip(0)], dim=0)
    f_reflected = torch.cat([f_reflected_phi, f_reflected_phi.flip(1)], dim=1)

    # Compute spline coefficients
    # For phi direction: interpolate along each theta
    coeffs_phi = natural_cubic_spline_coeffs(phi_reflected, f_reflected)
    # For theta direction: interpolate along each phi
    coeffs_theta = natural_cubic_spline_coeffs(theta_reflected, f_reflected.T)

    # Create spline objects
    spline_phi = NaturalCubicSpline(coeffs_phi)
    spline_theta = NaturalCubicSpline(coeffs_theta)

    return spline_phi, spline_theta


def evaluate_bivariate_spline(spline_phi, spline_theta, phi_eval, theta_eval):
    """
    Evaluate the bivariate spline at given phi and theta points.

    Parameters:
    spline_phi: NaturalCubicSpline for phi direction.
    spline_theta: NaturalCubicSpline for theta direction.
    phi_eval (torch.Tensor): 1D tensor of phi angles to evaluate.
    theta_eval (torch.Tensor): 1D tensor of theta angles to evaluate.

    Returns:
    f_eval (torch.Tensor): 2D tensor of interpolated values.
    """
    # Evaluate spline in phi direction for all theta_eval
    f_phi = spline_phi.evaluate(phi_eval)
    # Evaluate spline in theta direction for all phi_eval
    f_theta = spline_theta.evaluate(theta_eval)

    # Tensor product: outer combination of phi and theta interpolations
    # Reshape to enable broadcasting
    f_eval = f_phi[:, None] * f_theta[None, :]

    return f_eval