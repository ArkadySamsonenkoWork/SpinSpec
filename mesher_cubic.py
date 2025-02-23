import torch


def cubic_interpolate(mesh, f, new_points):
    """
    Perform cubic RBF interpolation on non-uniform 2D data.

    Args:
        mesh (torch.Tensor): Original mesh points of shape (M, 2).
        f (torch.Tensor): Function values at mesh points of shape (M,).
        new_points (torch.Tensor): Query points of shape (N, 2) to interpolate.

    Returns:
        torch.Tensor: Interpolated values at new_points, shape (N,).
    """
    M = mesh.size(0)
    device = mesh.device
    dtype = mesh.dtype

    # Compute pairwise distances between original points
    dists = torch.cdist(mesh, mesh)  # M x M

    # Cubic kernel matrix (r^3)
    A = dists ** 3

    # Construct polynomial terms matrix [1, x, y]
    P = torch.cat([torch.ones(M, 1, device=device, dtype=dtype), mesh], dim=1)  # M x 3

    # Form the block matrix [[A, P], [P^T, 0]]
    top = torch.cat([A, P], dim=1)  # M x (M + 3)
    bottom = torch.cat([P.T, torch.zeros(3, 3, device=device, dtype=dtype)], dim=1)  # 3 x (M + 3)
    full_matrix = torch.cat([top, bottom], dim=0)  # (M + 3) x (M + 3)

    # Form the right-hand side vector [f, 0, 0, 0]
    rhs = torch.cat([f, torch.zeros(3, device=device, dtype=dtype)])

    # Solve the linear system
    try:
        weights_coeffs = torch.linalg.solve(full_matrix, rhs)
    except torch.linalg.LinAlgError:
        # Add a small regularization to handle singular matrices
        full_matrix += torch.eye(full_matrix.size(0), device=device, dtype=dtype) * 1e-9
        weights_coeffs = torch.linalg.solve(full_matrix, rhs)

    w = weights_coeffs[:M]
    a = weights_coeffs[M:]

    # Compute distances from new_points to original mesh
    new_dists = torch.cdist(new_points, mesh)  # N x M

    # Cubic kernel evaluation for new points
    new_A = new_dists ** 3

    # Polynomial terms for new points
    new_P = torch.cat([torch.ones(new_points.size(0), 1, device=device, dtype=dtype), new_points], dim=1)

    # Calculate interpolated values
    interpolated = torch.mm(new_A, w.unsqueeze(1)).squeeze() + torch.mv(new_P, a)

    return interpolated


# Example usage:
if __name__ == "__main__":
    # Original mesh and function values
    mesh = torch.tensor([[0.0, 0.0],
                         [0.0, 1.0],
                         [0.5, 1.0],
                         [1.0, 1.0]], dtype=torch.float32)
    f = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

    # Query points
    new_points = torch.tensor([[0.2, 0.5],
                               [0.5, 1.0]], dtype=torch.float32)

    # Perform interpolation
    interpolated_values = cubic_interpolate(mesh, f, new_points)
    print("Interpolated values:", interpolated_values)