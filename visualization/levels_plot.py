from scipy.optimize import linear_sum_assignment
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_energy_system(B: torch.Tensor, energies: torch.Tensor,
                       vecs: torch.Tensor, levels: list[int], saved_order=False) -> None:
    """
    :param B: magnetic field range in T. The shape is [num_points, 1, 1]
    :param energies: The energies of the levels. The shape is [num_points, spin_dimension]
    :param vecs: The eigen vectors of the levels. The shape is [num_points, spin_dimension, spin_dimension]
    :param levels: The numbers of the levels to be plotted on the graph
    :param saved_order:
    If it is True, than the color of energy levels do not change after overlap
    If it is False, the order of energy levels is always descending
    :return: None
    """

    vecs = vecs.numpy()
    energies = energies.numpy()
    B = B.numpy()

    if saved_order:
        energies = get_saved_order(energies, vecs)
    else:
        pass

    plt.figure(figsize=(6, 4))
    for i in levels:
        plt.plot(B.squeeze(), energies[:, i], linewidth=1.2)
    plt.xlabel("Field (T)")
    plt.ylabel("Energy (Hz)")
    plt.tight_layout()



def get_saved_order(energies: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """
    :param energies: The energies of the levels. The shape is [num_points, spin_dimension]
    :param vecs: The eigen vectors of the levels. The shape is [num_points, spin_dimension, spin_dimension]
    :return: tracked_eps: energies in saved order. The shape is [num_points, spin_dimension]
    """
    nB, dim = energies.shape
    tracked_eps = np.zeros_like(energies)
    tracked_eps[0] = energies[0]

    tracked_vecs = np.zeros_like(vecs)
    tracked_vecs[0] = vecs[0]

    for b in range(1, nB):
        prev_vecs = tracked_vecs[b - 1]
        curr_vecs = vecs[b]

        overlap = np.abs(prev_vecs.conj().T @ curr_vecs)

        row_idx, col_idx = linear_sum_assignment(-overlap)

        tracked_eps[b, row_idx] = energies[b, col_idx]
        tracked_vecs[b] = curr_vecs[:, col_idx]

    return tracked_eps
