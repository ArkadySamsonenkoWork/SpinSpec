import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.patches import Rectangle
import seaborn as sns


x_delta = 0.2

def _get_energy_positions(E: np.ndarray, power: float = 1/2):
    order = np.argsort(E)
    E_sorted = E[order]
    delta_E = np.diff(E_sorted)
    delta_scaled = np.power(np.abs(delta_E), power)
    scaled_E_sorted = np.concatenate([[0], np.cumsum(delta_scaled)])
    E_scaled = np.zeros_like(E)
    E_scaled[order] = scaled_E_sorted
    E_scaled = E_scaled / max(E_scaled)
    return E_scaled


def _compute_pair_location(spin_dim: int):
    pair_offsets = {}
    tot_transitions = spin_dim * (spin_dim - 1) // 2
    offset_base = 2 * x_delta / tot_transitions
    for idx, (i, j) in enumerate(combinations(range(spin_dim), 2)):
        pair_offsets[frozenset({i, j})] = offset_base * (idx - (spin_dim * (spin_dim - 1) // 4) + 1/2)
    return pair_offsets


def _plot_energy_levels(E_scaled: np.ndarray, E: np.ndarray):
    spin_dim = E_scaled.shape[-1]
    for i in range(spin_dim):
        plt.hlines(E_scaled[i], -x_delta, x_delta, color='black')
        plt.text(x_delta + 0.08, E_scaled[i], f"E={E[i]:.2g} (Hz)", va='center',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))


def _point_transitions(pair_offsets: dict[frozenset[int, int]],
                       transition_matrix: np.ndarray,
                       E_scaled: np.ndarray
                       ):
    spin_dim = E_scaled.shape[-1]
    for i in range(spin_dim):
        for j in range(spin_dim):
            if i != j and transition_matrix[i, j] != 0:
                pair_key = frozenset({i, j})
                x_shift = pair_offsets[pair_key]

                x = x_shift
                y_start = E_scaled[j]
                y_end = E_scaled[i]

                arrow_padding = 0.005
                if y_end > y_start:
                    y_end_adjusted = y_end - arrow_padding
                else:
                    y_end_adjusted = y_end + arrow_padding

                if i > j:
                    x += 0.01
                    y_mid = (y_start + y_end) / 2
                    y_delta = (y_end - y_start)

                    y_text = y_mid + np.clip(y_delta / 6, -0.05, 0.05)
                    plt.text(x - 0.01, y_text, f"{abs(transition_matrix[i, j]):.2e} (1/s)", fontsize=7,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
                else:
                    x -= 0.01
                    y_mid = (y_start + y_end) / 2
                    y_delta = (y_end - y_start)
                    y_text = y_mid + np.clip(y_delta / 6, -0.05, 0.05)

                    plt.text(x - 0.01, y_text, f"{abs(transition_matrix[i, j]):.2e} (1/s)", fontsize=7,
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=1))
                plt.arrow(
                    x, y_start, 0, y_end_adjusted - y_start,
                    head_width=0.005, head_length=0.005, width=0.0005,
                    length_includes_head=True,
                    color='blue',
                )


def _get_out_rates(transition_matrix: np.ndarray):
    inner_rates = -transition_matrix.sum(axis=-2)
    return -inner_rates


def _point_out_transitions(transition_matrix: np.ndarray, E_scaled: np.ndarray) -> None:
    """
    Plot arrows representing net population flow (in/out) at each energy level.
    :param transition_matrix: Matrix of transition rates between energy levels
    :param E_scaled: Array of scaled energy positions for visualization
    :return:
    """

    out_rates = _get_out_rates(transition_matrix)
    threshold = np.min(
        np.abs(transition_matrix[np.nonzero(transition_matrix)])
    ) / 1_000

    spin_dim = E_scaled.shape[-1]

    x_position = x_delta + 0.01

    energy_gaps = []
    for i in range(spin_dim - 1):
        energy_gaps.append(E_scaled[i + 1] - E_scaled[i])
    min_gap = min(energy_gaps) if energy_gaps else 0.05

    arrow_length = 0.06
    y_offset = min(0.02, min_gap / 3)

    for i in range(spin_dim):
        if abs(out_rates[i]) < threshold:
            continue

        if out_rates[i] > 0:
            plt.arrow(
                x_position + arrow_length, E_scaled[i] + y_offset,
                -arrow_length, -y_offset,
                head_width=0.005, head_length=0.005, width=0.001,
                length_includes_head=True,
                color='green',
            )
            plt.text(x_position + 0.01, E_scaled[i],
                     f"+{out_rates[i]:.2e} (1/s)", fontsize=8, va='center',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

        else:
            plt.arrow(
                x_position, E_scaled[i],
                arrow_length, y_offset,
                head_width=0.005, head_length=0.005, width=0.001,
                length_includes_head=True,
                color='red',
            )
            plt.text(x_position + 0.01, E_scaled[i],
                     f"-{abs(out_rates[i]):.2e} (1/s)", fontsize=8, va='center',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))


def plot_transitions_diagram(energy: np.ndarray, transition_matrix: np.ndarray, energy_scale: float = 1/2) -> None:
    E_scaled = _get_energy_positions(energy, energy_scale)
    spin_dim = E_scaled.shape[-1]
    pair_offsets = _compute_pair_location(spin_dim)
    plt.figure(figsize=(14, 9))
    _plot_energy_levels(E_scaled, energy)
    _point_transitions(pair_offsets, transition_matrix, E_scaled)
    _point_out_transitions(transition_matrix, E_scaled)

    plt.ylim(min(E_scaled) - 0.01, max(E_scaled) + 0.01)
    plt.axis('off')
    plt.title("Energy Levels and Paired Transitions")
    plt.show()


def plot_transitions_colormap(transition_matrix: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        transition_matrix, cmap='coolwarm', annot=True, fmt=".2e", center=0.0, cbar_kws={'label': 'Rate (1/s)'})
    plt.title("Transition Matrix")
    plt.xlabel("State j")
    plt.ylabel("State i")
    plt.show()
