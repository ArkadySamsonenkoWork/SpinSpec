from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from typing import Optional, Dict, Any, Tuple, List
import matplotlib.patches as patches
from matplotlib.widgets import Button
import textwrap

import plotly.graph_objs as go


"""
SpinSystemPlotter

Provides a class to convert a SpinSystem into a graph and plot it.

Features
- Nodes: electrons (e0, e1...) and nuclei (n0, n1...)
- Edges: interactions (electron-nucleus, electron-electron, nucleus-nucleus)
- Option to display tensor values (g-tensors, interaction tensors) in the plot
- Two plotting backends: matplotlib (static) and Plotly (interactive with hover)

Usage example
--------------
from SpinSystemPlotter import SpinSystemPlotter
plotter = SpinSystemPlotter(spin_system)
plotter.build_graph()
plotter.plot_matplotlib(show_tensors=True)
# or
plotter.plot_plotly(show_tensors=True)

"""
import math
import textwrap
from typing import Optional, Dict, Any

import torch


class SpinSystemPlotter:
    """Create and plot a graph representation of a SpinSystem.

    Parameters
    ----------
    spin_system: SpinSystem
        Instance of the SpinSystem class provided by the user.

    Notes
    -----
    This class expects the SpinSystem object to expose the following attributes
    (matching the user's SpinSystem posted earlier):
      - electrons: list
      - nuclei: list
      - g_tensors: list[BaseInteraction] (same length as electrons)
      - electron_nuclei: list of (e_idx, n_idx, interaction)
      - electron_electron: list of (e_idx, e_idx2, interaction)
      - nuclei_nuclei: list of (n_idx, n_idx2, interaction)

    The code attempts to be defensive about types (torch.Tensor / numpy arrays / python lists).
    """

    def __init__(self, spin_system):
        self.ss = spin_system
        self.G = nx.Graph()
        self._built = False
        self.pos: Optional[Dict[str, np.ndarray]] = None

    # -----------------------------
    # Graph construction
    # -----------------------------
    def build_graph(self):
        """Construct a NetworkX graph from the SpinSystem.

        Node names: electrons -> 'e{idx}', nuclei -> 'n{idx}'.
        Node attributes include:
            - 'type': 'electron' or 'nucleus'
            - 'obj': original particle object (if present)
            - 'g_tensor': tensor or None (for electrons)
        Edge attributes include:
            - 'itype': 'en', 'ee', 'nn'
            - 'interaction': interaction object
            - 'tensor': interaction.tensor (if available)
            - 'strained_tensor': interaction.strained_tensor (if available)
        """
        self.G = nx.Graph()

        # Electrons
        for i, e in enumerate(self.ss.electrons):
            name = f"e{i}"
            g_tensor = None
            if i < len(self.ss.g_tensors):
                g_tensor = getattr(self.ss.g_tensors[i], 'tensor', None)
            self.G.add_node(name, type='electron', obj=e, g_tensor=g_tensor)

        # Nuclei
        for j, n in enumerate(self.ss.nuclei):
            name = f"n{j}"
            self.G.add_node(name, type='nucleus', obj=n)

        # Electron-Nucleus
        for e_idx, n_idx, interaction in self.ss.electron_nuclei:
            a, b = f"e{e_idx}", f"n{n_idx}"
            tensor = getattr(interaction, 'tensor', None)
            strained = getattr(interaction, 'strained_tensor', None)
            self.G.add_edge(a, b, itype='en', interaction=interaction, tensor=tensor, strained_tensor=strained)

        # Electron-Electron
        for e1, e2, interaction in self.ss.electron_electron:
            a, b = f"e{e1}", f"e{e2}"
            tensor = getattr(interaction, 'tensor', None)
            strained = getattr(interaction, 'strained_tensor', None)
            self.G.add_edge(a, b, itype='ee', interaction=interaction, tensor=tensor, strained_tensor=strained)

        # Nucleus-Nucleus
        for n1, n2, interaction in self.ss.nuclei_nuclei:
            a, b = f"n{n1}", f"n{n2}"
            tensor = getattr(interaction, 'tensor', None)
            strained = getattr(interaction, 'strained_tensor', None)
            self.G.add_edge(a, b, itype='nn', interaction=interaction, tensor=tensor, strained_tensor=strained)

        self._built = True
        return self.G

    # -----------------------------
    # Utilities for tensor formatting
    # -----------------------------
    def _to_numpy(self, t):
        """Convert torch or numpy to numpy array (real/complex allowed)."""
        if t is None:
            return None
        if isinstance(t, torch.Tensor):
            try:
                t = t.detach().cpu().numpy()
            except Exception:
                t = np.array(t)
        elif not isinstance(t, np.ndarray):
            t = np.array(t)
        return t

    def _short_tensor_str(self, t, maxchars=200):
        """Return a short printable representation of a tensor (numpy or torch).

        If the full representation is longer than maxchars we truncate with an ellipsis.
        """
        t_np = self._to_numpy(t)
        if t_np is None:
            return "None"
        # Limit precision
        if np.iscomplexobj(t_np):
            arr_str = np.array2string(t_np, precision=3, separator=',', suppress_small=True)
        else:
            arr_str = np.array2string(t_np, precision=3, separator=',', suppress_small=True)
        if len(arr_str) <= maxchars:
            return arr_str
        # Truncate intelligently
        single_line = ' '.join(arr_str.split())
        if len(single_line) <= maxchars:
            return single_line
        return single_line[: maxchars - 3] + '...'

    # -----------------------------
    # Matplotlib plotting
    # -----------------------------
    def plot_matplotlib(self, show_tensors: bool = False, show_edge_labels: bool = True,
                        node_positions: Optional[Dict[str, np.ndarray]] = None,
                        figsize=(10, 8), node_size=800, cmap: Optional[str] = None):
        """Static plot using matplotlib + networkx.

        Parameters
        ----------
        show_tensors: bool
            If True, draw node g-tensors and edge tensors as text labels.
        show_edge_labels: bool
            If True and show_tensors is False, still show short interaction type labels.
        node_positions: dict or None
            Custom positions for nodes (mapping node->(x,y)). If None, a spring layout is used.
        """
        if plt is None:
            raise ImportError("matplotlib is required for plot_matplotlib. Install it with `pip install matplotlib`.")

        if not self._built:
            self.build_graph()

        G = self.G
        if node_positions is None:
            pos = nx.spring_layout(G, seed=42)
            self.pos = pos
        else:
            pos = node_positions
            self.pos = pos

        plt.figure(figsize=figsize)

        # node colors by type
        node_types = [G.nodes[n].get('type', 'electron') for n in G.nodes()]
        unique_types = list(sorted(set(node_types)))
        type_to_int = {t: i for i, t in enumerate(unique_types)}
        node_colors = [type_to_int[t] for t in node_types]

        if cmap is None:
            cmap = 'tab10'

        nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, cmap=cm.get_cmap(cmap))
        nx.draw_networkx_labels(G, pos, font_size=10)
        nx.draw_networkx_edges(G, pos)

        # Edge labels
        if show_tensors:
            # show full short tensor for edges
            edge_labels = {}
            for a, b, d in G.edges(data=True):
                t = d.get('tensor', None)
                edge_labels[(a, b)] = self._short_tensor_str(t, maxchars=120)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            # node labels for g-tensors
            for n, d in G.nodes(data=True):
                if d.get('type') == 'electron':
                    gt = d.get('g_tensor')
                    s = self._short_tensor_str(gt, maxchars=120)
                    x, y = pos[n]
                    plt.text(x, y - 0.08, f"g: {s}", fontsize=7, ha='center')
        else:
            if show_edge_labels:
                # show only interaction type
                edge_labels = {(a, b): d.get('itype', '') for a, b, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Plotly interactive plotting
    # -----------------------------
    def plot_plotly(self, show_tensors: bool = True, node_positions: Optional[Dict[str, np.ndarray]] = None,
                    width: int = 900, height: int = 700):
        """Interactive plot with plotly. Hovering nodes/edges shows tensor values.

        If plotly is not installed, raises ImportError.
        """
        if not self._built:
            self.build_graph()

        G = self.G
        if node_positions is None:
            pos = nx.spring_layout(G, seed=42)
            self.pos = pos
        else:
            pos = node_positions
            self.pos = pos

        node_x = []
        node_y = []
        node_text = []
        node_marker = dict(size=18)
        node_names = list(G.nodes())

        for n in node_names:
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            d = G.nodes[n]
            lines = [f"{n} ({d.get('type')})"]
            if d.get('type') == 'electron':
                lines.append("g-tensor:")
                lines.append(self._short_tensor_str(d.get('g_tensor'), maxchars=400))
            node_text.append('<br>'.join(lines))

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_names,
            hoverinfo='text',
            hovertext=node_text,
            marker=node_marker
        )

        edge_traces = []
        for a, b, d in G.edges(data=True):
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            # edge line
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                hoverinfo='text',
                line=dict(width=2),
                text=[self._edge_hover_text(a, b, d)] * 3
            ))

        layout = go.Layout(
            showlegend=False,
            hovermode='closest',
            width=width,
            height=height,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
        fig.show()
        return fig

    def _edge_hover_text(self, a: str, b: str, d: Dict[str, Any]):
        """Construct hover text for an edge.

        Includes interaction type and a short representation of the tensor(s).
        """
        itype = d.get('itype', '')
        lines = [f"{a} — {b} ({itype})"]
        t = d.get('tensor', None)
        s = self._short_tensor_str(t, maxchars=500)
        lines.append('tensor:')
        lines.append(s)
        st = d.get('strained_tensor', None)
        if st is not None:
            lines.append('strained_tensor:')
            lines.append(self._short_tensor_str(st, maxchars=500))
        return '<br>'.join(lines)

    # -----------------------------
    # Convenience: save static image
    # -----------------------------
    def save_matplotlib(self, filename: str, **kwargs):
        """Save a matplotlib static rendering to file. Accepts same kwargs as plot_matplotlib."""
        if plt is None:
            raise ImportError("matplotlib is required for save_matplotlib. Install it with `pip install matplotlib`.")
        # draw to the current figure and save
        self.plot_matplotlib(**kwargs)
        plt.savefig(filename, bbox_inches='tight')


# End of file
