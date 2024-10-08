from pathlib import Path
import abc
from typing import List, Dict, Tuple, Any

from matplotlib import pyplot as plt
import numpy as np


class Plotter(abc.ABC):
    def __init__(self, output_path: Path = None, show: bool = True):
        self.output_path = Path(output_path) if output_path is not None else None
        self.show = show

    def _save_and_and_show(self, fig: plt.Figure):
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.output_path)

        if self.show:
            plt.show()

    def _get_grid_axes(
        self,
        n: int,
        n_cols: int,
        width_per_ax: int = 5,
        height_per_ax: int = 5,
        squeeze: bool = True,
        **subplot_kwargs: Dict,
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        n_rows = n // n_cols
        n_rows += 1 if n % n_cols else 0

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * width_per_ax, n_rows * height_per_ax),
            **subplot_kwargs,
        )

        if n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        else:
            axes = axes.tolist()

        if squeeze:
            axes = np.array(axes).flatten()

        return fig, axes

    @abc.abstractmethod
    def __call__(self, **plotting_data) -> Tuple[plt.Figure, plt.Axes]:
        raise NotImplementedError


class TimeBetweenSamplePlotter(Plotter):
    def __call__(
        self, recording_data: Dict[str, Dict[str, Tuple[List[float], List[Any]]]]
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()

        for obj, topic_data in recording_data.items():
            for topic, (timestamps, _) in topic_data.items():
                diffs = np.diff(timestamps)
                diffs = np.append(
                    diffs, 0
                )  # add a 0 at the end to make the lengths match
                ax.plot(timestamps, diffs, label=f"{obj}.{topic}")

        ax.legend()
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Time between samples")

        self._save_and_and_show(fig)

        return fig, ax
