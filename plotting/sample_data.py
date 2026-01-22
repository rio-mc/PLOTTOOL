from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

@dataclass(frozen=True)
class SeriesData:
    x: np.ndarray
    y: np.ndarray
    label: str
    z: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None


def make_sample_series(
    n_series: int,
    n_points: int = 60,
    seed: int = 7,
    is_3d: bool = False,
) -> List[SeriesData]:
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 10.0, n_points)

    out: List[SeriesData] = []
    for i in range(n_series):
        phase = i * 0.6
        noise = rng.normal(0.0, 0.25, size=n_points)

        ys = np.sin(xs + phase) + 0.15 * xs + noise

        if is_3d:
            # Simple but visually nice 3D structure
            zs = np.cos(xs + phase) + 0.1 * xs + rng.normal(0.0, 0.15, size=n_points)
        else:
            zs = None

        out.append(
            SeriesData(
                x=xs,
                y=ys,
                z=zs,
                label=f"Series {i + 1}",
                x_std=None,
                y_std=None,
            )
        )

    return out
