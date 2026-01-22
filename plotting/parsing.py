from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ParseIssue:
    message: str
    series_index: Optional[int] = None
    axis: Optional[str] = None


_FLOAT_TOKEN = re.compile(r"""
    [-+]?(
        (\d+(\.\d*)?)|(\.\d+)
    )([eE][-+]?\d+)?   # exponent
""", re.VERBOSE)


def parse_inline_vector(text: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Parse a vector of floats from arbitrary pasted text.
    Accepts commas, spaces, newlines, brackets, and mixed separators.
    """
    if not text or not text.strip():
        return np.array([], dtype=float), None

    tokens = _FLOAT_TOKEN.findall(text)
    if not tokens:
        return np.array([], dtype=float), "No numeric values found."

    # Each match returns groups, but the full match is at index 0 of the tuple item via findall?
    # Our regex uses groups, so findall returns tuples. We can instead use finditer for full match.
    vals: List[float] = []
    for m in _FLOAT_TOKEN.finditer(text):
        try:
            vals.append(float(m.group(0)))
        except ValueError:
            continue

    if not vals:
        return np.array([], dtype=float), "Could not parse numeric values."

    return np.asarray(vals, dtype=float), None


def validate_xy_lengths(x: np.ndarray, y: np.ndarray) -> Optional[str]:
    if x.size == 0 or y.size == 0:
        return "Both x and y must contain at least one value."
    if x.size != y.size:
        return f"x and y must have the same length (got {x.size} and {y.size})."
    return None
