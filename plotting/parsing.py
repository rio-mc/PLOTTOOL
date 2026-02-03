from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable
from datetime import datetime, date, time
import matplotlib.dates as mdates
import numpy as np
import csv
from io import StringIO

@dataclass(frozen=True)
class ParseIssue:
    message: str
    series_index: Optional[int] = None
    axis: Optional[str] = None


# Matches floats like: 1, -2, 3.14, .5, 1., 1e-3, -2.3E+5
_FLOAT_TOKEN = re.compile(
    r"""
    [-+]?
    (?:
        (?:\d+(?:\.\d*)?)   # 1 or 1. or 1.23
        |
        (?:\.\d+)           # .23
    )
    (?:[eE][-+]?\d+)?       # optional exponent
    """,
    re.VERBOSE,
)


def parse_inline_vector(text: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Parse a 1D vector of floats from arbitrary pasted text.
    Accepts commas/spaces/newlines/brackets/mixed separators.
    Returns (array, error_message_or_None).
    """
    raw = (text or "").strip()
    if not raw:
        return np.asarray([], dtype=float), None

    vals: List[float] = []
    for m in _FLOAT_TOKEN.finditer(raw):
        token = m.group(0)
        try:
            vals.append(float(token))
        except Exception:
            # Should be rare due to regex, but safe anyway.
            continue

    if not vals:
        return np.asarray([], dtype=float), "No numeric values found."

    return np.asarray(vals, dtype=float), None


def validate_xy_lengths(x: np.ndarray, y: np.ndarray) -> Optional[str]:
    """
    Validate that x and y exist and have identical lengths.
    Returns error string or None.
    """
    if x.size == 0 or y.size == 0:
        return "Both x and y must contain at least one value."
    if x.size != y.size:
        return f"x and y must have the same length (got {x.size} and {y.size})."
    return None


def parse_table_with_header(text: str) -> Tuple[List[str], List[List[str]], Optional[str]]:
    """
    Parse a pasted table with a header row.

    Supports:
      - CSV (comma)
      - TSV (tab)
      - semicolon separated
      - whitespace separated

    Returns: (header, rows, err)
    """
    raw = (text or "").strip()
    if not raw:
        return [], [], "Table text is empty."

    # Keep non-empty lines only
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], [], "Table must have at least a header row and one data row."

    sample = "\n".join(lines[:10])

    # Try to detect delimiter (simple + robust)
    candidate_delims = [",", "\t", ";"]
    delim = None
    best = 0
    for d in candidate_delims:
        c = sample.count(d)
        if c > best:
            best = c
            delim = d

    if delim is not None and best > 0:
        # Use csv.reader for comma/tab/semicolon
        try:
            reader = csv.reader(StringIO("\n".join(lines)), delimiter=delim)
            table = [[cell.strip() for cell in row] for row in reader if row and any(c.strip() for c in row)]
        except Exception as e:
            return [], [], f"Could not parse table ({type(e).__name__}: {e})"
    else:
        # Fallback: whitespace-separated
        table = [ln.strip().split() for ln in lines]

    if not table or len(table) < 2:
        return [], [], "Table must contain a header and at least one data row."

    header = [h.strip() for h in table[0]]
    if not header or all(h == "" for h in header):
        return [], [], "Header row is empty."

    rows = table[1:]
    # Normalize row lengths (pad shorter rows with "")
    ncol = len(header)
    norm_rows: List[List[str]] = []
    for r in rows:
        r2 = list(r[:ncol]) + ([""] * max(0, ncol - len(r)))
        norm_rows.append([c.strip() for c in r2])

    return header, norm_rows, None


def table_to_columns(header: List[str], rows: List[List[str]]) -> Dict[str, List[str]]:
    """
    Convert (header, rows) to dict of columns: name -> list of cell strings
    """
    cols: Dict[str, List[str]] = {h: [] for h in header}
    for r in rows:
        for i, h in enumerate(header):
            cols[h].append(r[i] if i < len(r) else "")
    return cols


def col_as_float(col: List[str], col_name: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Convert a column of strings to float numpy array.
    Empty cells are skipped.
    Errors on first non-numeric non-empty cell.
    """
    out: List[float] = []
    for i, cell in enumerate(col):
        s = "" if cell is None else str(cell).strip()
        if s == "":
            continue
        try:
            out.append(float(s))
        except Exception:
            return np.asarray([], dtype=float), f"Non-numeric value in column '{col_name}' at row {i + 2}: {s!r}"
            # +2 because row 1 is header, and we want 1-based human row numbers

    if not out:
        return np.asarray([], dtype=float), f"No numeric values found in column '{col_name}'."

    return np.asarray(out, dtype=float), None


_DT_SPLIT = re.compile(r"[,\t;\n\r]+|\s{2,}")  # commas, tabs, semicolons, newlines, or big whitespace

def _tokenise_datetime_text(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    # Keep single-space separated tokens intact when user pastes ISO dates
    # We split primarily on commas/newlines/tabs/semicolons. If the user pastes "2025-01-01 2025-01-02"
    # it will also work because of the secondary split on 2+ spaces.
    parts = []
    for chunk in _DT_SPLIT.split(raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        # If chunk still contains single-space separated items, split them too.
        # This helps "2025-01-01 2025-01-02" but preserves "2025-01-01T12:30:00".
        if " " in chunk and "T" not in chunk:
            parts.extend([p for p in chunk.split() if p.strip()])
        else:
            parts.append(chunk)
    return parts

def _parse_one_datetime(tok: str) -> datetime:
    """
    Parse one datetime token. Supports:
      - ISO date: 2025-01-31
      - ISO datetime: 2025-01-31T13:45:00 or 2025-01-31 13:45:00
      - date only -> midnight
    """
    t = tok.strip()
    # Accept "YYYY-MM-DD HH:MM:SS" by replacing space with 'T' for fromisoformat
    if " " in t and "T" not in t:
        t_iso = t.replace(" ", "T")
    else:
        t_iso = t
    try:
        dt = datetime.fromisoformat(t_iso)
        return dt
    except Exception:
        pass

    # Last-resort: numpy datetime64 can parse many ISO-like forms, then convert
    try:
        import numpy as _np
        v = _np.datetime64(t)
        # Convert to python datetime via timestamp in ns
        # This conversion is a bit fiddly; easiest is to go through string
        return datetime.fromisoformat(str(v).replace("Z", ""))
    except Exception:
        raise ValueError(
            f"Could not parse datetime token {tok!r}. Use ISO like 2025-01-31 or 2025-01-31T13:45:00."
        )

def parse_inline_datetime_vector(text: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Parse datetimes from text and convert to Matplotlib date numbers (float days).
    Returns (array, err).
    """
    toks = _tokenise_datetime_text(text)
    if not toks:
        return np.asarray([], dtype=float), "No datetime values found."

    dts: List[datetime] = []
    for tok in toks:
        try:
            dts.append(_parse_one_datetime(tok))
        except Exception as e:
            return np.asarray([], dtype=float), str(e)

    return np.asarray(mdates.date2num(dts), dtype=float), None

def col_as_datetime(col: List[str], col_name: str) -> Tuple[np.ndarray, Optional[str]]:
    """
    Convert a column of strings to Matplotlib date numbers.
    Empty cells are skipped.
    Errors on first non-empty unparsable cell.
    """
    import matplotlib.dates as mdates

    out: List[float] = []
    for i, cell in enumerate(col):
        s = "" if cell is None else str(cell).strip()
        if s == "":
            continue
        try:
            dt = _parse_one_datetime(s)
            out.append(float(mdates.date2num(dt)))
        except Exception as e:
            # +2: header is row 1, data starts at row 2
            return np.asarray([], dtype=float), f"Bad datetime in column '{col_name}' at row {i + 2}: {s!r}. {e}"

    if not out:
        return np.asarray([], dtype=float), f"No datetime values found in column '{col_name}'."

    return np.asarray(out, dtype=float), None
