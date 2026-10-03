"""chartbuilder.py — SPSS-style simple chart building.

Pick an X variable, a Y variable, and optionally a cluster variable and a
filter. Get a clean bar or line chart. Handles nominal, ordinal, and
continuous variables:

  * nominal / ordinal X  -> one bar/point per category
  * continuous X         -> grouped by value (<=12 distinct) or auto-binned
  * date / datetime X    -> binned by calendar (day/week/month/quarter/year)
                           on a real date axis, never coerced to numbers
  * continuous Y         -> aggregated (mean / sum / median)
  * categorical Y        -> counted (n per category)

Type is auto-detected from the column; override per call if needed:
    chart.x("Year", "ordinal")

Usage
-----
    from chartbuilder import Chart

    Chart(df).x("Region").y("Sales").cluster("Quarter").plot(kind="bar")
    Chart(df).x("Year").y("Sales").cluster("Region").plot(kind="line")
    Chart(df).x("Region").y("Sales").cluster("Quarter").filter("Year", 2025) \
        .plot(kind="bar").save("chart.png")
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
from matplotlib import dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Tableau palette — clean, colour-blind-friendly, good print contrast.
PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.8,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": "#333333",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelcolor": "#333333",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_STAT_LABEL = {
    "mean": "Mean {y}",
    "median": "Median {y}",
    "sum": "Total {y}",
    "count": "Count",
}


def _infer_scale(series: pd.Series) -> str:
    """Guess the measurement scale of a column."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if not pd.api.types.is_numeric_dtype(series):
        return "nominal"
    s = series.dropna()
    u = s.nunique()
    # Repeated numeric values are categories (counted); all-distinct
    # values are data points (plotted as-is, never collapsed to counts).
    if u < len(s) and u <= 8:
        return "ordinal"
    return "continuous"


def _date_freq(xs: pd.Series) -> str:
    """Pick a calendar bin size for a datetime column from its span.

    Sparse data (<=12 distinct timestamps) -> one point per exact
    timestamp at its real position; denser data gets calendar bins sized
    by span: <=30 min -> 1 min; <=2 d -> 1 h; <=5 d -> 4 h; <=12 d ->
    1 d; <=120 d -> 1 week; <=2 y -> 1 month; <=6 y -> 1 quarter.
    """
    if xs.empty:
        raise ValueError("No valid dates in the X column")
    # Sparse data -> one point per exact timestamp at its real position
    # (no calendar collapsing); denser data -> a calendar bin sized by span.
    if xs.nunique() <= 12:
        return "ts"
    span = xs.max() - xs.min()
    span_s = span.total_seconds()
    span_d = span_s / 86400
    if span_s <= 1800:
        return "min"
    if span_d <= 2:
        return "h"
    if span_d <= 5:
        return "4h"
    if span_d <= 12:
        return "D"
    if span_d <= 120:
        return "W-SUN"
    if span_d <= 730:
        return "M"
    if span_d <= 2190:
        return "Q-DEC"
    return "Y"


# Bar widths in *days* (matplotlib date-axis units) per calendar bin size.
_DATE_BAR_WIDTH = {
    "min": 1 / 1440,  # one minute, as a fraction of a day
    "h": 0.9 / 24,
    "4h": 3.5 / 24,
    "D": 0.8,
    "W-SUN": 5.6,
    "M": 24.4,
    "Q-DEC": 73.0,
    "Y": 292.0,
}


def _date_label(freq: str, p) -> str:
    """One tick label for a calendar period."""
    if freq in ("min", "h"):
        return p.strftime("%H:%M")
    if freq == "4h":
        return p.strftime("%b %d %H:%M")
    if freq in ("D", "W-SUN"):
        return p.strftime("%b %d")
    if freq == "M":
        return p.strftime("%b %Y")
    if freq == "Q-DEC":
        return str(p)
    return str(p.year)


def _stat(values: pd.Series, stat: str) -> float:
    values = pd.Series(values).dropna()
    if len(values) == 0:
        return float("nan")  # no data in this bin -> visible gap on the axis
    if stat == "count":
        return float(len(values))
    if stat == "mean":
        return float(values.mean()) if len(values) else np.nan
    if stat == "median":
        return float(values.median()) if len(values) else np.nan
    if stat == "sum":
        return float(values.sum()) if len(values) else np.nan
    raise ValueError(f"Unknown stat {stat!r} (use mean/median/sum/count)")


class Chart:
    """SPSS-style chart builder: x, y, optional cluster, optional filter."""

    def __init__(self, df: pd.DataFrame, title: str | None = None):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df
        self._title = title
        self._x = self._y = self._cluster = None
        self._xscale = self._yscale = self._cscale = None
        self._datefreq: str | None = None
        self._ts_width: float = 1.0 / 24  # bar width (days) for "ts" bins
        self._filters: list[tuple[str, object]] = []
        self._stat: str | None = None
        self._fig: plt.Figure | None = None
        self._ax = None
        self._ylabels: list[str] | None = None
        self._stat_used = "mean"

    # ------------------------------------------------------------------ #
    # spec                                                               #
    # ------------------------------------------------------------------ #
    def x(self, col: str, scale: str | None = None) -> "Chart":
        self._x = col
        self._is_date = pd.api.types.is_datetime64_any_dtype(self.df[col])
        if scale:
            self._xscale = scale
        elif self._is_date:
            self._xscale = "date"
        else:
            self._xscale = _infer_scale(self.df[col])
        return self

    def y(self, col: str, scale: str | None = None) -> "Chart":
        self._y = col
        self._yscale = scale or _infer_scale(self.df[col])
        return self

    def cluster(self, col: str, scale: str | None = None) -> "Chart":
        self._cluster = col
        self._cscale = scale or _infer_scale(self.df[col])
        return self

    def filter(self, col: str, values) -> "Chart":
        """Restrict rows: scalar -> ==, list/tuple/set -> isin."""
        self._filters.append((col, values))
        return self

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #
    def _filtered(self) -> pd.DataFrame:
        df = self.df
        for col, values in self._filters:
            if isinstance(values, (list, tuple, set)):
                df = df[df[col].isin(list(values))]
            else:
                df = df[df[col] == values]
        return df

    def _effective_stat(self) -> str:
        """Categorical Y is counted; continuous Y uses the requested stat."""
        if self._yscale in ("nominal", "ordinal"):
            return "count"
        return self._stat if self._stat in ("mean", "median", "sum", "count") else "mean"

    def _aggregate(self, sub: pd.DataFrame) -> tuple[list[str], list[float], list[float]]:
        """Return (labels, y values, x positions) for one series."""
        xcol, ycol, stat = self._x, self._y, self._stat_used
        xs, ys = sub[xcol], sub[ycol]

        if self._xscale == "date":
            return self._aggregate_date(sub, xcol, ycol, stat)

        if self._xscale == "continuous":
            if xs.nunique(dropna=True) <= 12:
                order = np.sort(xs.dropna().unique())
                labels = [f"{v:g}" if pd.api.types.is_numeric_dtype(xs) else str(v)
                          for v in order]
                vals = [_stat(sub.loc[sub[xcol] == v, ycol], stat) for v in order]
                return labels, vals, [float(v) for v in order]
            # too many distinct values -> equal-width bins
            binned = pd.cut(xs, 8)
            sub2 = sub.copy()
            sub2["__bin__"] = binned
            cats = binned.cat.categories
            labels = [f"{iv.left:g}–{iv.right:g}" for iv in cats]
            vals, pos = [], []
            for iv in cats:
                g = sub2[sub2["__bin__"] == iv][ycol]
                vals.append(_stat(g, stat))
                pos.append(float(iv.mid))
            return labels, vals, pos

        order = list(xs.dropna().unique())  # appearance order for categories
        labels = [str(v) for v in order]
        vals = [_stat(sub.loc[sub[xcol] == v, ycol], stat) for v in order]
        return labels, vals, list(range(len(order)))

    def _aggregate_date(self, sub, xcol, ycol, stat):
        """Calendar-aware binning for a date/datetime X column.

        Rows are grouped by calendar period (day/week/month/quarter/year);
        positions are real dates (matplotlib date units), never raw numbers.
        Empty periods stay in the axis so gaps are visible.
        """
        xs = sub[xcol]
        ys = sub[ycol]
        if not pd.api.types.is_datetime64_any_dtype(xs):
            raise ValueError(f"Column {xcol!r} is not a date column — "
                             "cannot use a date axis for it")
        span = xs.dropna()
        if span.empty:
            raise ValueError(f"No valid dates in the {xcol!r} column")
        freq = self._datefreq or _date_freq(span)
        self._datefreq = freq

        if freq == "ts":
            # One point per exact timestamp (sparse intraday data) — real
            # time-of-day positions, no calendar collapsing.
            g = sub[~xs.isna()].sort_values(xcol)
            xs_g = g[xcol].reset_index(drop=True)
            ys_g = g[ycol].reset_index(drop=True)

            def _num(t):
                ts = pd.Timestamp(t)
                if getattr(ts, "tzinfo", None) is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                return mdates.date2num(ts.to_pydatetime())

            centers, labels, vals = [], [], []
            for t in xs_g.unique():
                start = pd.Timestamp(t)
                if getattr(start, "tzinfo", None) is not None:
                    start = start.tz_convert("UTC").tz_localize(None)
                centers.append(_num(t))
                labels.append(start.strftime("%b %d %H:%M"))
                tvals = ys_g[xs_g == t]
                vals.append(_stat(tvals, stat))
            # Bar width: 75% of the smallest gap between distinct timestamps
            # (days), floored so bars never vanish on a wide axis.
            uniq = xs_g.unique()
            if len(uniq) > 1:
                min_gap = float(np.diff([_num(t) for t in uniq]).min())
            else:
                min_gap = 1.0 / 24  # default: one hour
            self._ts_width = max(min_gap * 0.75, 1e-4)
            return labels, vals, centers

        periods = span.dt.to_period(freq)
        lo = pd.period_range(start=periods.min(), end=periods.max(), freq=freq)
        sub2 = sub[~periods.isna()].copy()
        sub2["__p__"] = sub2[xcol].dt.to_period(freq)

        centers, labels, vals = [], [], []
        for p in lo:
            # tz-naive UTC so date2num gets a real datetime, not tz-aware
            start = pd.Timestamp(p.start_time)
            if getattr(start, "tzinfo", None) is not None:
                start = start.tz_convert("UTC").tz_localize(None)
            centers.append(mdates.date2num(start.to_pydatetime())
                           + (p.end_time - p.start_time).total_seconds()
                           / 86400 / 2)
            labels.append(_date_label(freq, p))
            g = ys[sub2["__p__"] == p]
            vals.append(_stat(g, stat))
        return labels, vals, centers

    def _series(self) -> list[tuple[str, list[str], list[float], list[float]]]:
        """[(name, labels, values, positions), ...]"""
        sub = self._filtered()
        if sub.empty:
            raise ValueError("No rows left after filters — check filter values.")
        out: list[tuple[str, list[str], list[float], list[float]]] = []
        if self._cluster is None:
            labels, vals, pos = self._aggregate(sub)
            out.append((self._x, labels, vals, pos))
        else:
            ccol = self._cluster
            order = list(sub[ccol].dropna().unique())
            for c in order:
                cs = sub[sub[ccol] == c]
                labels, vals, pos = self._aggregate(cs)
                out.append((str(c), labels, vals, pos))
        return out

    # ------------------------------------------------------------------ #
    # render                                                             #
    # ------------------------------------------------------------------ #
    def plot(self, kind: str = "bar", title: str | None = None,
             stat: str | None = None, figsize: tuple[float, float] = (9, 5.5),
             dpi: int = 150) -> "Chart":
        if kind not in ("bar", "line"):
            raise ValueError("kind must be 'bar' or 'line'")
        if self._x is None or self._y is None:
            raise ValueError("Call .x(...) and .y(...) first")
        if self._x == self._y:
            raise ValueError("x and y must be different columns")
        if self._cluster is not None and self._cluster in (self._x, self._y):
            raise ValueError("cluster must be a different column from x and y")
        if kind == "bar" and self._xscale in ("continuous", "date") and self._cluster is not None:
            raise ValueError("Clustered bar charts need a nominal/ordinal x — "
                             "use a line chart or drop cluster")
        if stat:
            self._stat = stat
        self._stat_used = self._effective_stat()

        series = self._series()
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        n = len(series[0][1])
        labels = series[0][1]

        if kind == "bar":
            self._draw_bar(ax, series, n, labels)
        else:
            self._draw_line(ax, series, n, labels)

        # axis cosmetics
        ax.grid(axis="y")
        ax.set_xlabel(self._x)
        ylab = _STAT_LABEL[self._stat_used].format(y=self._y)
        ax.set_ylabel(ylab)
        if self._xscale == "date":
            pos0 = series[0][3]
            step = max(1, -(-n // 9))  # at most ~9 tick labels
            ax.set_xticks(pos0[::step], labels[::step], rotation=35, fontsize=8)
            width_days = (self._ts_width if self._datefreq == "ts"
                         else _DATE_BAR_WIDTH[self._datefreq])
            lo = min(min(s[3]) for s in series)
            hi = max(max(s[3]) for s in series)
            ax.set_xlim(lo - width_days / 2 - 0.1, hi + width_days / 2 + 0.1)
        elif self._xscale == "continuous":
            pos0 = series[0][3]
            long = len(labels) > 8 or max(len(str(l)) for l in labels) > 6
            if long and self._xscale == "continuous" and len(set(pos0)) > 1:
                # fewer tick labels, rotated
                step = max(1, len(pos0) // 8)
                ax.set_xticks(pos0[::step], labels[::step], rotation=35, fontsize=8)
            else:
                ax.set_xticks(pos0, labels)
                ax.tick_params(axis="x", labelsize=9)
        else:
            ax.set_xticks(range(n), labels)
            if len(labels) > 8 or max(len(str(l)) for l in labels) > 6:
                ax.tick_params(axis="x", rotation=35, labelsize=8)
            else:
                ax.tick_params(axis="x", labelsize=9)

        if self._xscale == "continuous":
            lo = min(min(s[3]) for s in series)
            hi = max(max(s[3]) for s in series)
            ax.set_xlim(lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08)
        elif self._xscale != "date":
            ax.set_xlim(-0.6, n - 0.4)
        ax.margins(y=0.06)

        if len(series) > 1:
            ax.legend(title="Cluster" if self._cluster is None else None,
                      labels=[s[0] for s in series], loc="upper left")
        t = title or self._title or self._default_title()
        ax.set_title(t)
        fig.tight_layout()
        self._fig, self._ax = fig, ax
        self._ylabels = labels
        return self

    def _draw_bar(self, ax, series, n, labels):
        if self._xscale == "date":
            width = (self._ts_width if self._datefreq == "ts"
                     else _DATE_BAR_WIDTH[self._datefreq])
            for i, (name, _l, vals, pos) in enumerate(series):
                ax.bar(pos, vals, width=width,
                       color=PALETTE[i % len(PALETTE)],
                       edgecolor="white", linewidth=0.8,
                       label=name, zorder=3)
            return
        width = 0.75 / len(series)
        for i, (name, _l, vals, pos) in enumerate(series):
            xs = [p - (len(series) - 1) * width / 2 + i * width if self._xscale != "continuous"
                  else p for p in pos]
            ax.bar(xs, vals, width=width if self._xscale != "continuous" else width,
                   color=PALETTE[i % len(PALETTE)],
                   edgecolor="white", linewidth=0.8,
                   label=name, zorder=3)

    def _draw_line(self, ax, series, n, labels):
        for i, (name, _l, vals, pos) in enumerate(series):
            ax.plot(pos, vals, marker="o", markersize=4, linewidth=1.8,
                    color=PALETTE[i % len(PALETTE)], label=name, zorder=3)

    def _default_title(self) -> str:
        t = f"{self._y} by {self._x}"
        if self._cluster is not None:
            t += f" — {self._cluster}"
        return t

    # ------------------------------------------------------------------ #
    # output                                                             #
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> "Chart":
        if self._fig is None:
            raise ValueError("Call .plot() before .save()")
        self._fig.savefig(path, dpi=self._fig.dpi, facecolor="white")
        return self

    def show(self):
        if self._fig is None:
            raise ValueError("Call .plot() before .show()")
        plt.show()
        return self

    def __repr__(self):
        parts = [f"Chart(x={self._x!r}, y={self._y!r}"]
        if self._cluster is not None:
            parts.append(f", cluster={self._cluster!r}")
        if self._filters:
            parts.append(f", filter={self._filters}")
        parts.append(")")
        return "".join(parts)
