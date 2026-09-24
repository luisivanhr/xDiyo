"""Optional throttled notebook loss visualization, with no server or widget."""

from html import escape
import time

import numpy as np


class LiveLossPlot:
    """TrainingEvent observer that updates one lightweight SVG display in IPython.

    Rendering is throttled by interval seconds; final events always refresh.
    Only selected metric names are drawn. max_points limits display points per
    line, not the runner's saved history. Use a fresh instance per desired panel.
    No server, TensorBoard, JavaScript or widget extension is required. Frontends
    must support IPython display updates; ordinary runs need no observer.
    """
    def __init__(self, *, interval=1.0, metrics=("train_loss", "validation_loss"), max_points=500):
        if isinstance(interval, bool) or not np.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be positive and finite.")
        if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 2:
            raise ValueError("max_points must be an integer >= 2.")
        self.interval, self.metrics, self.max_points = interval, ((metrics,) if isinstance(metrics, str) else tuple(metrics)), max_points
        self.series, self.handle, self.last_draw = {}, None, -np.inf

    def __call__(self, event):
        if not event.final:
            for name, value in event.metrics.items():
                if name in self.metrics:
                    self.series.setdefault((event.fold_id, event.attempt, name), []).append((event.step, value))
        now = time.monotonic()
        if event.final or now-self.last_draw >= self.interval:
            from IPython.display import HTML, display
            content = HTML(self.to_html())
            if self.handle is None:
                self.handle = display(content, display_id=True)
            else:
                self.handle.update(content)
            self.last_draw = now

    def to_html(self):
        """Static SVG snapshot, also useful outside a running notebook display."""
        all_values = [(x, y) for values in self.series.values() for x, y in values if np.isfinite(y)]
        if not all_values:
            return '<div role="status">Waiting for finite training losses…</div>'
        xmax = max(x for x, _ in all_values)
        ymin, ymax = min(y for _, y in all_values), max(y for _, y in all_values)
        span = ymax-ymin or max(abs(ymax)*0.1, 1.0)
        ymin, ymax = ymin-span*.05, ymax+span*.05
        colors = ("#58c7b2", "#ed7975", "#b9a3ff", "#e8bb65", "#72b8ee", "#dddddd")
        lines, legends = [], []
        for i, (key, values) in enumerate(self.series.items()):
            indices = np.arange(len(values)) if len(values) <= self.max_points else np.unique(np.linspace(0, len(values)-1, self.max_points).astype(int))
            segments, points = [], []
            previous = -1
            for index in indices:
                x, y = values[index]
                # Preserve missing-data breaks even when display downsampling
                # omitted the nonfinite step between two displayed points.
                if any(not np.isfinite(value) for _, value in values[previous+1:index+1]):
                    if points:
                        segments.append(points)
                    points = []
                previous = index
                if not np.isfinite(y):
                    continue
                points.append(f"{60+600*(x-1)/max(xmax-1,1):.2f},{250-215*(y-ymin)/(ymax-ymin):.2f}")
            if points:
                segments.append(points)
            color = colors[i % len(colors)]
            for points in segments:
                lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
                if len(points) == 1:
                    x, y = points[0].split(",")
                    lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
            fold, attempt, name = key
            legends.append(f'<span style="color:{color};margin-right:18px">{escape(str(name))} · fold {fold} · attempt {attempt}</span>')
        return (f'<div style="background:#17202b;color:#e8eef7;padding:12px;border-radius:8px">'
                '<svg viewBox="0 0 700 290" role="img" aria-label="Training and validation loss by step" style="width:100%;max-width:850px">'
                '<path d="M60 25V250H675" fill="none" stroke="#9eb0c5"/>'
                f'<text x="4" y="40" fill="#e8eef7" font-size="12">{ymax:.3g}</text>'
                f'<text x="4" y="250" fill="#e8eef7" font-size="12">{ymin:.3g}</text>'
                '<text x="60" y="270" fill="#e8eef7" font-size="12">1</text>'
                f'<text x="640" y="270" fill="#e8eef7" font-size="12">{xmax}</text>'
                '<text x="330" y="285" fill="#e8eef7" font-size="12">Training step</text>'
                + ''.join(lines) + '</svg><div>' + ''.join(legends) + '</div></div>')
