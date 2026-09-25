"""chart_app.py — web app: upload a CSV, pick X / Y / cluster / filter,
get a clean bar or line chart. Built on chartbuilder.py.

Run:  .venv-grapher/bin/python chart_app.py  (port 8011, bind 0.0.0.0)
"""
from __future__ import annotations

import io
import uuid

import pandas as pd
from flask import (Flask, abort, flash, jsonify, redirect, render_template_string,
                   request, url_for)

from chartbuilder import Chart

app = Flask(__name__)
app.secret_key = "chart-builder-local-" + uuid.uuid4().hex

# uploaded datasets: token -> DataFrame (in memory; fine for a single-user tool)
_DATASETS: dict[str, pd.DataFrame] = {}


SAMPLE_CSV = """Year,Region,Product,Sales
2019,North,Widget,120
2019,North,Gadget,80
2019,South,Widget,150
2019,South,Gizmo,60
2020,North,Widget,140
2020,North,Gadget,95
2020,East,Widget,110
2020,East,Gizmo,45
2021,South,Widget,170
2021,South,Gadget,120
2021,West,Widget,90
2021,West,Gizmo,75
2022,North,Widget,210
2022,North,Gizmo,130
2022,East,Widget,180
2022,East,Gadget,100
2022,West,Widget,160
2022,West,Gadget,85
"""

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chart Builder</title>
<style>
  :root { --ink:#2d2d2d; --line:#e3e3e3; --accent:#4C72B0; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: var(--ink); margin: 0; background: #fafafa; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { color: #777; font-size: 13px; margin-bottom: 20px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 10px;
          padding: 18px; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px 14px; }
  label { display: block; font-size: 12px; font-weight: 600; color: #555;
           margin-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }
  select, input[type=text], input[type=file] {
    width: 100%; padding: 7px 9px; border: 1px solid #ccc; border-radius: 6px;
    font-size: 14px; background: #fff; }
  input[type=file] { padding: 4px; }
  .row { display: flex; gap: 18px; flex-wrap: wrap; }
  .row > div { flex: 1; min-width: 140px; }
  .radio { display: flex; gap: 16px; align-items: center; margin-top: 2px; }
  .radio label { text-transform: none; font-weight: 500; font-size: 14px;
                 display: flex; align-items: center; gap: 5px; margin: 0; }
  button { background: var(--accent); color: #fff; border: 0; border-radius: 6px;
            padding: 9px 22px; font-size: 15px; font-weight: 600; cursor: pointer; }
  button:hover { background: #3b61a0; }
  .btn-row { display: flex; gap: 10px; align-items: center; margin-top: 14px; flex-wrap: wrap; }
  .ghost { background: #fff; color: var(--accent); border: 1px solid var(--accent);
            padding: 8px 18px; border-radius: 6px; font-size: 14px; font-weight: 600; }
  .chart-box { text-align: center; }
  .chart-box img { max-width: 100%; height: auto; border: 1px solid var(--line);
                    border-radius: 6px; background: #fff; }
  .err { background: #fdecea; color: #b03030; border: 1px solid #f0c4c4;
          border-radius: 6px; padding: 8px 12px; font-size: 13px; margin-bottom: 14px; }
  .info { color: #777; font-size: 13px; }
  .dl { font-size: 13px; }
  .dl a { color: var(--accent); font-weight: 600; text-decoration: none; }
  .hint { font-size: 12px; color: #999; margin-top: 3px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Chart Builder</h1>
  <div class="sub">Pick an X, a Y, and optionally a cluster and a filter. Bar or line, done.</div>

  {% if errors %}
  <div class="err">{{ errors[0] }}</div>
  {% endif %}

  <form method="post" enctype="multipart/form-data" class="card">
    <input type="hidden" name="token" value="{{ token or '' }}">
    <div class="row">
      <div>
        <label for="file">CSV file</label>
        <input type="file" name="file" accept=".csv,text/csv">
        <div class="hint">or keep using the loaded data</div>
      </div>
      <div>
        <label for="sample">Sample data</label>
        <select name="sample">
          <option value="">— keep current —</option>
          <option value="sample" {{ 'selected' if sample is undefined }}>Load sample dataset</option>
        </select>
      </div>
    </div>
    <div class="grid" style="margin-top:14px">
      <div>
        <label for="x">X axis</label>
        <select name="x" required>
          {% for c in cols %}<option value="{{ c }}" {% if c == form.x %}selected{% endif %}>{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="y">Y axis</label>
        <select name="y" required>
          {% for c in cols %}<option value="{{ c }}" {% if c == form.y %}selected{% endif %}>{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="cluster">Cluster (optional)</label>
        <select name="cluster">
          <option value="">— none —</option>
          {% for c in cols %}<option value="{{ c }}" {% if c == form.cluster %}selected{% endif %}>{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="fcol">Filter column (optional)</label>
        <select name="fcol">
          <option value="">— none —</option>
          {% for c in cols %}<option value="{{ c }}" {% if c == form.fcol %}selected{% endif %}>{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="fval">Filter value(s)</label>
        <input type="text" name="fval" value="{{ form.fval }}" placeholder="one, or comma-separated">
      </div>
    </div>
    <div class="row" style="margin-top:14px">
      <div>
        <label>Chart</label>
        <div class="radio">
          <label><input type="radio" name="kind" value="bar" {{ 'checked' if form.kind == 'bar' }} /> Bar</label>
          <label><input type="radio" name="kind" value="line" {{ 'checked' if form.kind == 'line' }} /> Line</label>
        </div>
      </div>
      <div>
        <label for="stat">Y statistic</label>
        <select name="stat">
          <option value="auto" {{ 'selected' if form.stat == 'auto' }}>Auto</option>
          <option value="mean" {{ 'selected' if form.stat == 'mean' }}>Mean</option>
          <option value="median" {{ 'selected' if form.stat == 'median' }}>Median</option>
          <option value="sum" {{ 'selected' if form.stat == 'sum' }}>Sum</option>
          <option value="count" {{ 'selected' if form.stat == 'count' }}>Count</option>
        </select>
      </div>
      <div>
        <label for="title">Title (optional)</label>
        <input type="text" name="title" value="{{ form.title }}">
      </div>
    </div>
    <div class="btn-row">
      <button type="submit">Build chart</button>
      <a class="ghost" href="/sample">Load sample data</a>
    </div>
  </form>

  {% if chart_url %}
  <div class="card chart-box">
    <img src="{{ chart_url }}" alt="chart">
    <div class="dl"><a href="{{ chart_url }}">Download PNG</a></div>
  </div>
  {% endif %}

  <p class="info">{{ rows }} rows &times; {{ cols | length }} columns
    {% if token %}&middot; dataset <code>{{ token[:8] }}</code>{% endif %}</p>
</div>
</body>
</html>
"""


def _cols(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]


def _form() -> dict:
    f = {k: request.form.get(k, "") for k in
         ("x", "y", "cluster", "fcol", "fval", "kind", "stat", "title")}
    f["kind"] = f["kind"] or "bar"
    f["stat"] = f["stat"] or "auto"
    return f


def _render(df, form, token, chart_url=None, errors=None):
    cols = _cols(df)
    # sensible defaults on first view
    if not form["x"] and len(cols) >= 2:
        form["x"] = cols[0]
    if not form["y"] and len(cols) >= 2:
        form["y"] = cols[1]
    return render_template_string(PAGE, df=df, cols=cols, form=form, token=token,
                                   chart_url=chart_url, errors=errors, rows=len(df))


@app.route("/", methods=["GET", "POST"])
def index():
    token = request.form.get("token") or request.args.get("token")
    df = _DATASETS.get(token) if token else None
    form = _form()

    if request.method == "POST":
        chart_url = None
        # 1) new CSV upload?
        f = request.files.get("file")
        if f is not None and f.filename:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                flash(f"Could not read CSV: {e}")
                df = None
            if df is not None:
                if df.shape[1] < 2:
                    flash("CSV needs at least two columns.")
                else:
                    token = uuid.uuid4().hex
                    _DATASETS[token] = df
        if request.form.get("sample") == "sample" or (
                df is None and request.method == "POST" and not f):
            df = pd.read_csv(io.StringIO(SAMPLE_CSV))
            token = "sample"
            _DATASETS["sample"] = df
        if df is None and token is None:
            flash("Upload a CSV to get started.")
            df = pd.DataFrame({"A": [1], "B": [2]})

        # 2) validate spec
        errors = None
        if df is not None:
            if form["x"] and form["y"] and form["x"] != form["y"]:
                errors, chart_url = _build(df, form)
            elif form["x"] and form["y"]:
                errors = ["X and Y must be different columns."]
            else:
                errors = ["Pick both an X and a Y column."]
        return _render(df, form, token, chart_url, errors)

    # GET
    if df is None:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV))
        token = "sample"
    return _render(df, form, token)


def _build(df: pd.DataFrame, form: dict) -> tuple[list | None, str | None]:
    try:
        c = Chart(df).x(form["x"]).y(form["y"])
        if form["cluster"]:
            c = c.cluster(form["cluster"])
        if form["fcol"]:
            vals = [v.strip() for v in form["fval"].split(",") if v.strip()]
            if not vals:
                return ["Filter column set but no filter value given."], None
            col = df[form["fcol"]]
            if len(vals) == 1:
                # coerce the scalar to the column's dtype so "2019" matches 2019
                try:
                    if pd.api.types.is_numeric_dtype(col):
                        v = float(vals[0])
                        v = int(v) if v.is_integer() else v
                    else:
                        v = vals[0]
                    c = c.filter(form["fcol"], v)
                except ValueError:
                    c = c.filter(form["fcol"], vals[0])
            else:
                c = c.filter(form["fcol"], vals)
        c = c.plot(kind=form["kind"],
                   stat=None if form["stat"] == "auto" else form["stat"],
                   title=form["title"] or None)
        buf = io.BytesIO()
        c._fig.savefig(buf, dpi=150, facecolor="white")
        buf.seek(0)
        app.chart_png = buf.read()
        return None, url_for("chart_png")
    except Exception as e:
        return [f"{type(e).__name__}: {e}"], None


@app.route("/chart.png")
def chart_png():
    if not hasattr(app, "chart_png"):
        abort(404)
    return app.chart_png, 200, {"Content-Type": "image/png",
                                "Content-Disposition": "attachment; filename=chart.png"}


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8011)))
