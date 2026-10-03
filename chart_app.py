"""chart_app.py — web app: upload a CSV, pick X / Y / cluster / filter,
get a clean bar or line chart. Built on chartbuilder.py.

Flow:
  * Landing page shows the sample dataset with a chart already built —
    value is visible before the user does anything.
  * "Upload CSV" (or "Load sample data") loads a dataset; a sensible
    default chart is built immediately.
  * "Example charts" chips are one-click presets derived from the data.
  * Fine-tune X / Y / cluster / filter / stat, then "Build chart".
  * Any build can be shared: the full spec is encoded in the URL.

Run:  python chart_app.py  (port $PORT or 8011, bind 0.0.0.0)
Prod: gunicorn -k gthread --threads 4 chart_app:app  (see Procfile)
"""
from __future__ import annotations
import base64
import io
import uuid
from urllib.parse import urlencode

import pandas as pd
from flask import (Flask, jsonify, render_template_string,
                   request)

from chartbuilder import Chart, PALETTE

app = Flask(__name__)
app.secret_key = "chart-builder-local-" + uuid.uuid4().hex
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB CSV cap

_MAX_DATASETS = 10
_MAX_COLS = 200
_MAX_ROWS = 1_000_000

# uploaded datasets: token -> DataFrame (in memory; fine for a single-user tool)
_DATASETS: dict[str, pd.DataFrame] = {}
_META: dict[str, dict] = {}  # token -> {"name": ...}

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


def _remember(token: str, df: pd.DataFrame, name: str) -> None:
    _DATASETS[token] = df
    _META[token] = {"name": name}
    while len(_DATASETS) > _MAX_DATASETS:
        oldest = next(iter(_DATASETS))
        _DATASETS.pop(oldest, None)
        _META.pop(oldest, None)


def _sample() -> pd.DataFrame:
    if "sample" not in _DATASETS:
        _remember("sample", pd.read_csv(io.StringIO(SAMPLE_CSV)), "Sample data")
    return _DATASETS["sample"]


def _meta_name(token: str | None) -> str:
    if token and token in _META:
        return _META[token]["name"]
    return ""


def _kind(df: pd.DataFrame, col: str) -> str:
    s = df[col]
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        return "number"
    return "text"


def _kinds(df: pd.DataFrame) -> dict[str, str]:
    return {str(c): _kind(df, c) for c in df.columns}


# --------------------------------------------------------------------- #
# CSV intake                                                             #
# --------------------------------------------------------------------- #

def _read_csv_smart(raw: bytes) -> tuple[pd.DataFrame, str]:
    """Try comma, semicolon, tab — report which delimiter worked."""
    for label, delim in (("comma", ","), ("semicolon", ";"), ("tab", "\t")):
        try:
            df = pd.read_csv(io.BytesIO(raw), delimiter=delim)
            if df.shape[1] >= 2:
                return df, label
        except Exception:  # noqa: BLE001 — try next delimiter
            continue
    raise ValueError("Could not parse the CSV — check the header row and "
                     "delimiter (comma, semicolon, or tab).")


def _validate(df: pd.DataFrame) -> str | None:
    if df.shape[1] < 2:
        return "CSV needs at least two columns."
    if df.shape[1] > _MAX_COLS:
        return f"CSV has {df.shape[1]} columns — the limit is {_MAX_COLS}."
    return None


def _intake(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Make an oversized upload work instead of failing:

    * > 1M rows  -> keep the first 1M, warn in the UI
    * text columns that look like dates/datetimes -> real datetime columns

    Returns (df, warnings).
    """
    notes: list[str] = []

    if len(df) > _MAX_ROWS:
        total = len(df)
        df = df.head(_MAX_ROWS).copy()
        notes.append(
            f"Large file: showing the first {_MAX_ROWS:,} of {total:,} rows — "
            f"rows after that were trimmed, not an error.")

    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or \
                pd.api.types.is_numeric_dtype(s) or \
                pd.api.types.is_bool_dtype(s):
            continue
        sample = s.dropna().astype(str).head(500)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed",
                               dayfirst=False)
        ok = parsed.notna().sum()
        if ok >= 0.8 * len(sample):
            conv = pd.to_datetime(df[col], errors="coerce", format="mixed")
            notes.append(f"Converted {col!r} to dates.")
            df[col] = conv
    return df, notes


# --------------------------------------------------------------------- #
# presets                                                                #
# --------------------------------------------------------------------- #

def _presets(df: pd.DataFrame, token: str) -> list[dict]:
    """Sensible one-click chart suggestions derived from the data."""
    cols = [str(c) for c in df.columns]
    kinds = _kinds(df)
    nums = [c for c in cols if kinds[c] == "number"]
    cats = [c for c in cols if kinds[c] in ("text", "date")]
    if len(cols) < 2:
        return []

    y = nums[-1] if nums else cols[1]  # metric is usually the last number
    x_trend = next((c for c in nums if c != y), None)  # numbers before y

    out: list[tuple[str, str, str]] = []
    if x_trend:
        out.append((x_trend, "", "line"))
    for x in cats[:2]:
        if x != y:
            out.append((x, "", "bar"))
    if len(cats) >= 2 and y in nums:
        out.append((cats[0], cats[1], "bar"))
    out = out[:4]

    res = []
    for x, cluster, kind in out:
        if x == y or not x:
            continue
        params = {"token": token, "x": x, "y": y, "kind": kind}
        if cluster:
            params["cluster"] = cluster
        label = f"{y} by {x}" if not cluster else f"{y} by {x} · {cluster}"
        res.append({"label": label, "href": "/?" + urlencode(params)})
    return res


def _preset_form(df: pd.DataFrame, token: str) -> dict:
    """First preset decoded into a form (used for auto-build on load)."""
    from urllib.parse import parse_qs, urlparse
    form = _reset_form()
    for p in _presets(df, token):
        q = parse_qs(urlparse(p["href"]).query)
        form["x"] = q.get("x", [""])[0]
        form["y"] = q.get("y", [""])[0]
        form["cluster"] = q.get("cluster", [""])[0]
        form["kind"] = q.get("kind", ["bar"])[0]
        break
    return form


# --------------------------------------------------------------------- #
# build                                                                  #
# --------------------------------------------------------------------- #

def _form() -> dict:
    f = {k: request.form.get(k, "") for k in
         ("x", "y", "cluster", "fcol", "kind", "stat", "title")}
    f["kind"] = f["kind"] or "bar"
    f["stat"] = f["stat"] or "auto"
    f["fval"] = [v for v in request.form.getlist("fval") if v]
    return f


def _form_from_args() -> dict:
    """Form state from URL query params (GET: share links / preset chips)."""
    p = request.args
    f = _reset_form()
    if p.get("x") and p.get("y"):
        f["x"], f["y"] = p["x"], p["y"]
        f["cluster"] = p.get("cluster", "")
        f["fcol"] = p.get("fcol", "")
        f["fval"] = [v.strip() for v in p.get("fval", "") .split(",") if v.strip()]
        f["kind"] = p.get("kind") or "bar"
        f["stat"] = p.get("stat") or "auto"
        f["title"] = p.get("title", "")
    return f


def _reset_form() -> dict:
    return {"x": "", "y": "", "cluster": "", "fcol": "", "fval": [],
            "kind": "bar", "stat": "auto", "title": ""}


def _share_url(token: str, form: dict) -> str:
    params: dict[str, str] = {"token": token, "x": form["x"], "y": form["y"]}
    if form.get("cluster"):
        params["cluster"] = form["cluster"]
    if form.get("fcol"):
        params["fcol"] = form["fcol"]
        fvals = [str(v) for v in (form.get("fval") or []) if str(v).strip()]
        if fvals:
            params["fval"] = ",".join(fvals)
    params["kind"] = form.get("kind") or "bar"
    if form.get("stat") and form["stat"] != "auto":
        params["stat"] = form["stat"]
    if form.get("title"):
        params["title"] = form["title"]
    return "/" + "?" + urlencode(params, doseq=False)


def _build(df: pd.DataFrame, form: dict) -> tuple[list | None, str | None]:
    """Render the chart; return (errors, data-url). data-url is a base64
    PNG embedded in the page — no shared state, so concurrent users never
    clobber each other's chart (the old app-global /chart.png race)."""
    if not form["x"] or not form["y"]:
        return ["Pick both an X and a Y column."], None
    if form["x"] == form["y"]:
        return ["X and Y must be different columns."], None
    try:
        c = Chart(df).x(form["x"]).y(form["y"], scale=None)
        if form["cluster"]:
            c = c.cluster(form["cluster"])
        if form["fcol"]:
            vals = [v.strip() for v in form["fval"] if v.strip()]
            if vals:
                col = df[form["fcol"]]
                if pd.api.types.is_numeric_dtype(col):
                    conv = []
                    for v in vals:
                        try:
                            fv = float(v)
                            conv.append(int(fv) if fv.is_integer() else fv)
                        except ValueError:
                            conv.append(v)
                    c = c.filter(form["fcol"], conv)
                else:
                    c = c.filter(form["fcol"], vals)
        c = c.plot(kind=form["kind"],
                   stat=None if form["stat"] == "auto" else form["stat"],
                   title=form["title"] or None)
        if c._fig is None:
            raise RuntimeError("Chart render produced no figure")
        buf = io.BytesIO()
        c._fig.savefig(buf, dpi=150, facecolor="white")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return None, "data:image/png;base64," + b64
    except Exception as e:  # noqa: BLE001 — surface a friendly error
        return [f"{type(e).__name__}: {e}"], None


# --------------------------------------------------------------------- #
# template                                                               #
# --------------------------------------------------------------------- #

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chart Builder — turn a CSV into a clean chart in seconds</title>
<meta name="description" content="Upload a CSV, pick an X and a Y, get a clean bar or line chart. No account, no install.">
<meta property="og:title" content="Chart Builder">
<meta property="og:description" content="Turn a CSV into a clean bar or line chart in seconds.">
<meta property="og:type" content="website">
<meta property="og:image" content="{{ base_url }}/og.png">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect x='4' y='16' width='6' height='12' rx='1' fill='%234C72B0'/%3E%3Crect x='13' y='8' width='6' height='20' rx='1' fill='%23DD8452'/%3E%3Crect x='22' y='12' width='6' height='16' rx='1' fill='%2355A868'/%3E%3C/svg%3E">
<style>
  :root { --ink:#2d2d2d; --line:#e3e3e3; --accent:#4C72B0; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         color: var(--ink); margin: 0; background: #fafafa; }
  .wrap { max-width: 980px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color: #777; font-size: 14px; margin-bottom: 20px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 10px;
          padding: 18px; margin-bottom: 18px; }
  .card-title { font-size: 13px; font-weight: 600; color: #555; text-transform: uppercase;
                letter-spacing: .04em; margin: 0 0 12px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px 14px; }
  label { display: block; font-size: 12px; font-weight: 600; color: #555;
           margin-bottom: 4px; text-transform: uppercase; letter-spacing: .04em; }
  select, input[type=text], input[type=file] {
    width: 100%; padding: 7px 9px; border: 1px solid #ccc; border-radius: 6px;
    font-size: 14px; background: #fff; }
  select[multiple] { padding: 4px; }
  select[multiple] option { padding: 4px 8px; }
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
  .ghost:hover { background: #f2f6fc; }
  .dropzone { border: 2px dashed #c9d4e5; border-radius: 8px; padding: 14px;
              text-align: center; background: #fcfdff; }
  .dropzone.drag { border-color: var(--accent); background: #eef3fa; }
  .dropzone p { margin: 4px 0; color: #667; font-size: 13px; }
  .chips { display: flex; gap: 8px; flex-wrap: wrap; }
  .chip { display: inline-block; padding: 7px 14px; border-radius: 999px;
          background: #f0f4fa; color: var(--accent); font-size: 13px; font-weight: 600;
          text-decoration: none; border: 1px solid #d8e2f0; }
  .chip:hover { background: #e2ebf7; }
  .chart-box { text-align: center; }
  .chart-box img { max-width: 100%; height: auto; border: 1px solid var(--line);
                   border-radius: 6px; background: #fff; }
  .err { background: #fdecea; color: #b03030; border: 1px solid #f0c4c4;
          border-radius: 6px; padding: 8px 12px; font-size: 13px; margin-bottom: 14px; }
  .note { color: #557; background: #eef3fa; border: 1px solid #d8e2f0;
           border-radius: 6px; padding: 8px 12px; font-size: 13px; margin-bottom: 14px; }
  .info { color: #777; font-size: 13px; }
  .dl { font-size: 13px; }
  .dl a { color: var(--accent); font-weight: 600; text-decoration: none; }
  .hint { font-size: 12px; color: #999; margin-top: 3px; }
  table.preview { border-collapse: collapse; width: 100%; font-size: 13px; }
  table.preview th, table.preview td { border: 1px solid var(--line);
          padding: 5px 8px; text-align: left; }
  table.preview th { background: #f7f8fa; font-weight: 600; }
  .badge { display: inline-block; font-size: 10px; font-weight: 700; letter-spacing: .03em;
           text-transform: uppercase; color: #666; background: #eef0f3; border-radius: 4px;
           padding: 1px 5px; margin-left: 6px; vertical-align: middle; }
  footer { margin-top: 26px; color: #999; font-size: 12px; line-height: 1.5; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Chart Builder</h1>
  <div class="sub">Upload a CSV (or load the sample), pick an X and a Y, get a clean bar or line chart — no account, no install.</div>

  {% for e in errors %}
  <div class="err">{{ e }}</div>
  {% endfor %}
  {% for n in notes %}
  <div class="note">{{ n }}</div>
  {% endfor %}

  <form method="post" enctype="multipart/form-data" class="card" id="loadform">
    <div class="card-title">1 · Load data</div>
    <input type="hidden" name="token" value="{{ token or '' }}">
    <input type="hidden" name="mode" value="load">
    <div class="dropzone" id="dropzone">
      <p><strong>Drag &amp; drop your CSV here</strong>, or</p>
      <input type="file" name="file" accept=".csv,.txt,text/csv" id="fileinput">
      <div class="hint">CSV with a header row — comma, semicolon, or tab separated. Max 20&nbsp;MB.</div>
    </div>
    <div class="btn-row">
      <button type="submit" id="loadbtn">Upload CSV</button>
      <button type="submit" name="sample" value="sample" class="ghost">Load sample data</button>
    </div>
  </form>

  {% if head_rows is not none %}
  <div class="card">
    <div class="card-title">Your data · {{ rows }} rows × {{ cols | length }} columns
      {% if meta_name %}· {{ meta_name }}{% endif %}</div>
    <table class="preview">
      <tr>
        {% for c in cols %}
        <th>{{ c }}<span class="badge">{{ kinds[c] }}</span></th>
        {% endfor %}
      </tr>
      {% for r in head_rows %}
      <tr>
        {% for c in cols %}
        <td>{{ r[c] }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </table>
    <div class="hint">showing first {{ head_rows | length }} of {{ rows }} rows</div>
  </div>
  {% endif %}

  <form method="post" class="card" id="buildform">
    <div class="card-title">2 · Chart</div>
    <input type="hidden" name="token" value="{{ token or '' }}">
    <input type="hidden" name="mode" value="build">
    <div class="grid">
      <div>
        <label for="x">X axis</label>
        <select name="x" id="x" required>
          {% for c in cols %}<option value="{{ c }}" {{ 'selected' if c == form.x }}>{{ c }} ({{ kinds[c] }})</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="y">Y axis</label>
        <select name="y" id="y" required>
          {% for c in cols %}<option value="{{ c }}" {{ 'selected' if c == form.y }}>{{ c }} ({{ kinds[c] }})</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="cluster">Cluster (optional)</label>
        <select name="cluster" id="cluster">
          <option value="">— none —</option>
          {% for c in cols %}<option value="{{ c }}" {{ 'selected' if c == form.cluster }}>{{ c }} ({{ kinds[c] }})</option>{% endfor %}
        </select>
      </div>
      <div>
        <label for="fcol">Filter (optional)</label>
        <select name="fcol" id="fcol">
          <option value="">— none —</option>
          {% for c in cols %}<option value="{{ c }}" {{ 'selected' if c == form.fcol }}>{{ c }} ({{ kinds[c] }})</option>{% endfor %}
        </select>
      </div>
    </div>
    {% if form.fcol %}
    <div style="margin-top:14px">
      <label for="fval">Filter values <span class="hint">(none selected = no filter)</span></label>
      <select name="fval" id="fval" multiple size="6">
        {% for v in fval_opts %}
        <option value="{{ v }}" {{ 'selected' if v in form.fval }}>{{ v }}</option>
        {% endfor %}
        {% if not fval_opts %}<option value="">(no values)</option>{% endif %}
      </select>
    </div>
    {% endif %}
    <div class="row" style="margin-top:14px">
      <div>
        <label>Chart</label>
        <div class="radio" role="radiogroup" aria-label="Chart type">
          <label><input type="radio" name="kind" value="bar" {{ 'checked' if form.kind == 'bar' }} /> Bar</label>
          <label><input type="radio" name="kind" value="line" {{ 'checked' if form.kind == 'line' }} /> Line</label>
        </div>
      </div>
      <div>
        <label for="stat">Y statistic</label>
        <select name="stat" id="stat">
          <option value="auto" {{ 'selected' if form.stat == 'auto' }}>Auto</option>
          <option value="mean" {{ 'selected' if form.stat == 'mean' }}>Mean</option>
          <option value="median" {{ 'selected' if form.stat == 'median' }}>Median</option>
          <option value="sum" {{ 'selected' if form.stat == 'sum' }}>Sum</option>
          <option value="count" {{ 'selected' if form.stat == 'count' }}>Count</option>
        </select>
      </div>
      <div>
        <label for="title">Title (optional)</label>
        <input type="text" name="title" id="title" value="{{ form.title }}">
      </div>
    </div>
    <div class="btn-row">
      <button type="submit">Build chart</button>
    </div>
  </form>

  {% if chart_url %}
  <div class="card chart-box">
    <img src="{{ chart_url }}" alt="chart" id="chartimg">
    <div class="dl" style="margin-top:10px">
      <a href="{{ chart_url }}" download="chart.png">Download PNG</a>
      &middot;
      <button type="button" class="ghost" id="copylink">Copy share link</button>
      <div class="hint">share link works while this dataset is loaded — it expires when the app restarts</div>
    </div>
  </div>
  {% endif %}

  {% if presets %}
  <div class="card">
    <div class="card-title">Try one of these</div>
    <div class="chips">
      {% for p in presets %}<a class="chip" href="{{ p.href }}">{{ p.label }}</a>{% endfor %}
    </div>
  </div>
  {% endif %}

  <p class="info">{{ rows }} rows &times; {{ cols | length }} columns
    {% if token %}&middot; dataset <code>{{ token[:8] }}</code>{% endif %}</p>

  <footer>
    Your CSV is held in memory on this server for the session — it is never written to disk and is gone when the app restarts.
  </footer>
</div>

<script>
(function () {
  var TOKEN = {{ token | tojson }};

  // drag & drop upload
  var dz = document.getElementById('dropzone');
  var fi = document.getElementById('fileinput');
  var lf = document.getElementById('loadform');
  ['dragenter', 'dragover'].forEach(function (ev) {
    dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add('drag'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove('drag'); });
  });
  dz.addEventListener('drop', function (e) {
    var files = e.dataTransfer.files;
    if (files && files.length) {
      fi.files = files;
      lf.submit();
    }
  });

  // filter values picker — options are server-rendered with selection;
  // only re-fetch when the user picks a different filter column
  var fcol = document.getElementById('fcol');
  var fval = document.getElementById('fval');
  function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                     .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function fillFval() {
    if (!fval) return;
    if (!fcol.value) { fval.innerHTML = ''; return; }
    fetch('/colvals?token=' + encodeURIComponent(TOKEN || '') +
          '&col=' + encodeURIComponent(fcol.value))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        fval.innerHTML = (d.values || []).map(function (v) {
          return '<option value="' + esc(v) + '">' + esc(v) + '</option>';
        }).join('') || '<option value="">(no values)</option>';
      });
  }
  if (fcol) fcol.addEventListener('change', fillFval);

  // copy share link (server-built, includes dataset + full spec)
  var cl = document.getElementById('copylink');
  var SHARE_URL = {{ share_url | tojson }};
  if (cl && SHARE_URL) cl.addEventListener('click', function () {
    var link = location.origin + SHARE_URL;
    navigator.clipboard.writeText(link).then(function () {
      cl.textContent = 'Copied!';
      setTimeout(function () { cl.textContent = 'Copy share link'; }, 1500);
    });
  });
})();
</script>
</body>
</html>
"""


def _render(df, form, token, chart_url=None, errors=None, notes=None):
    cols = _cols(df)
    kinds = _kinds(df)
    share = _share_url(token or "sample", form) if chart_url else ""
    fval_opts: list[str] = []
    if form.get("fcol") and form["fcol"] in df.columns:
        fval_opts = [str(v) for v in df[form["fcol"]].dropna().unique()[:200]]
    return render_template_string(
        PAGE, df=df, cols=cols, form=form, token=token,
        chart_url=chart_url, share_url=share,
        errors=errors or [], notes=notes or [],
        rows=len(df), kinds=kinds,
        fval_opts=fval_opts,
        head_rows=[{c: ("" if pd.isna(v) else str(v)) for c, v in row.items()}
                   for row in df.head(5).to_dict(orient="records")],
        meta_name=_meta_name(token), presets=_presets(df, token or "sample"),
        base_url=request.host_url.rstrip("/"),
    )


def _cols(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]


# --------------------------------------------------------------------- #
# routes                                                                 #
# --------------------------------------------------------------------- #

@app.route("/")
def index():
    token = request.args.get("token")
    df = _DATASETS.get(token) if token else None
    notes: list[str] = []
    if token and df is None and token != "sample":
        notes.append("That dataset is no longer loaded (the app restarted). "
                     "Upload it again — settings are not stored.")
        token = None
    if df is None:
        df = _sample()
        token = "sample"

    # selection: URL params win (share links / preset chips), else first preset
    form = _form_from_args()
    if not form["x"] and not form["y"]:
        form = _preset_form(df, token)
    errors, chart_url = None, None
    if form["x"] and form["y"]:
        errors, chart_url = _build(df, form)
    return _render(df, form, token, chart_url, errors, notes)


@app.route("/", methods=["POST"])
def index_post():
    token = request.form.get("token")
    df = _DATASETS.get(token) if token else None
    form = _form()
    mode = request.form.get("mode", "")

    if mode == "load":
        notes: list[str] = []
        err = None
        delim = ""
        f = request.files.get("file")
        if f is not None and f.filename:
            try:
                newdf, delim = _read_csv_smart(f.read())
            except Exception as e:  # noqa: BLE001
                newdf, err = None, f"Could not read CSV: {e}"
            else:
                problem = _validate(newdf)
                if problem:
                    newdf, err = None, problem
                else:
                    newdf, notes = _intake(newdf)
            if newdf is not None:
                token = uuid.uuid4().hex
                _remember(token, newdf, f.filename)
                df = newdf
                form = _reset_form()
                notes.append(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns "
                             f"({delim}-separated).")
        elif request.form.get("sample") == "sample" or df is None:
            df = _sample()
            token = "sample"
            form = _reset_form()
            notes.append("Sample data loaded.")
        if err:
            if df is None:
                df = _sample()
                token = "sample"
            return _render(df, _reset_form(), token, errors=[err])
        errors, chart_url = None, None
        if form["x"] and form["y"]:
            errors, chart_url = _build(df, form)
        else:
            # auto-build first preset so value is visible immediately
            form = _preset_form(df, token or "sample")
            errors, chart_url = _build(df, form)
        return _render(df, form, token, chart_url, errors, notes)

    # mode == build (default)
    if df is None:
        df = _sample()
        token = "sample"
    errors, chart_url = _build(df, form)
    return _render(df, form, token, chart_url, errors)


@app.route("/colvals")
def colvals():
    token = request.args.get("token")
    col = request.args.get("col")
    df = _DATASETS.get(token)
    if df is None or col not in df.columns:
        return jsonify(values=[], numeric=False)
    vals = df[col].dropna().unique()[:200]
    numeric = bool(pd.api.types.is_numeric_dtype(df[col]))
    return jsonify(numeric=numeric, values=[str(v) for v in vals])


@app.route("/health")
def health():
    return "ok"


_og_png: bytes | None = None


@app.route("/og.png")
def og_png():
    """Social/share card (1200×630), generated once and cached."""
    global _og_png
    if _og_png is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6.3))
        vals = [42, 68, 55, 88, 74]
        ax.bar(range(5), vals, width=0.55,
               color=PALETTE[:5], edgecolor="white")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(0, 100)
        fig.text(0.06, 0.72, "Chart Builder", fontsize=30,
                 fontweight="bold", color="#2d2d2d")
        fig.text(0.06, 0.60,
                 "Upload a CSV → clean bar or line charts in seconds.",
                 fontsize=15, color="#555555")
        fig.text(0.06, 0.28,
                 "Pick an X and a Y · cluster · filter · share link",
                 fontsize=12, color="#888888")
        buf = io.BytesIO()
        fig.savefig(buf, dpi=100, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        _og_png = buf.read()
    return _og_png, 200, {"Content-Type": "image/png"}


if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8011)))
