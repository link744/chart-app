"""Regression: date/datetime X axes + oversized-upload intake."""
import io
import math
import numpy as np
import pandas as pd
import chart_app
from chartbuilder import Chart

fails = []
def check(name, cond, detail=""):
    if not cond:
        fails.append(name)
        print("FAIL", name, detail)

# --- 1. date line: 20 months -> monthly bins, real date positions ---
df = pd.DataFrame({
    "date": pd.date_range("2025-01-01", periods=20, freq="MS"),
    "v": range(20),
})
c = Chart(df).x("date").y("v").plot(kind="line")
check("1.date-line xscale", c._xscale == "date")
check("1.date-line freq", c._datefreq == "M", c._datefreq)
series = c._series()
check("1.date-line n bins", len(series[0][1]) == 20)
check("1.date-line real positions", min(series[0][3]) > 10000,
      min(series[0][3]))
check("1.date-line gap-safe", len(series[0][1]) == 20)
c.save("/tmp/t1_line.png")

# --- 2. sparse dates: <=12 distinct dates -> one point per timestamp ---
df2 = pd.DataFrame({
    "date": pd.to_datetime(["2026-01-05", "2026-01-08", "2026-01-12",
                            "2026-01-02", "2026-01-12"]),
    "amount": [3, 1, 5, 2, 2],
})
c2 = Chart(df2).x("date").y("amount", "continuous").plot(kind="bar", stat="sum")
check("2.sparse freq ts", c2._datefreq == "ts", c2._datefreq)
check("2.sparse labels", c2._ylabels == ["Jan 02 00:00", "Jan 05 00:00",
                                         "Jan 08 00:00", "Jan 12 00:00"],
      c2._ylabels)
h = [p.get_height() for p in c2._ax.patches]
check("2.sparse values", h == [2, 3, 1, 7], h)
check("2.sparse n bars", len(c2._ax.patches) == 4, len(c2._ax.patches))

# --- 2b. dense dates: >12 distinct -> daily bins ---
# 13 distinct dates, Jan 1 00:00 -> Jan 13 00:00 (exactly a 12-day span,
# the D rung of the ladder). All bins filled, no NaN.
df2b = pd.DataFrame({
    "date": pd.to_datetime([
        "2026-01-01 00:00:00", "2026-01-02 12:00:00", "2026-01-03 12:00:00",
        "2026-01-04 12:00:00", "2026-01-05 12:00:00", "2026-01-06 12:00:00",
        "2026-01-07 12:00:00", "2026-01-08 12:00:00", "2026-01-09 12:00:00",
        "2026-01-10 12:00:00", "2026-01-11 12:00:00", "2026-01-12 12:00:00",
        "2026-01-13 00:00:00"]),
    "amount": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
})
c2b = Chart(df2b).x("date").y("amount", "continuous").plot(kind="bar", stat="sum")
check("2b.dense freq D", c2b._datefreq == "D", c2b._datefreq)
h = [p.get_height() for p in c2b._ax.patches]
check("2b.dense n bins", len(c2b._ax.patches) == 13, len(c2b._ax.patches))
check("2b.dense no NaN", not any(math.isnan(x) for x in h), h)
check("2b.dense values", h[0] == 1 and h[6] == 7 and h[12] == 13, h)

# --- 3. weekly binning: 61 days ---
df3 = pd.DataFrame({
    "date": pd.date_range("2025-11-01", periods=61, freq="D"),
    "v": range(61),
})
c3 = Chart(df3).x("date").y("v", "continuous").plot(kind="line")
check("3.weekly freq", c3._datefreq == "W-SUN", c3._datefreq)
check("3.weekly 10 weeks", len(c3._ylabels) == 10, c3._ylabels)

# --- 4. monthly data: 36 monthly rows -> 36 monthly bars (no roll-up) ---
# Density-aware: one point per bin. The old span-only picker collapsed
# these into 12 quarterly bars.
df4 = pd.DataFrame({
    "date": pd.date_range("2023-01-01", "2025-12-01", freq="MS"),
    "v": range(36),
})
c4 = Chart(df4).x("date").y("v", "continuous").plot(kind="bar")
check("4.monthly freq M", c4._datefreq == "M", c4._datefreq)
check("4.monthly 36 bins", len(c4._ylabels) == 36, len(c4._ylabels))
check("4.monthly labels", c4._ylabels[0] == "Jan 2023"
      and c4._ylabels[-1] == "Dec 2025", c4._ylabels[:2])
h4 = [p.get_height() for p in c4._ax.patches]
check("4.monthly no NaN", not any(math.isnan(x) for x in h4), h4)

# --- 5. quarterly: 111 monthly rows over 10 years -> Q-DEC (40 bars) ---
dates5 = pd.date_range("2015-06-01", "2024-06-01", freq="MS")
df5 = pd.DataFrame({"date": dates5, "v": range(len(dates5))})
c5 = Chart(df5).x("date").y("v", "continuous").plot(kind="bar")
check("5.quarterly freq Q-DEC", c5._datefreq == "Q-DEC", c5._datefreq)
check("5.quarterly labels", c5._ylabels[0] == "2015Q2"
      and c5._ylabels[-1] == "2024Q2", c5._ylabels[:2])
check("5.quarterly 37 bins", len(c5._ylabels) == 37, len(c5._ylabels))

# --- 6. datetime (not just date) column, 2-day span -> daily bins ---
df6 = pd.DataFrame({
    "ts": pd.to_datetime(["2026-01-15 08:30", "2026-01-15 14:10",
                          "2026-01-16 09:00", "2026-01-16 11:20"]),
    "v": [1, 2, 3, 4],
})
c6 = Chart(df6).x("ts").y("v", "continuous").plot(kind="line")
check("6.datetime works", c6._xscale == "date", c6._xscale)

# --- 7. nominal / continuous X still work (pre-existing behavior) ---
d = pd.DataFrame({"cat": ["a", "b", "a", "b", "c", "c", "a", "b", "c", "a", "b", "c", "a", "b"],
                  "n": list(range(14)),
                  "num": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0,
                          11.0, 12.0, 13.0, 14.0]})
cb = Chart(d).x("cat").y("n").plot(kind="bar")
check("7.nominal bar", cb._xscale == "nominal"
      and len(cb._ax.patches) == 3, (cb._xscale, len(cb._ax.patches)))
cn = Chart(d).x("num").y("n").plot(kind="line")
check("7.continuous line", cn._xscale == "continuous", cn._xscale)

# --- 8. oversized intake: >1M rows -> crop + warning ---
big = pd.DataFrame({"x": range(chart_app._MAX_ROWS + 1000),
                    "v": range(chart_app._MAX_ROWS + 1000)})
out, notes = chart_app._intake(big.copy())
check("8a.crop len", len(out) == chart_app._MAX_ROWS, len(out))
check("8b.crop warning", any("rows" in n for n in notes), notes)

# --- 9. date-like text column auto-converted ---
df8 = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-08", "2026-01-15"],
    "v": [1, 2, 3],
})
out8, n8 = chart_app._intake(df8.copy())
check("9.date convert", pd.api.types.is_datetime64_any_dtype(out8["date"]),
      out8["date"].dtype)
check("9b.convert note", any("Converted" in n for n in n8), n8)

# --- 10. non-date text column stays untouched ---
df9 = pd.DataFrame({"col": ["abc", "def", "ghi", "jkl"], "v": range(4)})
out9, n9 = chart_app._intake(df9.copy())
check("10.text untouched",
      not pd.api.types.is_datetime64_any_dtype(out9["col"]) and n9 == [])

# --- 11. end-to-end: upload a date CSV through Flask, build a chart ---
raw = (b"id,date,amount\n"
       b"0,2026-01-01,5\n1,2026-01-08,9\n2,2026-01-15,3\n")
with chart_app.app.test_client() as tc:
    r = tc.post("/", data={
        "mode": "load", "file": (io.BytesIO(raw), "dates.csv")},
        content_type="multipart/form-data")
    html = r.get_data(as_text=True)
    check("11.load 200", r.status_code == 200)
    check("11.date note in UI", "Converted" in html and "to dates." in html)

# --- 12. end-to-end: oversized upload degrades (warning, not failure) ---
rows = [f"{i},2026-01-{i % 28 + 1:02d},{i % 7}" for i in range(500)]
bigcsv = ("id,date,amount\n" + "\n".join(rows) + "\n").encode()
old_max = chart_app._MAX_ROWS
chart_app._MAX_ROWS = 50
try:
    with chart_app.app.test_client() as tc:
        r = tc.post("/", data={
            "mode": "load", "file": (io.BytesIO(bigcsv), "big.csv")},
            content_type="multipart/form-data")
        html = r.get_data(as_text=True)
        check("12.oversized 200", r.status_code == 200)
        check("12.crop warning in UI", "trimmed" in html
              and "first 50 of 500" in html)
        check("12.chart still renders",
              "data:image/png;base64," in html)
finally:
    chart_app._MAX_ROWS = old_max

# --- 13. forced scale="date" on a non-date column errors cleanly ---
try:
    Chart(pd.DataFrame({"a": [1, 2, 3], "v": [1, 2, 3]})) \
        .x("a", "date").y("v").plot()
    check("13.forced date guard", False, "no error raised")
except ValueError:
    pass
except Exception as e:
    check("13.forced date guard", False, repr(e))

# --- 14. THE task example: 3 tz-aware timestamps ~10 min apart ---
df14 = pd.DataFrame({
    "ts": pd.to_datetime([
        "2026-09-28 13:09:23.878000+00:00",
        "2026-09-28 13:14:24.049000+00:00",
        "2026-09-28 13:19:23.906000+00:00",
    ]),
    "calls": [10, 25, 15],
})
c14 = Chart(df14).x("ts").y("calls").plot(kind="line")
check("14.intraday ts xscale", c14._xscale == "date", c14._xscale)
check("14.intraday ts freq", c14._datefreq == "ts", c14._datefreq)
s14 = c14._series()
check("14.intraday 3 points", len(s14[0][1]) == 3, s14[0][1])
check("14.intraday labels", s14[0][1] == ["Sep 28 13:09", "Sep 28 13:14",
                                           "Sep 28 13:19"], s14[0][1])
check("14.intraday values", s14[0][2] == [10.0, 25.0, 15.0], s14[0][2])
from matplotlib import dates as mdates
back = [mdates.num2date(p).strftime("%Y-%m-%d %H:%M:%S") for p in s14[0][3]]
check("14.intraday real positions",
      back == ["2026-09-28 13:09:23", "2026-09-28 13:14:24",
               "2026-09-28 13:19:23"], back)
c14.save("/tmp/t14_line.png")

# --- 15. same example as a bar chart: 3 real bars, no NaN collapse ---
c15 = Chart(df14).x("ts").y("calls").plot(kind="bar", stat="sum")
h15 = [p.get_height() for p in c15._ax.patches]
check("15.intraday bar values", h15 == [10.0, 25.0, 15.0], h15)
check("15.intraday bar n", len(c15._ax.patches) == 3, len(c15._ax.patches))
# bar widths must be a sliver of a day (intraday), not 0.8 days
w15 = [p.get_width() for p in c15._ax.patches]
check("15.intraday bar width", all(0 < w < 0.01 for w in w15), w15)
c15.save("/tmp/t15_bar.png")

# --- 16. dense intraday: minute bins; one empty minute stays NaN ---
# data in 13:00-13:03 and 13:05-13:09 (13:04 empty) -> 10 minute bins,
# one NaN
full = pd.date_range("2026-09-28 13:00:00", "2026-09-28 13:09:57", freq="3s")
keep = pd.Series(full).dt.strftime("%H:%M").ne("13:04").values
df16 = pd.DataFrame({"ts": full[keep]})
df16["v"] = list(range(len(df16)))
c16 = Chart(df16).x("ts").y("v").plot(kind="line")
check("16.dense-min freq", c16._datefreq == "min", c16._datefreq)
s16 = c16._series()
check("16.dense-min n bins", len(s16[0][1]) == 10, s16[0][1])
check("16.dense-min one empty bin NaN",
      sum(math.isnan(v) for v in s16[0][2]) == 1, s16[0][2])
check("16.dense-min empty is 13:04",
      math.isnan(s16[0][2][4]) and s16[0][1][4] == "13:04",
      (s16[0][1], s16[0][2]))
check("16.dense-min label", c16._ylabels[0] == "13:00"
      and c16._ylabels[-1] == "13:09", c16._ylabels[:2])

# --- 17. hour-scale: 30 hourly points over 5 days -> 4H bins ---
df17 = pd.DataFrame({
    "ts": pd.date_range("2026-01-01 00:00", periods=120, freq="h"),
    "v": range(120),
})
c17 = Chart(df17).x("ts").y("v").plot(kind="line")
check("17.4h freq", c17._datefreq == "4h", c17._datefreq)
check("17.4h label fmt", c17._ylabels[0] == "Jan 01 00:00", c17._ylabels[:2])

# --- 18. clustered line on datetime X ---
df18 = pd.DataFrame({
    "ts": pd.to_datetime([
        "2026-09-28 13:09:23+00:00", "2026-09-28 13:14:24+00:00",
        "2026-09-28 13:19:23+00:00",
        "2026-09-28 13:09:25+00:00", "2026-09-28 13:14:25+00:00",
        "2026-09-28 13:19:25+00:00"]),
    "svc": ["a", "a", "a", "b", "b", "b"],
    "v": [1, 2, 3, 4, 5, 6],
})
c18 = Chart(df18).x("ts").y("v").cluster("svc").plot(kind="line")
s18 = c18._series()
check("18.cluster n series", len(s18) == 2, len(s18))
check("18.cluster 3 pts each",
      all(len(s[1]) == 3 for s in s18), [len(s[1]) for s in s18])

# --- 19. cluster + bar on date X is still rejected ---
try:
    Chart(df18).x("ts").y("v").cluster("svc").plot(kind="bar")
    check("19.cluster bar guard", False, "no error raised")
except ValueError:
    pass

# --- 20. the task's EXACT CSV through the real pipeline ---
# ms precision + "+00:00" tz, as a CSV upload would arrive, straight
# through _read_csv_smart -> _intake -> Chart (no manual pre-conversion).
raw20 = (
    b"ts,calls\n"
    b"2026-09-28 13:09:23.878000+00:00,10\n"
    b"2026-09-28 13:14:24.049000+00:00,25\n"
    b"2026-09-28 13:19:23.906000+00:00,15\n"
)
df20, delim20 = chart_app._read_csv_smart(raw20)
check("20.delim comma", delim20 == "comma", delim20)
check("20.raw is text",
      not pd.api.types.is_datetime64_any_dtype(df20["ts"]), df20["ts"].dtype)
out20, notes20 = chart_app._intake(df20.copy())
check("20.intake converts",
      pd.api.types.is_datetime64_any_dtype(out20["ts"]), out20["ts"].dtype)
check("20.intake note", any("Converted" in n for n in notes20), notes20)
c20 = Chart(out20).x("ts").y("calls").plot(kind="line")
check("20.date axis", c20._xscale == "date" and c20._datefreq == "ts",
      (c20._xscale, c20._datefreq))
s20 = c20._series()
check("20.values", s20[0][2] == [10.0, 25.0, 15.0], s20[0][2])

# --- 21. THE collapse bug: ~2000 rows at 5-min cadence over ~14 days ---
# Old span-only picker: 13.8-day span -> W-SUN -> exactly 2 x-axis
# samples. Density-aware: 28 half-day bars (finest rung with <=40 bins),
# each holding ~72 points.
full21 = pd.date_range("2026-09-15 03:07:23", "2026-09-28 21:59:59",
                      freq="5min")
df21 = pd.DataFrame({"ts": full21, "v": np.arange(len(full21))})
c21 = Chart(df21).x("ts").y("v").plot(kind="bar", stat="mean")
check("21.intraday-5min freq 12h", c21._datefreq == "12h", c21._datefreq)
s21 = c21._series()
check("21.intraday-5min 28 bins", len(s21[0][1]) == 28, len(s21[0][1]))
h21 = [p.get_height() for p in c21._ax.patches]
check("21.intraday-5min all bars populated", all(x > 0 for x in h21))
check("21.intraday-5min no NaN", not any(math.isnan(x) for x in h21))
c21.save("/tmp/t21_intraday_5min.png")

if fails:
    print("\nFAILURES:", fails)
    raise SystemExit(1)
print("ALL PASS (21 sections)")
