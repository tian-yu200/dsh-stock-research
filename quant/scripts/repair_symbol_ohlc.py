import urllib.request, shutil, sys, datetime, json
import pandas as pd
from pathlib import Path
sym = sys.argv[1]
proj = Path(__file__).resolve().parent.parent
csv = proj / "data" / "stocks" / (sym + ".csv")
metap = proj / "data" / "stocks" / (sym + ".meta.json")
qdir = proj / "data" / "stocks" / "quarantine"
qdir.mkdir(parents=True, exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
shutil.copy2(csv, qdir / (sym + ".pre-repair-" + ts + ".csv"))
shutil.copy2(metap, qdir / (sym + ".pre-repair-" + ts + ".meta.json"))
url = "http://d.10jqka.com.cn/v2/line/hs_" + sym + "/01/last.js"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "http://q.10jqka.com.cn/"})
raw = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", errors="replace")
mark = chr(34) + "data" + chr(34) + ":" + chr(34)
s = raw.split(mark, 1)[1]
s = s.rsplit(chr(34), 2)[0]
ths = {}
for row in s.split(";"):
    c = row.split(",")
    if len(c) < 7 or not c[0].isdigit():
        continue
    d = c[0]
    ds = d[0:4] + "-" + d[4:6] + "-" + d[6:8]
    tn = float(c[7]) if len(c) > 7 and c[7] else None
    ths[ds] = [float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]), float(c[6]), tn]
df = pd.read_csv(csv)
start = "2026-08-28"
fixed = 0
for i, r in df.iterrows():
    ds = str(r["date"])[:10]
    if ds >= start and ds in ths:
        t = ths[ds]
        df.at[i, "open"] = t[0]
        df.at[i, "high"] = t[1]
        df.at[i, "low"] = t[2]
        df.at[i, "close"] = t[3]
        fixed += 1
added = 0
lastd = str(df["date"].max())[:10]
for ds in sorted(ths):
    if ds > lastd:
        t = ths[ds]
        turn = (t[6] / 100.0) if t[6] is not None else 0.0
        df.loc[len(df)] = [ds, t[0], t[1], t[2], t[3], t[4], t[5], turn]
        added += 1
df = df.sort_values("date").reset_index(drop=True)
df.to_csv(csv, index=False, encoding="utf-8-sig")
meta = json.loads(metap.read_text(encoding="utf-8"))
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
meta["rows"] = int(len(df))
meta["data_end_date"] = str(df["date"].max())[:10]
meta["fetched_at"] = now
meta["repaired_from"] = "ths-d.10jqka"
meta["repaired_range"] = start + ".." + str(df["date"].max())[:10]
meta["repaired_at"] = now
metap.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
tail = df.tail(6).copy()
tail["implied"] = tail["amount"] / tail["volume"]
tail["dev"] = (tail["implied"] - tail["close"]).abs() / tail["close"]
print("fixed=" + str(fixed) + " added=" + str(added) + " rows=" + str(len(df)))
print(tail[["date", "close", "implied", "dev"]].to_string(index=False))
print("maxdev=" + str(float(tail["dev"].max())))
