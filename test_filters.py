"""Run: python test_filters.py   (uses MySQL if reachable) |  FORCE_CSV=1 python test_filters.py  (CSV only).
Prints PASS/FAIL per check and a fingerprint so MySQL and CSV runs can be compared."""
import json, sys
from urllib.parse import quote
import app

c = app.app.test_client()
with c.session_transaction() as s: s["user"] = "admin"
fails = 0
def check(name, ok, extra=""):
    global fails
    fails += (not ok); print(("PASS " if ok else "FAIL ") + name, extra)

f = c.get("/api/filters").json
print("SOURCE:", f["source"], "| range", f["min_date"], "->", f["max_date"], "| cities", f["cities"])
check("8 cities discovered from data", len(f["cities"]) == 8)
check("date range matches dataset", (f["min_date"], f["max_date"]) == ("2024-01-01", "2025-12-31"))
fp = {}
for city in f["cities"]:
    r = c.get("/api/summary?city=" + quote(city)).json
    check(f"{city}: city only = 731", r.get("records") == 731, r.get("records"))
    r = c.get(f"/api/summary?city={quote(city)}&start=2025-01-01&end=2025-12-31").json
    check(f"{city}: city + 2025 = 365", r.get("records") == 365, r.get("records"))
    fp[city] = [r["aqi"], r["pm25"], r["records"]]
    ch = c.get(f"/api/charts?city={quote(city)}&start=2024-06-01&end=2024-06-30").json
    check(f"{city}: June 2024 trend has 30 days", len(ch["trend"]["x"]) == 30)
for variant in [" delhi ", "DELHI", "DeLhI", "Delhi%20", "%20delhi"]:
    r = c.get("/api/summary?city=" + variant).json
    check(f"variant {variant!r} -> Delhi", r.get("records") == 731, r.get("records"))
r = c.get("/api/summary").json;                                     check("All cities = 5848", r.get("records") == 5848, r.get("records"))
r = c.get("/api/summary?start=2024-01-01&end=2025-12-31").json;     check("All cities + full range = 5848", r.get("records") == 5848)
r = c.get("/api/summary?start=2026-01-01&end=2026-12-31");          check("2026 range -> friendly empty message", r.status_code == 400 and "No records" in r.json["error"])
r = c.get("/api/summary?city=Atlantis");                            check("unknown city -> friendly error", r.status_code == 400 and "Available" in r.json["error"])
r = c.get("/api/summary?start=2025-05-01&end=2025-01-01");          check("start after end -> friendly error", r.status_code == 400)
r = c.get("/api/summary?start=garbage");                            check("bad date -> friendly error", r.status_code == 400)
r = c.get("/api/summary?city=Delhi&cat=Good&start=2025-07-01&end=2025-08-31"); print("INFO Delhi+Good+Jul-Aug:", r.status_code, r.json)
r = c.get("/api/olap/slice?city=%20pune%20").json;                  check("OLAP slice normalised city", "rows" in r and len(r["rows"]) == 24)
r = c.get("/api/olap/dice?cities=delhi,MUMBAI&year=2025&metric=pm25").json; check("OLAP dice normalised cities", {x[0] for x in r["rows"]} == {"Delhi", "Mumbai"})
print("FINGERPRINT", json.dumps(fp, sort_keys=True))
sys.exit(1 if fails else 0)
