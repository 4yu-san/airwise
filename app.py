import os, io, logging
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from generate_dataset import compute_aqi, category
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("airwise")
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-only-secret")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_HASH = generate_password_hash(os.getenv("ADMIN_PASSWORD", "admin123"), method="pbkdf2:sha256")
USER_NAME = os.getenv("USER_NAME", "user")
USER_HASH = generate_password_hash(os.getenv("USER_PASSWORD", "user123"), method="pbkdf2:sha256")
FEATURES = ["pm25", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity"]
CATS = ["Good", "Moderate", "Poor", "Very Poor", "Severe"]
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MEASURES = {"aqi": "AQI", "pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "o3": "O3", "co": "CO"}
_cache = {}
_state = {"source_mode": "auto", "fallback_reason": None, "loaded_at": None, "last_etl": None}   # admin-visible runtime state

def db_conn():
    import pymysql
    return pymysql.connect(host=os.getenv("DB_HOST", "localhost"), user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""), database=os.getenv("DB_NAME", "air_quality_dw"), connect_timeout=3)

STAR_QUERY = """SELECT d.full_date AS date, d.year, d.month, d.month_name, d.quarter, l.city, l.state,
 l.latitude, l.longitude, p.pollutant_name AS dominant, f.temperature, f.humidity, f.pm25, f.pm10, f.co,
 f.no2, f.so2, f.o3, f.aqi FROM fact_air_quality f JOIN dim_date d ON f.date_id=d.date_id
 JOIN dim_location l ON f.location_id=l.location_id JOIN dim_pollutant p ON f.pollutant_id=p.pollutant_id"""

COMMON_COLS = ["date", "city", "state", "latitude", "longitude"] + FEATURES + ["aqi"]

def norm(v):
    """Single normalisation rule used everywhere a city is compared: trim, collapse spaces, casefold."""
    return " ".join(str(v).split()).casefold()

def prepare(df):
    """Makes MySQL and CSV data identical: same columns, dtypes, city spelling, date type and row order."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in COMMON_COLS if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing columns: " + ", ".join(missing))
    df = df[COMMON_COLS]
    df["city"] = df["city"].astype(str).map(lambda v: " ".join(v.split()))
    df["city_key"] = df["city"].map(norm)
    canon = df.groupby("city_key")["city"].agg(lambda s: s.mode().iat[0])   # one display spelling per city
    df["city"] = df["city_key"].map(canon)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in ["latitude", "longitude", "aqi"] + FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "aqi"]).sort_values(["date", "city"]).reset_index(drop=True)
    df["year"], df["month"], df["quarter"] = df.date.dt.year, df.date.dt.month, df.date.dt.quarter
    df["month_name"] = df.date.dt.strftime("%B")
    df["category"] = category(df.aqi).astype(str)
    return df

def read_source():
    """Returns (raw dataframe, source label). MySQL first, CSV fallback; FORCE_CSV=1 skips MySQL."""
    _state["fallback_reason"] = None
    if _state["source_mode"] == "csv" or os.getenv("FORCE_CSV") == "1":
        _state["fallback_reason"] = "CSV forced " + ("by admin" if _state["source_mode"] == "csv" else "by FORCE_CSV=1")
    else:
        try:
            conn = db_conn()
            try:
                raw = pd.read_sql(STAR_QUERY, conn)
            finally:
                conn.close()
            return raw, "MySQL Data Warehouse"
        except Exception as e:
            _state["fallback_reason"] = f"MySQL unavailable ({type(e).__name__}: {str(e)[:120]})"
            log.warning("MySQL unavailable (%s); using CSV fallback", type(e).__name__)
    return pd.read_csv(os.path.join("data", "air_quality.csv")), "CSV Fallback"

def load_df(force=False):
    if "df" in _cache and not force:
        return _cache["df"]
    raw, source = read_source()
    if raw.empty:
        raise ValueError("The warehouse has no records. Run database.sql or upload a CSV on the ETL page.")
    df = prepare(raw)
    _cache.update(df=df, source=source)
    _state["loaded_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("Loaded %d rows from %s | cities=%s | %s to %s", len(df), source, df.city.nunique(),
             df.date.min().date(), df.date.max().date())
    return df

def resolve_city(name):
    """Maps whatever the client sent (any case, stray spaces, encoded) to the normalised city key."""
    df = load_df()
    key = norm(name)
    if key not in set(df.city_key):
        raise ValueError(f"City '{str(name).strip()}' is not in the dataset. Available: {', '.join(sorted(df.city.unique()))}.")
    return key

def parse_date(value, label):
    if not value:
        return None
    d = pd.to_datetime(value, errors="coerce")
    if pd.isna(d):
        raise ValueError(f"{label} date '{value}' is not valid. Use YYYY-MM-DD.")
    return d.normalize()

def measure():
    m = request.args.get("measure", "aqi").strip().lower() or "aqi"
    if m not in MEASURES: raise ValueError(f"Unknown measure '{m}'. Choose one of: {', '.join(MEASURES)}.")
    return m

def filtered():
    df = load_df()
    city, cat = request.args.get("city", "").strip(), request.args.get("cat", "").strip()
    start, end = parse_date(request.args.get("start"), "Start"), parse_date(request.args.get("end"), "End")
    if start is not None and end is not None and start > end:
        raise ValueError("The start date is after the end date.")
    if city: df = df[df.city_key == resolve_city(city)]
    if cat: df = df[df.category.map(norm) == norm(cat)]
    if start is not None: df = df[df.date >= start]
    if end is not None: df = df[df.date <= end]
    log.info("filter city=%r cat=%r start=%s end=%s -> %d rows (%s)", city, cat, start, end, len(df), _cache.get("source"))
    return df

def empty_message():
    bits = [request.args.get(k) for k in ("city", "cat")] + [request.args.get("start"), request.args.get("end")]
    return "No records match these filters (" + ", ".join(b for b in bits if b) + "). Try widening the date range or choosing All Cities."

@app.route("/api/filters")
def filters():
    if "user" not in session: return jsonify(error="Please log in first."), 401
    try:
        df = load_df()
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(cities=sorted(df.city.unique()), min_date=str(df.date.min().date()), max_date=str(df.date.max().date()),
                   categories=CATS, source=_cache["source"], records=int(len(df)))

def api(fn, admin=False):
    def wrap(*a, **k):
        if "user" not in session:
            return jsonify(error="Please log in first."), 401
        if admin and session.get("role") != "admin":
            return jsonify(error="Admin access required."), 403
        try:
            return fn(*a, **k)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except Exception:
            app.logger.exception("API failure")
            return jsonify(error="Something went wrong while processing this request."), 500
    wrap.__name__ = fn.__name__
    return wrap

def admin_api(fn):
    return api(fn, admin=True)

def do_login(role, name, pw_hash, template, home):
    if session.get("role") == role:
        return redirect(url_for(home))
    if request.method == "POST":
        u, p = request.form.get("username", "").strip(), request.form.get("password", "")
        if u == name and check_password_hash(pw_hash, p):
            session.clear(); session["user"] = u; session["role"] = role
            return redirect(url_for("index"))
        return render_template(template, error="Wrong username or password.")
    return render_template(template)

@app.route("/login", methods=["GET", "POST"])
def login():
    return do_login("user", USER_NAME, USER_HASH, "login.html", "index")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    return do_login("admin", ADMIN_USER, ADMIN_HASH, "admin_login.html", "index")

@app.route("/logout")
def logout():
    role = session.get("role"); session.clear()
    return redirect(url_for("admin_login" if role == "admin" else "login"))

@app.route("/")
def index():
    return render_template("index.html", role=session["role"]) if "user" in session else redirect(url_for("login"))

@app.route("/api/summary")
@api
def summary():
    df = filtered()
    if df.empty: raise ValueError(empty_message())
    last = df[df.date == df.date.max()]
    m = last[["aqi"] + FEATURES].mean().round(1).to_dict()
    m["category"] = str(category(np.array([m["aqi"]]))[0]); m["date"] = str(df.date.max().date())
    m["source"] = _cache["source"]; m["records"] = int(len(df))
    return jsonify(m)

@app.route("/api/cities")
@api
def cities():
    m = measure(); df = load_df(); last = df[df.date == df.date.max()]
    out = last.groupby(["city", "latitude", "longitude"])[["aqi", "pm25", "pm10", "no2", "so2", "o3", "co", "temperature", "humidity"]].mean().round(1).reset_index()
    out["category"] = category(out.aqi).astype(str)
    out["value"] = out[m]
    out["share"] = ((out[m] - out[m].min()) / (out[m].max() - out[m].min())).fillna(0).round(3) if len(out) > 1 else 0.0   # 0..1 position for non-AQI map colours
    return jsonify(out.to_dict("records"))

@app.route("/api/charts")
@api
def charts():
    m = measure(); df = filtered()
    if df.empty: raise ValueError(empty_message())
    t = df.groupby("date")[m].mean().round(2)
    ranged = bool(request.args.get("start") or request.args.get("end"))
    if not ranged: t = t.tail(90)   # default view: most recent 90 days; an explicit date range is shown in full
    trend_label = f"{t.index.min().date()} to {t.index.max().date()}" if ranged else "last 90 days"
    mo = df.groupby(["year", "month"])[m].mean().round(2)
    cat = df.category.value_counts().reindex(CATS, fill_value=0)
    city = load_df().groupby("city")[m].mean().round(2).sort_values(ascending=False)
    return jsonify(measure=MEASURES[m], trend={"label": trend_label, "x": [str(d.date()) for d in t.index], "y": t.tolist()},
        pollutants={"x": ["PM2.5", "PM10", "CO", "NO2", "SO2", "O3"], "y": df[["pm25","pm10","co","no2","so2","o3"]].mean().round(1).tolist()},
        categories={"x": CATS, "y": cat.tolist()}, cities={"x": city.index.tolist(), "y": city.tolist()},
        monthly={"x": [f"{MONTHS[m-1]} {y}" for y, m in mo.index], "y": mo.tolist()})

@app.route("/api/olap/<op>")
@api
def olap(op):
    df = load_df(); metric = request.args.get("metric", "aqi")
    if metric not in ["aqi"] + FEATURES[:6]: raise ValueError("Unknown measure.")
    if op == "rollup":
        lvl = request.args.get("level", "month")
        keys = {"day": ["date"], "month": ["year", "month"], "year": ["year"]}.get(lvl)
        if not keys: raise ValueError("Level must be day, month or year.")
        r = df.groupby(keys)[metric].mean().round(1).reset_index()
        if lvl == "day": r = r.tail(60)
    elif op == "drilldown":
        y, mth = request.args.get("year", type=int), request.args.get("month", type=int)
        if y and mth: r = df[(df.year == y) & (df.month == mth)].groupby("date")[metric].mean().round(1).reset_index()
        elif y: r = df[df.year == y].groupby("month")[metric].mean().round(1).reset_index()
        else: r = df.groupby("year")[metric].mean().round(1).reset_index()
    elif op == "slice":
        c = request.args.get("city", "")
        if not c.strip(): raise ValueError("Choose a city to slice on.")
        r = df[df.city_key == resolve_city(c)].groupby(["year", "month"])[metric].mean().round(1).reset_index()
    elif op == "dice":
        cs = [resolve_city(c) for c in request.args.get("cities", "").split(",") if c.strip()]
        y = request.args.get("year", type=int)
        if not cs or not y: raise ValueError("Pick at least one city and a year.")
        r = df[df.city_key.isin(cs) & (df.year == y)].groupby(["city", "month"])[metric].mean().round(1).reset_index()
        if r.empty: raise ValueError("No data for that combination.")
    else:
        raise ValueError("Unknown OLAP operation.")
    r["date"] = r["date"].astype(str) if "date" in r else None
    r = r.drop(columns=[c for c in r if r[c].isna().all()])
    return jsonify(columns=list(r.columns), rows=r.values.tolist(), measure=metric)

@app.route("/api/mining/clustering")
@api
def clustering():
    k = request.args.get("k", 3, type=int)
    cols = ["aqi", "pm25", "pm10", "no2", "so2", "co"]
    g = load_df().groupby("city")[cols].mean()
    if not 2 <= k <= len(g): raise ValueError(f"K must be between 2 and {len(g)}.")
    km = KMeans(n_clusters=k, n_init=10, random_state=1).fit(StandardScaler().fit_transform(g))
    g["raw"] = km.labels_
    order = g.groupby("raw").aqi.mean().sort_values().index.tolist()
    g["cluster"] = g.raw.map({o: i for i, o in enumerate(order)})
    names = ["Cleaner air", "Moderate air", "Poor air", "Very poor air", "Severe air"]
    label = lambda i: names[round(i * 4 / max(k - 1, 1))]
    return jsonify(k=k, points=[{"city": c, "cluster": int(r.cluster), "aqi": round(r.aqi, 1), "pm25": round(r.pm25, 1)} for c, r in g.iterrows()],
        clusters=[{"id": i, "label": label(i), "avg_aqi": round(g[g.cluster == i].aqi.mean(), 1), "cities": g[g.cluster == i].index.tolist()} for i in range(k)])

def models():
    if "m" not in _cache:
        df = load_df()
        X, y = df[FEATURES], df.category
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
        tree = DecisionTreeClassifier(max_depth=4, random_state=7).fit(Xtr, ytr)
        reg = LinearRegression().fit(*train_test_split(X, df.aqi, test_size=0.25, random_state=7)[::2])
        _cache["m"] = (tree, reg, Xte, yte, df)
    return _cache["m"]

@app.route("/api/mining/classification")
@api
def classification():
    tree, _, Xte, yte, _ = models(); pred = tree.predict(Xte)
    labels = [c for c in CATS if c in set(yte) | set(pred)]
    p, r, f, _ = precision_recall_fscore_support(yte, pred, average="weighted", zero_division=0)
    return jsonify(accuracy=round(accuracy_score(yte, pred), 3), precision=round(p, 3), recall=round(r, 3), f1=round(f, 3),
        labels=labels, matrix=confusion_matrix(yte, pred, labels=labels).tolist(),
        tree=export_text(tree, feature_names=FEATURES, decimals=1),
        importance=sorted(zip(FEATURES, tree.feature_importances_.round(3).tolist()), key=lambda t: -t[1])[:5])

REG_MODELS = {
    "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge Regression": lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso Regression": lambda: make_pipeline(StandardScaler(), Lasso(alpha=0.05, max_iter=5000)),
    "Polynomial (degree 2)": lambda: make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1.0)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=80, max_depth=10, random_state=7, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=7),
}
MAX_FEATURES, PARSIMONY = 4, 1.02   # a smaller feature set wins if its CV error is within 2% of the best set

def regression_study():
    """Ranks features, then for every model picks the smallest feature set that is nearly as accurate as the best
    (5-fold CV on the training split) and scores it on a held-out test split. The best model by CV error wins."""
    if "reg" in _cache:
        return _cache["reg"]
    df = load_df()
    if len(df) < 50: raise ValueError("At least 50 records are needed to compare regression models.")
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df.aqi, test_size=0.25, random_state=7)
    mi = mutual_info_regression(Xtr, ytr, random_state=7)
    ranked = [f for f, _ in sorted(zip(FEATURES, mi), key=lambda t: -t[1])]
    rows = []
    for name, make in REG_MODELS.items():
        best = None
        for k in range(1, MAX_FEATURES + 1):
            cols = ranked[:k]
            rmse = -cross_val_score(make(), Xtr[cols], ytr, cv=5, scoring="neg_root_mean_squared_error").mean()
            if best is None or rmse < best[1] / PARSIMONY:   # only replace when clearly better than the smaller set
                best = (cols, rmse)
        cols, cv_rmse = best
        model = make().fit(Xtr[cols], ytr); pred = model.predict(Xte[cols])
        rows.append(dict(model=name, features=cols, cv_rmse=round(cv_rmse, 2), r2=round(r2_score(yte, pred), 4),
                         rmse=round(mean_squared_error(yte, pred) ** 0.5, 2), mae=round(mean_absolute_error(yte, pred), 2), _fit=model))
    rows.sort(key=lambda r: r["cv_rmse"])
    winner = rows[0]
    study = dict(rows=rows, winner=winner, ranking=[(f, round(float(m), 3)) for f, m in sorted(zip(FEATURES, mi), key=lambda t: -t[1])],
                 defaults={f: round(float(df[f].median()), 1) for f in FEATURES}, test_size=len(Xte), train_size=len(Xtr))
    _cache["reg"] = study
    log.info("Regression study: winner=%s features=%s r2=%s", winner["model"], winner["features"], winner["r2"])
    return study

@app.route("/api/mining/regression")
@api
def regression():
    st = regression_study(); w = st["winner"]
    return jsonify(models=[{k: v for k, v in r.items() if k != "_fit"} for r in st["rows"]], best=w["model"], features=w["features"],
                   defaults={f: st["defaults"][f] for f in w["features"]}, ranking=st["ranking"],
                   train_size=st["train_size"], test_size=st["test_size"])

@app.route("/api/mining/prediction", methods=["POST"])
@api
def prediction():
    w = regression_study()["winner"]; data = request.get_json(silent=True) or {}
    try:
        x = [float(data[f]) for f in w["features"]]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Enter a number for every field.")
    if any(v < 0 for v in x) or any(v > 1500 for v in x): raise ValueError("Values are outside a realistic range.")
    aqi = int(np.clip(round(w["_fit"].predict(pd.DataFrame([x], columns=w["features"]))[0]), 0, 500))
    return jsonify(aqi=aqi, category=str(category(np.array([aqi]))[0]), r2=w["r2"], model=w["model"])

@app.route("/api/etl/upload", methods=["POST"])
@admin_api
def etl_upload():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".csv"): raise ValueError("Upload a .csv file.")
    try: raw = pd.read_csv(io.BytesIO(f.read()))
    except Exception: raise ValueError("That file could not be read as CSV.")
    raw.columns = [c.strip().lower() for c in raw.columns]
    need = ["date", "city"] + FEATURES
    miss = [c for c in need if c not in raw.columns]
    if miss: raise ValueError("Missing required columns: " + ", ".join(miss))
    st = {"extracted": len(raw)}
    df = raw.copy(); df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city"])
    df["city"] = df["city"].astype(str).map(lambda v: " ".join(v.split()))
    for c in FEATURES: df[c] = pd.to_numeric(df[c], errors="coerce")
    st["missing_handled"] = int(df[FEATURES].isna().sum().sum())
    df[FEATURES] = df.groupby("city")[FEATURES].transform(lambda s: s.fillna(s.median())).fillna(df[FEATURES].median())
    before = len(df); df = df.drop_duplicates(subset=["date", "city"]); st["duplicates_removed"] = before - len(df)
    df["aqi"] = compute_aqi(df.pm25, df.pm10) if "aqi" not in df or df.aqi.isna().any() else df.aqi.astype(int)
    st["cleaned"] = st["transformed"] = len(df)
    st["loaded"] = 0; st["target"] = "Validated only (MySQL not reachable)"
    try:
        conn = db_conn(); cur = conn.cursor()
        known = load_df()[["city", "state", "latitude", "longitude"]].drop_duplicates("city").set_index("city").to_dict("index")
        for city, g in df.groupby("city"):
            info = known.get(city, {"state": g.get("state", pd.Series(["Unknown"])).iloc[0], "latitude": 0, "longitude": 0})
            cur.execute("INSERT IGNORE INTO dim_location (city,state,latitude,longitude) VALUES (%s,%s,%s,%s)", (city, info["state"], info["latitude"], info["longitude"]))
            cur.execute("SELECT location_id FROM dim_location WHERE city=%s", (city,)); lid = cur.fetchone()[0]
            for r in g.itertuples():
                cur.execute("INSERT IGNORE INTO dim_date VALUES (%s,%s,%s,%s,%s,%s,%s)", (int(r.date.strftime("%Y%m%d")), r.date.date(), r.date.day, r.date.month, r.date.strftime("%B"), (r.date.month-1)//3+1, r.date.year))
                cur.execute("INSERT IGNORE INTO fact_air_quality (date_id,location_id,pollutant_id,temperature,humidity,pm25,pm10,co,no2,so2,o3,aqi) VALUES (%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (int(r.date.strftime("%Y%m%d")), lid, r.temperature, r.humidity, r.pm25, r.pm10, r.co, r.no2, r.so2, r.o3, int(r.aqi)))
                st["loaded"] += cur.rowcount
        conn.commit(); conn.close(); st["target"] = "MySQL data warehouse"
        _cache.clear()
    except Exception:
        pass
    _state["last_etl"] = {**st, "file": f.filename, "at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "by": session["user"]}
    log.info("ETL upload by %s: %s", session["user"], st)
    return jsonify(st)

def status_payload():
    df = load_df()
    return jsonify(source=_cache["source"], mode=_state["source_mode"], records=int(len(df)), cities=int(df.city.nunique()),
                   start=str(df.date.min().date()), end=str(df.date.max().date()), loaded_at=_state["loaded_at"],
                   fallback_reason=_state["fallback_reason"], last_etl=_state["last_etl"])

@app.route("/api/admin/status")
@admin_api
def admin_status():
    return status_payload()

@app.route("/api/admin/reload", methods=["POST"])
@admin_api
def admin_reload():
    mode = (request.get_json(silent=True) or {}).get("mode", "auto")
    if mode not in ("auto", "csv"): raise ValueError("Mode must be 'auto' or 'csv'.")
    old = dict(_state)
    _state["source_mode"] = mode; _cache.clear()
    try:
        load_df()
    except Exception:
        _state.update(source_mode=old["source_mode"]); _cache.clear()   # keep the previous mode if the new one cannot load
        raise
    log.info("Admin %s reloaded data: mode=%s source=%s", session["user"], mode, _cache["source"])
    return status_payload()

if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")
