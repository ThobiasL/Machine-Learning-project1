import netCDF4 as nc4
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
 
# ── Paths ─────────────────────────────────────────────────────────────────────
WEATHER_DIR  = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\weather"
HOURLY_NC    = rf"{WEATHER_DIR}\weather_hourly.nc"
ACCUM_NC     = rf"{WEATHER_DIR}\weather_accum.nc"
FLIGHT_PATH  = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20220301_20220331.csv"
 
WEATHER_VARS = [
    "wind_speed", "temperature", "dewpoint", "pressure_hpa",
    "low_cloud", "cape", "precip_mm", "snowfall_mm",
    "wind_gust", "cloud_base_m",
]
 
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_var(nc, *candidates):
    """Return the first variable name that exists in nc, or None."""
    for name in candidates:
        if name in nc.variables:
            return name
    return None
 
 
def _nc_times(nc):
    """Return a DatetimeIndex of the time coordinate, floored to the hour."""
    tname = _find_var(nc, "time", "valid_time")
    if tname is None:
        raise RuntimeError("No time variable found in NetCDF")
    tv = nc.variables[tname]
    cal = getattr(tv, "calendar", "standard")
    dts = nc4.num2date(tv[:], units=tv.units, calendar=cal)
    return pd.DatetimeIndex([d.isoformat() for d in dts]).floor("h")
 
 
def _nc_coords(nc):
    """Return (lat_1d, lon_1d) as float arrays, lon in -180/180."""
    lat = nc.variables[_find_var(nc, "latitude",  "lat")][:]
    lon = nc.variables[_find_var(nc, "longitude", "lon")][:]
    # Normalise 0-360 → -180/180
    lon = ((lon + 180) % 360) - 180
    return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
 
 
def _read_point(nc, varname, lat_name, lon_name, lat_idx, lon_idx):
    """
    Extract the full time series for one grid point.
    Works for any number of dimensions by locating the lat/lon axes
    and integer-indexing them — only the time-series is loaded into RAM.
    """
    var  = nc.variables[varname]
    dims = list(var.dimensions)
 
    # Find which axis is lat and which is lon
    lat_ax = dims.index(lat_name)
    lon_ax = dims.index(lon_name)
 
    # Build an index tuple: slice everything except lat/lon
    idx = [slice(None)] * len(dims)
    idx[lat_ax] = lat_idx
    idx[lon_ax] = lon_idx
 
    data = np.asarray(var[tuple(idx)], dtype=np.float32)
    return data.squeeze()   # remove any leftover size-1 dims (number, step …)
 
 
# ── 1. Open NetCDF files (metadata only — no data loaded yet) ─────────────────
print("Loading weather NetCDF files …")
nc_h = nc4.Dataset(HOURLY_NC,  "r")
nc_a = nc4.Dataset(ACCUM_NC,   "r")
 
h_lat, h_lon = _nc_coords(nc_h)
a_lat, a_lon = _nc_coords(nc_a)
h_times = _nc_times(nc_h)
a_times = _nc_times(nc_a)
 
# Find the actual lat/lon dimension names (may differ by file)
h_lat_name = _find_var(nc_h, "latitude", "lat")
h_lon_name = _find_var(nc_h, "longitude", "lon")
a_lat_name = _find_var(nc_a, "latitude", "lat")
a_lon_name = _find_var(nc_a, "longitude", "lon")
 
print(f"  Hourly: {len(h_times)} time steps, "
      f"lat {h_lat.min():.1f}→{h_lat.max():.1f}, "
      f"lon {h_lon.min():.1f}→{h_lon.max():.1f}")
print(f"  Accum:  {len(a_times)} time steps, "
      f"lat {a_lat.min():.1f}→{a_lat.max():.1f}, "
      f"lon {a_lon.min():.1f}→{a_lon.max():.1f}")
 
# Check which variables actually exist in each file
def _has(nc, name):
    return name in nc.variables
 
print(f"  Hourly vars: {[v for v in nc_h.variables if v not in ('time','valid_time','latitude','longitude','lat','lon')]}")
print(f"  Accum  vars: {[v for v in nc_a.variables if v not in ('time','valid_time','latitude','longitude','lat','lon')]}")
 
 
# ── 2. Load flights ───────────────────────────────────────────────────────────
print("\nLoading flight data …")
flights = pd.read_csv(FLIGHT_PATH, low_memory=False)
 
DATE_COLS = [
    "FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME",
    "FILED ARRIVAL TIME",   "ACTUAL ARRIVAL TIME",
]
for col in DATE_COLS:
    if col in flights.columns:
        flights[col] = pd.to_datetime(flights[col], dayfirst=True, errors="coerce")
 
flights["DEP_DELAY_MIN"] = (
    flights["ACTUAL OFF BLOCK TIME"] - flights["FILED OFF BLOCK TIME"]
).dt.total_seconds() / 60
 
flights["ARR_DELAY_MIN"] = (
    flights["ACTUAL ARRIVAL TIME"] - flights["FILED ARRIVAL TIME"]
).dt.total_seconds() / 60
 
req_cols = ["DEP_DELAY_MIN", "ARR_DELAY_MIN", "ADEP Latitude", "ADEP Longitude"]
flights = flights.dropna(subset=req_cols)
flights = flights[flights["DEP_DELAY_MIN"].between(-60, 600)]
flights = flights[flights["ARR_DELAY_MIN"].between(-60, 600)]
 
N_SAMPLE = 50_000
if len(flights) > N_SAMPLE:
    flights = flights.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)
 
flights["DEP_HOUR"] = flights["ACTUAL OFF BLOCK TIME"].dt.floor("h")
 
print(f"  Rows used: {len(flights):,}")
print(f"  Avg departure delay: {flights['DEP_DELAY_MIN'].mean():.1f} min")
 
 
# ── 3. Per-airport weather extraction (lat-strip batching — fast) ─────────────
# Instead of 1031 × 11 individual point reads, we group airports by their
# nearest latitude index and read one (T × Nlon) strip per unique lat per
# variable (~9 MB each).  All lon lookups then happen from RAM.
print("\nExtracting weather per airport …")
unique_airports = (
    flights[["ADEP", "ADEP Latitude", "ADEP Longitude"]]
    .drop_duplicates("ADEP")
    .reset_index(drop=True)
)
print(f"  Unique airports: {len(unique_airports)}")
 
# Precompute nearest grid indices for every airport
h_li, h_lj, a_li, a_lj = {}, {}, {}, {}
for _, row in unique_airports.iterrows():
    adep = row["ADEP"]
    lat  = float(row["ADEP Latitude"])
    lon  = float(row["ADEP Longitude"])
    h_li[adep] = int(np.argmin(np.abs(h_lat - lat)))
    h_lj[adep] = int(np.argmin(np.abs(h_lon - lon)))
    a_li[adep] = int(np.argmin(np.abs(a_lat - lat)))
    a_lj[adep] = int(np.argmin(np.abs(a_lon - lon)))
 
# Group airports by latitude index
from collections import defaultdict
h_by_lat = defaultdict(list)
a_by_lat = defaultdict(list)
for adep in unique_airports["ADEP"]:
    h_by_lat[h_li[adep]].append(adep)
    a_by_lat[a_li[adep]].append(adep)
 
def read_lat_strip(nc, varname, lat_name, lat_idx):
    """
    Read all times and longitudes for one latitude row.
    Returns shape (T, Nlon) regardless of extra degenerate dims in the file.
    """
    var  = nc.variables[varname]
    dims = list(var.dimensions)
    idx  = [slice(None)] * len(dims)
    idx[dims.index(lat_name)] = lat_idx
    data = np.asarray(var[tuple(idx)], dtype=np.float32).squeeze()
    if data.ndim == 1:          # edge case: single-lon file
        data = data[:, np.newaxis]
    return data                 # (T, Nlon)
 
H_VARS = ["u10", "v10", "t2m", "d2m", "msl", "lcc", "cape"]
A_VARS = ["tp",  "sf",  "fg10", "cbh"]
 
# Read raw time-series per airport, one lat-strip at a time
raw_h = {adep: {} for adep in unique_airports["ADEP"]}
raw_a = {adep: {} for adep in unique_airports["ADEP"]}
 
total = len(h_by_lat) * len(H_VARS) + len(a_by_lat) * len(A_VARS)
done  = 0
 
print(f"  Reading {len(h_by_lat)} unique lat-strips × {len(H_VARS)} hourly vars …")
for varname in H_VARS:
    if varname not in nc_h.variables:
        print(f"  WARNING: {varname} missing from hourly NC, skipping"); continue
    for lat_i, airports in h_by_lat.items():
        strip = read_lat_strip(nc_h, varname, h_lat_name, lat_i)
        for adep in airports:
            raw_h[adep][varname] = strip[:, h_lj[adep]]
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total} strips …", flush=True)
 
print(f"  Reading {len(a_by_lat)} unique lat-strips × {len(A_VARS)} accum vars …")
for varname in A_VARS:
    if varname not in nc_a.variables:
        print(f"  WARNING: {varname} missing from accum NC, skipping"); continue
    for lat_i, airports in a_by_lat.items():
        strip = read_lat_strip(nc_a, varname, a_lat_name, lat_i)
        for adep in airports:
            raw_a[adep][varname] = strip[:, a_lj[adep]]
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total} strips …", flush=True)
 
nc_h.close()
nc_a.close()
 
# Assemble per-airport DataFrames from the cached arrays
airport_wx = {}
failed = 0
for adep in unique_airports["ADEP"]:
    try:
        u10  = raw_h[adep]["u10"];  v10 = raw_h[adep]["v10"]
        t2m  = raw_h[adep]["t2m"];  d2m = raw_h[adep]["d2m"]
        msl  = raw_h[adep]["msl"];  lcc = raw_h[adep]["lcc"]
        cape = raw_h[adep]["cape"]
 
        wh = pd.DataFrame({
            "wind_speed":   np.sqrt(u10**2 + v10**2),
            "temperature":  t2m - 273.15,
            "dewpoint":     d2m - 273.15,
            "pressure_hpa": msl / 100.0,
            "low_cloud":    lcc,
            "cape":         cape,
        }, index=h_times)
 
        wa = pd.DataFrame({
            "precip_mm":    raw_a[adep].get("tp",   np.zeros(len(a_times))) * 1000.0,
            "snowfall_mm":  raw_a[adep].get("sf",   np.zeros(len(a_times))) * 1000.0,
            "wind_gust":    raw_a[adep].get("fg10", np.zeros(len(a_times))),
            "cloud_base_m": raw_a[adep].get("cbh",  np.full(len(a_times), np.nan)),
        }, index=a_times)
 
        airport_wx[adep] = wh.join(wa, how="left")
 
    except Exception as e:
        failed += 1
        if failed <= 3:
            print(f"  FAILED {adep}: {type(e).__name__}: {e}")
 
print(f"  Done — weather for {len(airport_wx)} airports  ({failed} failed)")
 
if not airport_wx:
    raise RuntimeError(
        "No airport weather extracted.\n"
        "Check that the .nc files exist and cover the right lat/lon range."
    )
 
 
# ── 4. Join weather onto each flight ─────────────────────────────────────────
print("\nJoining weather onto flights …")
rows = []
for _, flight in flights.iterrows():
    adep = flight["ADEP"]
    t    = flight["DEP_HOUR"]
 
    rec = {
        "DEP_DELAY_MIN": flight["DEP_DELAY_MIN"],
        "ARR_DELAY_MIN": flight["ARR_DELAY_MIN"],
    }
 
    if adep in airport_wx:
        wx  = airport_wx[adep]
        idx = wx.index.get_indexer([t], method="nearest")[0]
        if 0 <= idx < len(wx):
            rec.update(wx.iloc[idx].to_dict())
 
    rows.append(rec)
 
df = pd.DataFrame(rows)
df = df.dropna(subset=["wind_speed"])
 
print(f"  Flights with weather data: {len(df):,}")
 
 
# ── 5. Correlation analysis ───────────────────────────────────────────────────
print("\nComputing Pearson correlations …")
available_vars = [v for v in WEATHER_VARS if v in df.columns]
 
corr_dep = (
    df[available_vars + ["DEP_DELAY_MIN"]]
    .corr()["DEP_DELAY_MIN"]
    .drop("DEP_DELAY_MIN")
)
corr_arr = (
    df[available_vars + ["ARR_DELAY_MIN"]]
    .corr()["ARR_DELAY_MIN"]
    .drop("ARR_DELAY_MIN")
)
 
print("\nPearson r — Departure delay:")
print(corr_dep.sort_values(key=abs, ascending=False).to_string())
print("\nPearson r — Arrival delay:")
print(corr_arr.sort_values(key=abs, ascending=False).to_string())
 
 
# ── 6. Full results ───────────────────────────────────────────────────────────
corr_matrix = df[available_vars + ["DEP_DELAY_MIN", "ARR_DELAY_MIN"]].corr()
 
print("\n" + "="*60)
print("CORRELATION WITH DEPARTURE DELAY  (Pearson r)")
print("="*60)
for var, r in corr_dep.sort_values(key=abs, ascending=False).items():
    bar = "█" * int(abs(r) * 40)
    sign = "+" if r >= 0 else "-"
    print(f"  {var:<18}  {sign}{abs(r):.4f}  {bar}")
 
print("\n" + "="*60)
print("CORRELATION WITH ARRIVAL DELAY  (Pearson r)")
print("="*60)
for var, r in corr_arr.sort_values(key=abs, ascending=False).items():
    bar = "█" * int(abs(r) * 40)
    sign = "+" if r >= 0 else "-"
    print(f"  {var:<18}  {sign}{abs(r):.4f}  {bar}")
 
print("\n" + "="*60)
print("FULL CORRELATION MATRIX")
print("="*60)
print(corr_matrix.round(3).to_string())
 
print("\n" + "="*60)
print("TOP 5 CORRELATES WITH DEPARTURE DELAY")
print("="*60)
top = corr_dep.sort_values(key=abs, ascending=False).head(5)
for var, r in top.items():
    direction = "→ longer delay" if r > 0 else "→ shorter delay"
    print(f"  {var:<18}  r = {r:+.4f}  {direction}")
print("="*60)
print("\nDone.")