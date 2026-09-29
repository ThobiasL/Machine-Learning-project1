"""
build_new_dataset.py

Merges flight CSV files with ERA5 weather at both departure (ADEP) and arrival (ADES).
"""

import netCDF4 as nc4
import numpy as np
import pandas as pd
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

WEATHER_DIR = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\weather"
HOURLY_NC   = rf"{WEATHER_DIR}\weather_hourly.nc"
ACCUM_NC    = rf"{WEATHER_DIR}\weather_accum.nc"

FLIGHT_FILES = [
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20220301_20220331.csv",
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20220601_20220630.csv",
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20220901_20220930.csv",
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20221201_20221231.csv",
]

OUTPUT_PATH = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights_weather_dataset.csv"


def _find_var(nc, *candidates):
    for name in candidates:
        if name in nc.variables:
            return name
    return None

def _nc_times(nc):
    tname = _find_var(nc, "time", "valid_time")
    tv    = nc.variables[tname]
    cal   = getattr(tv, "calendar", "standard")
    dts   = nc4.num2date(tv[:], units=tv.units, calendar=cal)
    return pd.DatetimeIndex([d.isoformat() for d in dts]).floor("h")

def _nc_coords(nc):
    lat = nc.variables[_find_var(nc, "latitude", "lat")][:]
    lon = nc.variables[_find_var(nc, "longitude", "lon")][:]
    lon = ((lon + 180) % 360) - 180
    return np.asarray(lat, float), np.asarray(lon, float)

def read_lat_strip(nc, varname, lat_name, lat_idx):
    """Read all times × all lons for one latitude row → (T, Nlon)."""
    var  = nc.variables[varname]
    dims = list(var.dimensions)
    idx  = [slice(None)] * len(dims)
    idx[dims.index(lat_name)] = lat_idx
    data = np.asarray(var[tuple(idx)], dtype=np.float32).squeeze()
    if data.ndim == 1:
        data = data[:, np.newaxis]
    return data



print("Loading flight files …")
frames = []
for path in FLIGHT_FILES:
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"  {path.split(chr(92))[-1]:50s}  {len(df):>8,} rows")
        frames.append(df)
    except FileNotFoundError:
        print(f"  NOT FOUND (skipped): {path}")

if not frames:
    raise RuntimeError("No flight files found — check the paths in CONFIG.")

flights = pd.concat(frames, ignore_index=True)
print(f"\n  Total rows after concat: {len(flights):,}")

# Parse date columns
DATE_COLS = [
    "FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME",
    "FILED ARRIVAL TIME",   "ACTUAL ARRIVAL TIME",
]
for col in DATE_COLS:
    if col in flights.columns:
        flights[col] = pd.to_datetime(flights[col], dayfirst=True, errors="coerce")

# Delay columns
if "ACTUAL OFF BLOCK TIME" in flights.columns and "FILED OFF BLOCK TIME" in flights.columns:
    flights["DEP_DELAY_MIN"] = (
        flights["ACTUAL OFF BLOCK TIME"] - flights["FILED OFF BLOCK TIME"]
    ).dt.total_seconds() / 60

if "ACTUAL ARRIVAL TIME" in flights.columns and "FILED ARRIVAL TIME" in flights.columns:
    flights["ARR_DELAY_MIN"] = (
        flights["ACTUAL ARRIVAL TIME"] - flights["FILED ARRIVAL TIME"]
    ).dt.total_seconds() / 60

# Keep only rows where we can do a departure weather lookup
# (arrival weather is filled where available, NaN otherwise)
req = [c for c in ["ADEP", "ADEP Latitude", "ADEP Longitude", "ACTUAL OFF BLOCK TIME"]
       if c in flights.columns]
flights = flights.dropna(subset=req)

# Round times to the hour for weather lookup
flights["DEP_HOUR"] = flights["ACTUAL OFF BLOCK TIME"].dt.floor("h")
flights["ARR_HOUR"] = flights["ACTUAL ARRIVAL TIME"].dt.floor("h")

print(f"  Rows with valid departure info: {len(flights):,}")
print(f"  Rows with valid arrival info:   "
      f"{flights[['ADES', 'ADES Latitude', 'ADES Longitude', 'ARR_HOUR']].notna().all(axis=1).sum():,}")



print("\nOpening weather files …")
nc_h = nc4.Dataset(HOURLY_NC, "r")
nc_a = nc4.Dataset(ACCUM_NC,  "r")

h_lat, h_lon = _nc_coords(nc_h)
a_lat, a_lon = _nc_coords(nc_a)
h_times      = _nc_times(nc_h)
a_times      = _nc_times(nc_a)

h_lat_name = _find_var(nc_h, "latitude", "lat")
a_lat_name = _find_var(nc_a, "latitude", "lat")

print(f"  Hourly: {len(h_times)} steps  |  lat {h_lat.min():.0f}→{h_lat.max():.0f}  "
      f"lon {h_lon.min():.0f}→{h_lon.max():.0f}")
print(f"  Accum : {len(a_times)} steps  |  lat {a_lat.min():.0f}→{a_lat.max():.0f}  "
      f"lon {a_lon.min():.0f}→{a_lon.max():.0f}")



print("\nExtracting weather per airport …")
# Airport table: union of departure and arrival airports
AP_COLS = ["ICAO", "LAT", "LON"]
dep_ap = flights[["ADEP", "ADEP Latitude", "ADEP Longitude"]].set_axis(AP_COLS, axis=1)
arr_ap = flights[["ADES", "ADES Latitude", "ADES Longitude"]].set_axis(AP_COLS, axis=1)
unique_airports = (
    pd.concat([dep_ap, arr_ap], ignore_index=True)
    .dropna()
    .drop_duplicates("ICAO")
    .reset_index(drop=True)
)
print(f"  Unique airports (dep + arr): {len(unique_airports)}")

# Drop airports outside the weather grid, and every flight that uses one
in_grid = (
    unique_airports["LAT"].between(max(h_lat.min(), a_lat.min()), min(h_lat.max(), a_lat.max()))
    & unique_airports["LON"].between(max(h_lon.min(), a_lon.min()), min(h_lon.max(), a_lon.max()))
)
oob = int((~in_grid).sum())
unique_airports = unique_airports[in_grid].reset_index(drop=True)
grid_codes = set(unique_airports["ICAO"])
n_before = len(flights)
flights = flights[flights["ADEP"].isin(grid_codes) & flights["ADES"].isin(grid_codes)]
print(f"  {oob} airports outside the weather grid extent — removed")
print(f"  Flights removed (dep or arr outside grid / no coords): {n_before - len(flights):,}  "
      f"→ {len(flights):,} remain")

# Nearest grid indices for each airport
h_li, h_lj, a_li, a_lj = {}, {}, {}, {}
for _, row in unique_airports.iterrows():
    icao = row["ICAO"]
    lat  = float(row["LAT"])
    lon  = float(row["LON"])
    h_li[icao] = int(np.argmin(np.abs(h_lat - lat)))
    h_lj[icao] = int(np.argmin(np.abs(h_lon - lon)))
    a_li[icao] = int(np.argmin(np.abs(a_lat - lat)))
    a_lj[icao] = int(np.argmin(np.abs(a_lon - lon)))

# Group airports by latitude index to minimise file seeks
h_by_lat = defaultdict(list)
a_by_lat = defaultdict(list)
for icao in unique_airports["ICAO"]:
    h_by_lat[h_li[icao]].append(icao)
    a_by_lat[a_li[icao]].append(icao)

H_VARS = ["u10", "v10", "t2m", "d2m", "msl", "lcc", "cape"]
A_VARS = ["tp",  "sf",  "fg10", "cbh"]

raw_h = {icao: {} for icao in unique_airports["ICAO"]}
raw_a = {icao: {} for icao in unique_airports["ICAO"]}

total = len(h_by_lat) * len(H_VARS) + len(a_by_lat) * len(A_VARS)
done  = 0

print(f"  Reading {len(h_by_lat)} lat-strips × {len(H_VARS)} hourly vars …")
for varname in H_VARS:
    if varname not in nc_h.variables:
        print(f"    skip {varname} (not in file)"); continue
    for lat_i, airports in h_by_lat.items():
        strip = read_lat_strip(nc_h, varname, h_lat_name, lat_i)
        for icao in airports:
            raw_h[icao][varname] = strip[:, h_lj[icao]]
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total} …", flush=True)

print(f"  Reading {len(a_by_lat)} lat-strips × {len(A_VARS)} accum vars …")
for varname in A_VARS:
    if varname not in nc_a.variables:
        print(f"    skip {varname} (not in file)"); continue
    for lat_i, airports in a_by_lat.items():
        strip = read_lat_strip(nc_a, varname, a_lat_name, lat_i)
        for icao in airports:
            raw_a[icao][varname] = strip[:, a_lj[icao]]
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total} …", flush=True)

nc_h.close()
nc_a.close()

WEATHER_COLS = [
    "wind_speed_ms", "temperature_c", "dewpoint_c", "pressure_hpa",
    "low_cloud_frac", "cape_jkg", "precip_mm", "snowfall_mm",
    "wind_gust_ms", "cloud_base_m",
]

# Build per-airport lookup DataFrames (indexed by hour)
airport_wx = {}
for icao in unique_airports["ICAO"]:
    try:
        u10  = raw_h[icao].get("u10",  np.zeros(len(h_times)))
        v10  = raw_h[icao].get("v10",  np.zeros(len(h_times)))
        t2m  = raw_h[icao].get("t2m",  np.full(len(h_times), np.nan))
        d2m  = raw_h[icao].get("d2m",  np.full(len(h_times), np.nan))
        msl  = raw_h[icao].get("msl",  np.full(len(h_times), np.nan))
        lcc  = raw_h[icao].get("lcc",  np.full(len(h_times), np.nan))
        cape = raw_h[icao].get("cape", np.full(len(h_times), np.nan))

        wh = pd.DataFrame({
            "wind_speed_ms":   np.sqrt(u10**2 + v10**2),
            "temperature_c":   t2m - 273.15,
            "dewpoint_c":      d2m - 273.15,
            "pressure_hpa":    msl / 100.0,
            "low_cloud_frac":  lcc,
            "cape_jkg":        cape,
        }, index=h_times)

        wa = pd.DataFrame({
            "precip_mm":    raw_a[icao].get("tp",   np.zeros(len(a_times))) * 1000.0,
            "snowfall_mm":  raw_a[icao].get("sf",   np.zeros(len(a_times))) * 1000.0,
            "wind_gust_ms": raw_a[icao].get("fg10", np.zeros(len(a_times))),
            "cloud_base_m": raw_a[icao].get("cbh",  np.full(len(a_times), np.nan)),
        }, index=a_times)

        airport_wx[icao] = wh.join(wa, how="left")[WEATHER_COLS]
    except Exception as e:
        pass   # airport stays out of dict; flights get NaN weather

print(f"  Weather ready for {len(airport_wx)} / {len(unique_airports)} airports")



def lookup_weather(df, code_col, hour_col, prefix):
    """Nearest-hour weather for each row's airport → DataFrame of prefixed columns."""
    out = np.full((len(df), len(WEATHER_COLS)), np.nan)
    pos = pd.Series(np.arange(len(df)), index=df.index)
    valid = df[[code_col, hour_col]].dropna()
    for code, grp in valid.groupby(code_col):
        wx = airport_wx.get(code)
        if wx is None:
            continue
        idx = wx.index.get_indexer(grp[hour_col], method="nearest")
        ok  = idx >= 0
        out[pos[grp.index].to_numpy()[ok]] = wx.to_numpy()[idx[ok]]
    return pd.DataFrame(out, index=df.index,
                        columns=[f"{prefix}{c}" for c in WEATHER_COLS])


print("\nJoining weather onto flights …")
dep_wx = lookup_weather(flights, "ADEP", "DEP_HOUR", "dep_")
arr_wx = lookup_weather(flights, "ADES", "ARR_HOUR", "arr_")

result = pd.concat([flights.drop(columns=["DEP_HOUR", "ARR_HOUR"]), dep_wx, arr_wx], axis=1)

for prefix, label in [("dep_", "departure"), ("arr_", "arrival")]:
    n = result[f"{prefix}wind_speed_ms"].notna().sum()
    print(f"  Flights with {label} weather: {n:,} / {len(result):,} "
          f"({100*n/len(result):.1f} %)")



print(f"\nSaving to:\n  {OUTPUT_PATH}")
result.to_csv(OUTPUT_PATH, index=False)

size_mb = result.memory_usage(deep=True).sum() / 1024**2
print(f"\nDone.  Shape: {result.shape[0]:,} rows × {result.shape[1]} columns  "
      f"(~{size_mb:.0f} MB in memory)")
print(f"\nWeather columns added:")
for prefix in ["dep_", "arr_"]:
    for col in WEATHER_COLS:
        name = f"{prefix}{col}"
        present = name in result.columns and result[name].notna().any()
        print(f"  {'✓' if present else '✗'}  {name}")

print(f"\nAll done — load the merged file with:")
print(f"  df = pd.read_csv(r'{OUTPUT_PATH}')")