"""
build_new_dataset.py

Merges flight CSV files with ERA5 weather.
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
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20220101_20221231.csv",
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20210101_20211231.csv",
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20200101_20201231.csv",
    r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20190101_20191231.csv",
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

# Keep only rows where we can do a weather lookup
req = [c for c in ["ADEP", "ADEP Latitude", "ADEP Longitude", "ACTUAL OFF BLOCK TIME"]
       if c in flights.columns]
flights = flights.dropna(subset=req)

# Round departure time to nearest hour for weather lookup
flights["DEP_HOUR"] = flights["ACTUAL OFF BLOCK TIME"].dt.floor("h")

print(f"  Rows with valid departure info: {len(flights):,}")



print("\nOpening weather files …")
nc_h = nc4.Dataset(HOURLY_NC, "r")
nc_a = nc4.Dataset(ACCUM_NC,  "r")

h_lat, h_lon = _nc_coords(nc_h)
a_lat, a_lon = _nc_coords(nc_a)
h_times      = _nc_times(nc_h)
a_times      = _nc_times(nc_a)

h_lat_name = _find_var(nc_h, "latitude", "lat")
a_lat_name = _find_var(nc_a, "latitude", "lat")
h_lon_name = _find_var(nc_h, "longitude", "lon")   # used only in diagnostics

print(f"  Hourly: {len(h_times)} steps  |  lat {h_lat.min():.0f}→{h_lat.max():.0f}  "
      f"lon {h_lon.min():.0f}→{h_lon.max():.0f}")
print(f"  Accum : {len(a_times)} steps  |  lat {a_lat.min():.0f}→{a_lat.max():.0f}  "
      f"lon {a_lon.min():.0f}→{a_lon.max():.0f}")



print("\nExtracting weather per airport …")
unique_airports = (
    flights[["ADEP", "ADEP Latitude", "ADEP Longitude"]]
    .drop_duplicates("ADEP")
    .reset_index(drop=True)
)
print(f"  Unique airports: {len(unique_airports)}")

# Nearest grid indices for each airport
h_li, h_lj, a_li, a_lj = {}, {}, {}, {}
oob = 0
for _, row in unique_airports.iterrows():
    adep = row["ADEP"]
    lat  = float(row["ADEP Latitude"])
    lon  = float(row["ADEP Longitude"])
    # Warn if airport is outside the weather grid
    if not (h_lat.min() <= lat <= h_lat.max() and h_lon.min() <= lon <= h_lon.max()):
        oob += 1
    h_li[adep] = int(np.argmin(np.abs(h_lat - lat)))
    h_lj[adep] = int(np.argmin(np.abs(h_lon - lon)))
    a_li[adep] = int(np.argmin(np.abs(a_lat - lat)))
    a_lj[adep] = int(np.argmin(np.abs(a_lon - lon)))

if oob:
    print(f"  WARNING: {oob} airports outside the weather grid extent "
          f"(nearest point used — values may be inaccurate)")

# Group airports by latitude index to minimise file seeks
h_by_lat = defaultdict(list)
a_by_lat = defaultdict(list)
for adep in unique_airports["ADEP"]:
    h_by_lat[h_li[adep]].append(adep)
    a_by_lat[a_li[adep]].append(adep)

H_VARS = ["u10", "v10", "t2m", "d2m", "msl", "lcc", "cape"]
A_VARS = ["tp",  "sf",  "fg10", "cbh"]

raw_h = {adep: {} for adep in unique_airports["ADEP"]}
raw_a = {adep: {} for adep in unique_airports["ADEP"]}

total = len(h_by_lat) * len(H_VARS) + len(a_by_lat) * len(A_VARS)
done  = 0

print(f"  Reading {len(h_by_lat)} lat-strips × {len(H_VARS)} hourly vars …")
for varname in H_VARS:
    if varname not in nc_h.variables:
        print(f"    skip {varname} (not in file)"); continue
    for lat_i, airports in h_by_lat.items():
        strip = read_lat_strip(nc_h, varname, h_lat_name, lat_i)
        for adep in airports:
            raw_h[adep][varname] = strip[:, h_lj[adep]]
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total} …", flush=True)

print(f"  Reading {len(a_by_lat)} lat-strips × {len(A_VARS)} accum vars …")
for varname in A_VARS:
    if varname not in nc_a.variables:
        print(f"    skip {varname} (not in file)"); continue
    for lat_i, airports in a_by_lat.items():
        strip = read_lat_strip(nc_a, varname, a_lat_name, lat_i)
        for adep in airports:
            raw_a[adep][varname] = strip[:, a_lj[adep]]
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{total} …", flush=True)

nc_h.close()
nc_a.close()

# Build per-airport lookup DataFrames (indexed by hour)
airport_wx = {}
for adep in unique_airports["ADEP"]:
    try:
        u10  = raw_h[adep].get("u10",  np.zeros(len(h_times)))
        v10  = raw_h[adep].get("v10",  np.zeros(len(h_times)))
        t2m  = raw_h[adep].get("t2m",  np.full(len(h_times), np.nan))
        d2m  = raw_h[adep].get("d2m",  np.full(len(h_times), np.nan))
        msl  = raw_h[adep].get("msl",  np.full(len(h_times), np.nan))
        lcc  = raw_h[adep].get("lcc",  np.full(len(h_times), np.nan))
        cape = raw_h[adep].get("cape", np.full(len(h_times), np.nan))

        wh = pd.DataFrame({
            "wind_speed_ms":   np.sqrt(u10**2 + v10**2),
            "temperature_c":   t2m - 273.15,
            "dewpoint_c":      d2m - 273.15,
            "pressure_hpa":    msl / 100.0,
            "low_cloud_frac":  lcc,
            "cape_jkg":        cape,
        }, index=h_times)

        wa = pd.DataFrame({
            "precip_mm":    raw_a[adep].get("tp",   np.zeros(len(a_times))) * 1000.0,
            "snowfall_mm":  raw_a[adep].get("sf",   np.zeros(len(a_times))) * 1000.0,
            "wind_gust_ms": raw_a[adep].get("fg10", np.zeros(len(a_times))),
            "cloud_base_m": raw_a[adep].get("cbh",  np.full(len(a_times), np.nan)),
        }, index=a_times)

        airport_wx[adep] = wh.join(wa, how="left")
    except Exception as e:
        pass   # airport stays out of dict; flights get NaN weather

print(f"  Weather ready for {len(airport_wx)} / {len(unique_airports)} airports")



print("\nJoining weather onto flights …")
WEATHER_COLS = [
    "wind_speed_ms", "temperature_c", "dewpoint_c", "pressure_hpa",
    "low_cloud_frac", "cape_jkg", "precip_mm", "snowfall_mm",
    "wind_gust_ms", "cloud_base_m",
]

wx_rows = []
for _, flight in flights.iterrows():
    adep = flight["ADEP"]
    t    = flight["DEP_HOUR"]

    rec = {col: np.nan for col in WEATHER_COLS}
    if adep in airport_wx:
        wx  = airport_wx[adep]
        idx = wx.index.get_indexer([t], method="nearest")[0]
        if 0 <= idx < len(wx):
            for col in WEATHER_COLS:
                if col in wx.columns:
                    rec[col] = wx.iloc[idx][col]

    wx_rows.append(rec)

wx_df  = pd.DataFrame(wx_rows, index=flights.index)
result = pd.concat([flights.drop(columns=["DEP_HOUR"]), wx_df], axis=1)

# How many rows have at least some weather?
has_weather = result["wind_speed_ms"].notna().sum()
print(f"  Flights with weather data: {has_weather:,} / {len(result):,} "
      f"({100*has_weather/len(result):.1f} %)")



print(f"\nSaving to:\n  {OUTPUT_PATH}")
result.to_csv(OUTPUT_PATH, index=False)

size_mb = result.memory_usage(deep=True).sum() / 1024**2
print(f"\nDone.  Shape: {result.shape[0]:,} rows × {result.shape[1]} columns  "
      f"(~{size_mb:.0f} MB in memory)")
print(f"\nWeather columns added:")
for col in WEATHER_COLS:
    present = col in result.columns and result[col].notna().any()
    print(f"  {'✓' if present else '✗'}  {col}")

print(f"\nAll done — load the merged file with:")
print(f"  df = pd.read_csv(r'{OUTPUT_PATH}')")