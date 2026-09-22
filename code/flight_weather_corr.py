import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
 
# ── Paths ─────────────────────────────────────────────────────────────────────
FLIGHT_PATH = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights\Flights_20220301_20220331.csv"
WEATHER_DIR = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\weather"
HOURLY_NC   = rf"{WEATHER_DIR}\weather_hourly.nc"
ACCUM_NC    = rf"{WEATHER_DIR}\weather_accum.nc"
 
 
# ── 1. Load flights ───────────────────────────────────────────────────────────
print("Loading flight data …")
flights = pd.read_csv(FLIGHT_PATH, low_memory=False)
 
for col in ["FILED OFF BLOCK TIME", "ACTUAL OFF BLOCK TIME",
            "FILED ARRIVAL TIME",   "ACTUAL ARRIVAL TIME"]:
    if col in flights.columns:
        flights[col] = pd.to_datetime(flights[col], dayfirst=True, errors="coerce")
 
flights["DEP_DELAY_MIN"] = (
    flights["ACTUAL OFF BLOCK TIME"] - flights["FILED OFF BLOCK TIME"]
).dt.total_seconds() / 60
 
flights["ARR_DELAY_MIN"] = (
    flights["ACTUAL ARRIVAL TIME"] - flights["FILED ARRIVAL TIME"]
).dt.total_seconds() / 60
 
flights = flights.dropna(subset=["DEP_DELAY_MIN", "ARR_DELAY_MIN",
                                  "ADEP Latitude", "ADEP Longitude"])
flights = flights[flights["DEP_DELAY_MIN"].between(-60, 600)]
flights = flights[flights["ARR_DELAY_MIN"].between(-60, 600)]
 
print(f"  Flights loaded: {len(flights):,}")
 
 
# ── 2. Direct flight-level correlation: lat/lon vs delay ─────────────────────
print("\n" + "="*60)
print("FLIGHT-LEVEL CORRELATION: LAT/LON → DELAY")
print("="*60)
 
for col in ["ADEP Latitude", "ADEP Longitude"]:
    r_dep = flights[[col, "DEP_DELAY_MIN"]].corr().iloc[0, 1]
    r_arr = flights[[col, "ARR_DELAY_MIN"]].corr().iloc[0, 1]
    print(f"  {col:<20}  dep r={r_dep:+.4f}   arr r={r_arr:+.4f}")
 
 
# ── 3. Per-airport averages ───────────────────────────────────────────────────
print("\nComputing per-airport averages …")
airport_stats = (
    flights.groupby("ADEP")
    .agg(
        lat          = ("ADEP Latitude",  "first"),
        lon          = ("ADEP Longitude", "first"),
        avg_dep_delay= ("DEP_DELAY_MIN",  "mean"),
        avg_arr_delay= ("ARR_DELAY_MIN",  "mean"),
        n_flights    = ("DEP_DELAY_MIN",  "count"),
    )
    .reset_index()
)
# Only keep airports with enough flights for a meaningful average
airport_stats = airport_stats[airport_stats["n_flights"] >= 10]
print(f"  Airports with ≥10 flights: {len(airport_stats)}")
 
print("\n" + "="*60)
print("AIRPORT-LEVEL CORRELATION: LAT/LON → MEAN DELAY")
print("="*60)
for col in ["lat", "lon"]:
    r_dep = airport_stats[[col, "avg_dep_delay"]].corr().iloc[0, 1]
    r_arr = airport_stats[[col, "avg_arr_delay"]].corr().iloc[0, 1]
    label = "Latitude " if col == "lat" else "Longitude"
    print(f"  {label:<12}  avg dep delay r={r_dep:+.4f}   avg arr delay r={r_arr:+.4f}")
 
print("\nTop 5 airports by latitude (northernmost):")
top_north = airport_stats.nlargest(5, "lat")[["ADEP","lat","lon","avg_dep_delay","n_flights"]]
print(top_north.to_string(index=False))
 
print("\nTop 5 airports by latitude (southernmost):")
top_south = airport_stats.nsmallest(5, "lat")[["ADEP","lat","lon","avg_dep_delay","n_flights"]]
print(top_south.to_string(index=False))
 
 
# ── 4. Load mean weather per airport from NetCDF ──────────────────────────────
import os
if os.path.exists(HOURLY_NC) and os.path.exists(ACCUM_NC):
    print("\nLoading mean weather per airport from NetCDF …")
    import netCDF4 as nc4
    from collections import defaultdict
 
    nc_h = nc4.Dataset(HOURLY_NC, "r")
    nc_a = nc4.Dataset(ACCUM_NC,  "r")
 
    def get_coord(nc, *names):
        for n in names:
            if n in nc.variables:
                return n, np.asarray(nc.variables[n][:], dtype=float)
        return None, None
 
    h_lat_name, h_lat = get_coord(nc_h, "latitude", "lat")
    h_lon_name, h_lon = get_coord(nc_h, "longitude", "lon")
    a_lat_name, a_lat = get_coord(nc_a, "latitude", "lat")
    a_lon_name, a_lon = get_coord(nc_a, "longitude", "lon")
    h_lon = ((h_lon + 180) % 360) - 180
    a_lon = ((a_lon + 180) % 360) - 180
 
    def read_lat_strip(nc, varname, lat_name, lat_idx):
        var  = nc.variables[varname]
        dims = list(var.dimensions)
        idx  = [slice(None)] * len(dims)
        idx[dims.index(lat_name)] = lat_idx
        data = np.asarray(var[tuple(idx)], dtype=np.float32).squeeze()
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data   # (T, Nlon)
 
    # Group airports by nearest lat index
    h_li, h_lj, a_li, a_lj = {}, {}, {}, {}
    for _, row in airport_stats.iterrows():
        adep = row["ADEP"]
        lat, lon = row["lat"], row["lon"]
        h_li[adep] = int(np.argmin(np.abs(h_lat - lat)))
        h_lj[adep] = int(np.argmin(np.abs(h_lon - lon)))
        a_li[adep] = int(np.argmin(np.abs(a_lat - lat)))
        a_lj[adep] = int(np.argmin(np.abs(a_lon - lon)))
 
    h_by_lat = defaultdict(list)
    a_by_lat = defaultdict(list)
    for adep in airport_stats["ADEP"]:
        h_by_lat[h_li[adep]].append(adep)
        a_by_lat[a_li[adep]].append(adep)
 
    # Read annual mean of each weather variable per airport
    wx_means = {adep: {} for adep in airport_stats["ADEP"]}
 
    for varname in ["u10", "v10", "t2m", "msl", "lcc", "cape"]:
        if varname not in nc_h.variables: continue
        for lat_i, airports in h_by_lat.items():
            strip = read_lat_strip(nc_h, varname, h_lat_name, lat_i)
            for adep in airports:
                wx_means[adep][varname] = float(np.nanmean(strip[:, h_lj[adep]]))
 
    for varname in ["tp", "fg10"]:
        if varname not in nc_a.variables: continue
        for lat_i, airports in a_by_lat.items():
            strip = read_lat_strip(nc_a, varname, a_lat_name, lat_i)
            for adep in airports:
                wx_means[adep][varname] = float(np.nanmean(strip[:, a_lj[adep]]))
 
    nc_h.close()
    nc_a.close()
 
    # Build weather DataFrame
    wx_df = pd.DataFrame(wx_means).T.reset_index().rename(columns={"index": "ADEP"})
    wx_df["u10"] = wx_df.get("u10", 0)
    wx_df["v10"] = wx_df.get("v10", 0)
    if "u10" in wx_df and "v10" in wx_df:
        wx_df["mean_wind"] = np.sqrt(wx_df["u10"]**2 + wx_df["v10"]**2)
    if "t2m" in wx_df:
        wx_df["mean_temp_c"] = wx_df["t2m"] - 273.15
    if "msl" in wx_df:
        wx_df["mean_pres_hpa"] = wx_df["msl"] / 100.0
    if "tp" in wx_df:
        wx_df["mean_precip_mm"] = wx_df["tp"] * 1000.0
 
    combined = airport_stats.merge(wx_df, on="ADEP", how="inner")
 
    print(f"  Airports with weather data: {len(combined)}")
 
    # Correlate lat/lon with mean weather
    WX_COLS = [c for c in ["mean_wind","mean_temp_c","mean_pres_hpa",
                            "mean_precip_mm","lcc","cape","fg10"]
               if c in combined.columns]
 
    print("\n" + "="*60)
    print("LAT/LON CORRELATION WITH MEAN ANNUAL WEATHER")
    print("="*60)
    print(f"  {'Variable':<18}  {'lat r':>8}  {'lon r':>8}")
    print(f"  {'-'*18}  {'------':>8}  {'------':>8}")
    for col in WX_COLS:
        r_lat = combined[["lat", col]].corr().iloc[0, 1]
        r_lon = combined[["lon", col]].corr().iloc[0, 1]
        print(f"  {col:<18}  {r_lat:>+8.4f}  {r_lon:>+8.4f}")
 
    # Correlate mean weather with mean delay
    print("\n" + "="*60)
    print("MEAN WEATHER CORRELATION WITH MEAN DEPARTURE DELAY")
    print("(airport-level averages)")
    print("="*60)
    for col in WX_COLS:
        r = combined[[col, "avg_dep_delay"]].corr().iloc[0, 1]
        bar = "█" * int(abs(r) * 40)
        sign = "+" if r >= 0 else "-"
        print(f"  {col:<18}  {sign}{abs(r):.4f}  {bar}")
 
else:
    print("\n(NetCDF files not found — skipping weather section)")
 
 
print("\nDone.")