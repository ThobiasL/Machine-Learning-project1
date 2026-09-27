import eccodes
import netCDF4 as nc4
import numpy as np
from datetime import datetime, timedelta
import os
 
# ── Paths ─────────────────────────────────────────────────────────────────────
GRIB_PATH = r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\weather\data.grib"
OUT_DIR   = os.path.dirname(GRIB_PATH)
HOURLY_NC = os.path.join(OUT_DIR, "weather_hourly.nc")
ACCUM_NC  = os.path.join(OUT_DIR, "weather_accum.nc")
 
# Maps GRIB shortName → (nc variable name, output file, is_accum)
# ERA5 GRIB shortNames differ from cfgrib's Python-side names:
#   10u/10v/2t/2d/10fg  ← actual GRIB shortName
#   u10/v10/t2m/d2m/fg10 ← what cfgrib calls them in xarray
TARGETS = {
    "10u":  ("u10",  HOURLY_NC, False),
    "10v":  ("v10",  HOURLY_NC, False),
    "2t":   ("t2m",  HOURLY_NC, False),
    "2d":   ("d2m",  HOURLY_NC, False),
    "msl":  ("msl",  HOURLY_NC, False),
    "lcc":  ("lcc",  HOURLY_NC, False),
    "cape": ("cape", HOURLY_NC, False),
    "tp":   ("tp",   ACCUM_NC,  True),
    "sf":   ("sf",   ACCUM_NC,  True),
    "10fg": ("fg10", ACCUM_NC,  True),
    "cbh":  ("cbh",  ACCUM_NC,  True),
}
 
 
def get_valid_dt(mid):
    """Compute the valid datetime from a GRIB message."""
    date_val = eccodes.codes_get(mid, "dataDate")   # YYYYMMDD
    time_val = eccodes.codes_get(mid, "dataTime")   # HHMM (e.g. 600 = 06:00)
    step_h   = int(eccodes.codes_get(mid, "endStep"))
    ref = datetime(date_val // 10000,
                   (date_val % 10000) // 100,
                   date_val % 100,
                   time_val // 100,
                   time_val % 100)
    return ref + timedelta(hours=step_h)
 
 
# ── Pass 1: index every relevant message by shortName + validTime ─────────────
print("Pass 1: scanning GRIB index …")
print("  (this reads only message headers — fast)")
 
# index[shortName] = sorted list of (valid_datetime, file_offset)
index    = {}
grid_meta = {}   # shortName → (lat_1d, lon_1d_sorted, sort_idx, Nj, Ni)
seen_sn  = set()
 
with open(GRIB_PATH, "rb") as f:
    msg_count = 0
    while True:
        offset = f.tell()
        try:
            mid = eccodes.codes_grib_new_from_file(f)
        except Exception as e:
            print(f"  Warning at offset {offset}: {e}")
            break
        if mid is None:
            break
        msg_count += 1
 
        try:
            sn = eccodes.codes_get(mid, "shortName")
        except Exception:
            eccodes.codes_release(mid)
            continue
 
        if sn not in seen_sn:
            seen_sn.add(sn)
            marker = "  ← TARGET" if sn in TARGETS else ""
            print(f"    found shortName: {sn!r}{marker}")
 
        if sn in TARGETS:
            # Record file offset
            index.setdefault(sn, []).append((get_valid_dt(mid), offset))
 
            # Capture grid shape & coordinates from the FIRST message per variable
            if sn not in grid_meta:
                Ni  = eccodes.codes_get(mid, "Ni")
                Nj  = eccodes.codes_get(mid, "Nj")
                lat = eccodes.codes_get_array(mid, "latitudes" ).reshape(Nj, Ni)[:, 0]
                lon = eccodes.codes_get_array(mid, "longitudes").reshape(Nj, Ni)[0, :]
                # Convert 0-360 → -180/180
                lon = ((lon + 180) % 360) - 180
                sort_idx = np.argsort(lon)
                lon = lon[sort_idx]
                grid_meta[sn] = (lat, lon, sort_idx, Nj, Ni)
 
        eccodes.codes_release(mid)
 
print(f"\n  Scanned {msg_count} messages.")
print(f"  Target variables found: {list(index.keys())}")
missing = [sn for sn in TARGETS if sn not in index]
if missing:
    print(f"  NOT found in GRIB: {missing}")
    print("  (these will be skipped)")
 
# Sort each variable's messages by valid time
for sn in index:
    index[sn].sort(key=lambda x: x[0])
 
 
# ── Set up NetCDF output files ────────────────────────────────────────────────
print("\nCreating NetCDF output files …")
 
nc_handles = {}   # nc_path → nc4.Dataset
nc_time_vars  = {}   # nc_path → time variable
nc_time_index = {}   # nc_path → {datetime: int_index}
nc_data_vars  = {}   # (sn) → nc4 variable
 
for nc_path in (HOURLY_NC, ACCUM_NC):
    if os.path.exists(nc_path):
        os.remove(nc_path)
    ds = nc4.Dataset(nc_path, "w", format="NETCDF4")
    ds.createDimension("time", None)   # unlimited
    tv = ds.createVariable("time", "f8", ("time",))
    tv.units     = "hours since 1900-01-01 00:00:00.0"
    tv.calendar  = "standard"
    nc_handles[nc_path]    = ds
    nc_time_vars[nc_path]  = tv
    nc_time_index[nc_path] = {}
 
# Add lat/lon dimensions to each file using the first variable's grid
for nc_path in (HOURLY_NC, ACCUM_NC):
    ds = nc_handles[nc_path]
    # Find a variable whose grid we have for this file
    for sn, (nc_name, np_, is_accum) in TARGETS.items():
        if np_ == nc_path and sn in grid_meta:
            lat, lon, sort_idx, Nj, Ni = grid_meta[sn]
            ds.createDimension("latitude",  Nj)
            ds.createDimension("longitude", Ni)
            lv = ds.createVariable("latitude",  "f4", ("latitude",))
            ov = ds.createVariable("longitude", "f4", ("longitude",))
            lv[:] = lat
            ov[:] = lon
            lv.units = "degrees_north"
            ov.units = "degrees_east"
            break
 
# Create a NetCDF variable for each target we actually found
for sn, (nc_name, nc_path, is_accum) in TARGETS.items():
    if sn not in index:
        continue
    ds = nc_handles[nc_path]
    v = ds.createVariable(
        nc_name, "f4", ("time", "latitude", "longitude"),
        zlib=True, complevel=1, fill_value=np.nan,
    )
    nc_data_vars[sn] = v
 
 
# ── Pass 2: stream data messages → NetCDF ────────────────────────────────────
def dt_to_num(dt, units, calendar="standard"):
    """Convert datetime to the numeric value used by time variable."""
    from netCDF4 import date2num
    return date2num(dt, units=units, calendar=calendar)
 
 
print("\nPass 2: reading data and writing to NetCDF …")
print("  (one GRIB message at a time — memory stays low)")
 
for sn, messages in index.items():
    nc_name, nc_path, is_accum = TARGETS[sn]
    lat, lon, sort_idx, Nj, Ni = grid_meta[sn]
    v  = nc_data_vars[sn]
    tv = nc_time_vars[nc_path]
    ti = nc_time_index[nc_path]
 
    print(f"\n  {sn} → {nc_name}  ({len(messages)} time steps) …", flush=True)
    dot_every = max(1, len(messages) // 20)
 
    with open(GRIB_PATH, "rb") as f:
        for step_num, (valid_dt, offset) in enumerate(messages):
            # Assign time index
            if valid_dt not in ti:
                tidx = len(ti)
                ti[valid_dt] = tidx
                tv[tidx] = dt_to_num(valid_dt, tv.units)
            else:
                tidx = ti[valid_dt]
 
            # Seek to this message and read just the data values
            f.seek(offset)
            try:
                mid    = eccodes.codes_grib_new_from_file(f)
                values = eccodes.codes_get_array(mid, "values").reshape(Nj, Ni)
                eccodes.codes_release(mid)
            except Exception as e:
                print(f"\n    Warning: skipping offset {offset}: {e}")
                continue
 
            # Sort longitude columns if we reordered them
            v[tidx, :, :] = values[:, sort_idx]
 
            if step_num % dot_every == 0:
                print(".", end="", flush=True)
 
    print(f" done")
 
# Close output files
for ds in nc_handles.values():
    ds.close()
 
 
# ── Sanity check ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("Sanity check")
print(f"{'='*60}")
import xarray as xr
for path, lbl in [(HOURLY_NC, "Hourly"), (ACCUM_NC, "Accum")]:
    if os.path.exists(path) and os.path.getsize(path) > 100:
        size_mb = os.path.getsize(path) / 1024 ** 2
        with xr.open_dataset(path) as ds:
            print(f"  {lbl}: vars={list(ds.data_vars)}")
            print(f"         dims={dict(ds.dims)}  |  {size_mb:.0f} MB")
    else:
        print(f"  {lbl}: NOT CREATED or empty")
 
print("\nConversion complete.  Run flight_and_weather_corr.py next.")