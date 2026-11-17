"""Measure the share of each Bulacan barangay inside NOAH flood-hazard zones."""

import zipfile

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.transform import from_origin

from common import BOUNDARY_ZIP, HAZARD_FILE, NOAH_DIR

# Philippine Reference System 1992 / Zone III suits area calculations in Luzon.
EQUAL_AREA_CRS = "EPSG:3123"
CELL_DEGREES = 0.00018
RETURN_PERIODS = ["5yr", "25yr", "100yr"]
HAZARD_LEVELS = {1: "low", 2: "medium", 3: "high"}


def read_barangays() -> gpd.GeoDataFrame:
    with zipfile.ZipFile(BOUNDARY_ZIP) as archive:
        gdb_names = {name.split(".gdb/")[0] + ".gdb" for name in archive.namelist()
                     if ".gdb/" in name}
    if len(gdb_names) != 1:
        raise ValueError(f"Expected one geodatabase in the ZIP, found {sorted(gdb_names)}")
    source = f"zip://{BOUNDARY_ZIP.as_posix()}!{gdb_names.pop()}"

    layers = gpd.list_layers(source)["name"].tolist()
    matches = [name for name in layers if name.lower().endswith("admin4")]
    if len(matches) != 1:
        raise ValueError(f"Could not identify the barangay layer in {layers}")
    barangays = gpd.read_file(source, layer=matches[0], where="adm2_name = 'Bulacan'")
    if barangays.empty:
        raise ValueError("The boundary file returned no Bulacan barangays.")
    keep = ["adm4_pcode", "adm4_name", "adm3_name", "geometry"]
    return barangays[keep].to_crs("EPSG:4326")


def read_hazard(return_period: str) -> gpd.GeoDataFrame:
    path = NOAH_DIR / f"Bulacan_{return_period}.zip"
    hazard = gpd.read_file(f"zip://{path.as_posix()}")
    if "Var" not in hazard.columns:
        raise ValueError(f"{path.name} has no Var field: {list(hazard.columns)}")
    return hazard[hazard["Var"].isin(HAZARD_LEVELS)]


def main() -> None:
    barangays = read_barangays()
    table = barangays.drop(columns="geometry").set_index("adm4_pcode")
    table["area_m2"] = barangays.to_crs(EQUAL_AREA_CRS).area.to_numpy()
    print(f"Barangays: {len(table)} in {table['adm3_name'].nunique()} towns")

    # A grid of about 20 m turns slow polygon overlays into simple cell counts.
    left, bottom, right, top = barangays.total_bounds
    width = int(np.ceil((right - left) / CELL_DEGREES))
    height = int(np.ceil((top - bottom) / CELL_DEGREES))
    transform = from_origin(left, top, CELL_DEGREES, CELL_DEGREES)
    ids = rasterize(((geom, i + 1) for i, geom in enumerate(barangays.geometry)),
                    out_shape=(height, width), transform=transform, fill=0, dtype="int32")
    cells = np.bincount(ids.ravel(), minlength=len(barangays) + 1)[1:]

    for period in RETURN_PERIODS:
        hazard = read_hazard(period)
        print(f"{period}: {len(hazard)} hazard rows")
        # The source levels can overlap, so each cell keeps its highest level.
        levels = np.zeros((height, width), dtype="uint8")
        for code in sorted(HAZARD_LEVELS):
            shapes = hazard.loc[hazard["Var"] == code, "geometry"]
            mask = rasterize(((geom, 1) for geom in shapes), out_shape=(height, width),
                             transform=transform, fill=0, dtype="uint8")
            levels[mask == 1] = code
        for code, level in HAZARD_LEVELS.items():
            counts = np.bincount(ids[levels == code], minlength=len(barangays) + 1)[1:]
            table[f"share_{level}_{period}"] = np.round(counts / np.maximum(cells, 1), 4)

    table["exposure_score"] = (table["share_medium_25yr"]
                               + 2 * table["share_high_25yr"]).round(4)
    HAZARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(HAZARD_FILE, index=False)
    print(f"Wrote {HAZARD_FILE}")


if __name__ == "__main__":
    main()
