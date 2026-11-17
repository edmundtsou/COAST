"""Inputs: Visium HD bins (Space Ranger binned outputs) and segmentation polygons (GeoJSON)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

POSITION_COLUMNS = ["in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]


def read_visium_hd(bin_dir) -> AnnData:
    """In-tissue bins of one Space Ranger bin directory (``binned_outputs/square_002um``), raw counts.

    The directory holds ``filtered_feature_bc_matrix.h5``, ``spatial/tissue_positions.parquet`` (or ``.csv``) and
    ``spatial/scalefactors_json.json``. Returns bins x genes with ``obs['x']``, ``obs['y']`` (bin centre in
    full-resolution image pixels, the frame segmentations of the H&E are drawn in), and ``uns['microns_per_pixel']``
    and ``uns['bin_area_px']`` (one bin's area in pixels²). Duplicate gene names are made unique.
    """
    import warnings

    import scanpy as sc

    bin_dir = Path(bin_dir)
    spatial = bin_dir / "spatial"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        bins = sc.read_10x_h5(str(bin_dir / "filtered_feature_bc_matrix.h5"))
    bins.var_names_make_unique()
    parquet = spatial / "tissue_positions.parquet"
    positions = pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(spatial / "tissue_positions.csv")
    positions.columns = [str(c).strip().lower() for c in positions.columns]
    positions["barcode"] = positions["barcode"].astype(str)
    positions = positions.drop_duplicates("barcode").set_index("barcode")[POSITION_COLUMNS]
    positions = positions.apply(pd.to_numeric, errors="coerce").reindex(bins.obs_names)
    bins = bins[(positions["in_tissue"].fillna(0) > 0).to_numpy()].copy()
    positions = positions.loc[bins.obs_names]
    bins.obs["x"] = positions["pxl_col_in_fullres"].to_numpy(float)
    bins.obs["y"] = positions["pxl_row_in_fullres"].to_numpy(float)
    scale = json.loads((spatial / "scalefactors_json.json").read_text())
    bins.uns["microns_per_pixel"] = float(scale["microns_per_pixel"])
    bins.uns["bin_area_px"] = float(scale["spot_diameter_fullres"]) ** 2
    return bins


def visium_hd_genes(bin_dir) -> pd.Index:
    """Gene names of a Space Ranger bin directory (made unique as in `read_visium_hd`), without reading counts."""
    import h5py
    from anndata.utils import make_index_unique

    with h5py.File(Path(bin_dir) / "filtered_feature_bc_matrix.h5", "r") as f:
        names = f["matrix/features/name"][:].astype(str)
    return make_index_unique(pd.Index(names))


def read_segmentation(path, id_property: str = "id", class_property: str = "classification",
                      score_measurement: str | None = "Score") -> pd.DataFrame:
    """Polygons of a GeoJSON FeatureCollection (a QuPath export, or any tool writing the same), one row per unit.

    Columns: ``unit_id`` (``properties[id_property]``, else ``unit_<feature_index>``), ``feature_index`` (position in
    the file), ``seg_class`` (``properties[class_property]``, a string or QuPath's ``{"name": ...}``), ``seg_score``
    (the ``properties.measurements`` entry named ``score_measurement``) and ``geometry`` (shapely; invalid polygons
    repaired with a zero-width buffer). Features without a usable polygon are skipped. Coordinates must be in the
    pixel frame of the bins.
    """
    from shapely.geometry import shape

    payload = json.loads(Path(path).read_text())
    features = payload.get("features", []) if isinstance(payload, dict) else list(payload)
    rows = []
    for i, feature in enumerate(features):
        try:
            polygon = shape(feature["geometry"]) if feature.get("geometry") else None
        except Exception:
            polygon = None
        if polygon is not None and not polygon.is_empty and not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon is None or polygon.is_empty or polygon.area <= 0:
            continue
        props = feature.get("properties", {}) or {}
        cls = props.get(class_property) or {}
        score = np.nan
        if score_measurement is not None:
            score = next((m.get("value") for m in props.get("measurements", []) or []
                          if m.get("name") == score_measurement), np.nan)
        rows.append({"unit_id": str(props.get(id_property, f"unit_{i}")), "feature_index": i,
                     "seg_class": str(cls.get("name", "unclassified") if isinstance(cls, dict) else cls),
                     "seg_score": float(score) if score is not None else np.nan, "geometry": polygon})
    if not rows:
        raise ValueError(f"{Path(path).name}: no usable polygons")
    table = pd.DataFrame(rows)
    if not table["unit_id"].is_unique:
        raise ValueError(f"{Path(path).name}: unit ids are not unique")
    return table
