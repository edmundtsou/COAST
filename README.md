# COAST

**C**ontinuous **O**rdering of **A**natomical **S**tructures from **T**ranscriptomics.

Many tissues are built from one structure repeated many times: nephrons in the kidney, crypts in the gut,
lobules in the liver. A tissue section cuts these structures into many separate cross-sections. COAST orders those
cross-sections along one continuous axis, the **pseudostructure** coordinate, using expression alone. It then tests
which genes change along that axis, and which change beyond the steps between discrete clusters.

COAST starts from the spatial data and a segmentation: the Visium HD bins of each section (Space Ranger's 2 µm
output) and the polygons of the segmented cross-sections (a GeoJSON file, e.g. a QuPath export). It sums the bins
inside each polygon into one *unit* per cross-section, and works on those units from then on. Segmenting the image
happens upstream; any unit x gene count matrix with a sample label can also enter directly at step 2.

## Install

```bash
pip install git+https://github.com/edmundtsou/COAST.git
```

Python ≥ 3.10. To run the tutorial, install the extras with `pip install "coast[tutorial] @ git+https://github.com/edmundtsou/COAST.git"`.

## Quick start

```python
import anndata as ad
import scanpy as sc
import coast

# 1. bins -> units: one row per segmented polygon, raw counts
units = {}
for sample in ["A", "B"]:
    bins = coast.io.read_visium_hd(f"{sample}/binned_outputs/square_002um")
    polygons = coast.io.read_segmentation(f"{sample}.geojson")
    units[sample] = coast.pp.bins_to_units(bins, polygons, sample)
adata = ad.concat(units, merge="same")

# 2. filter and normalise
adata = coast.pp.filter_units(adata, min_coverage=0.9)
coast.pp.normalize(adata)

# 3. integrate, label clusters, keep the structure, integrate again
coast.pp.integrate(adata, batch_key="sample")
sc.tl.leiden(adata)
coast.tl.label_clusters(adata, panels={"S1": [...], "S2": [...], "S3": [...], "other": [...]})
adata = adata[adata.obs["label"] != "other"].copy()
coast.pp.integrate(adata, batch_key="sample")

# 4. the pseudostructure coordinate
coast.tl.pseudostructure(adata, early_genes=[...], late_genes=[...])

# 5. genes along it (one sample at a time; keep genes that replicate)
result = coast.tl.test_genes(adata[adata.obs["sample"] == "A"])
curves = coast.tl.fit_curves(adata, result.index[result["significant"]])
coast.pl.heatmap(curves)
```

Two tutorials run the whole method on the kidney data of the paper (Visium HD) and reproduce its figures:

- [`tutorials/control_vs_aki.ipynb`](tutorials/control_vs_aki.ipynb): two control and two injured (AKI) mouse kidneys.
- [`tutorials/human_vs_mouse.ipynb`](tutorials/human_vs_mouse.ipynb): two control mouse kidneys and a healthy human
  kidney, on the shared 1:1 orthologs.

Each goes from the raw bins and the segmentation, through quality control, to the pseudostructure coordinate and its
check against segment labels. Along the proximal tubule it then tests which genes change along the axis
(`tl.test_genes`), which differ in level between the two groups (`tl.compare_groups`), and clusters the genes whose
shape changes (`tl.shape_clusters`). The input
matrices will be released with the paper.

## What each step does

| Function | What it does |
|---|---|
| `io.read_visium_hd`, `io.read_segmentation` | The in-tissue bins of a Space Ranger bin directory with their positions in image pixels, and the polygons of a GeoJSON segmentation with their class. |
| `pp.bins_to_units` | Gives every bin to the polygon covering its centre and sums the bins of each polygon: one unit per polygon, with its bins (`n_spots`), bin coverage of the polygon, and size and shape (area, perimeter, solidity, axes, ...). Records each bin's unit for `pp.doublets`. |
| `pp.filter_units` | Drops units below the hard floors (counts; optionally bin coverage, for units clipped by the capture edge). Then drops units whose log counts, log counts per bin (optional) or genes-vs-counts residual is more than 3 robust SDs from the median of their group (sample, or sample x any class column). |
| `pp.doublets` | Optional. Flags units holding two structures: two incompatible identities in the whole unit (1) or in two places of it (2, a permutation test on its bins; 2a when the identities are adjacent), large and not convex (3), or two pieces joined by a neck (4). |
| `pp.filter_genes` | Optional. Keeps genes detected in >= 5 % of units with >= 20 counts, plus listed markers. |
| `pp.normalize` | Keeps raw counts in `layers["counts"]` and sets `X` to log1p(counts per 10k). |
| `pp.integrate` | Variable genes (ranked within each sample; with `group_key`, only genes variable in every group) → PCA → Harmony → neighbours → UMAP. |
| `tl.label_clusters` | Gives each cluster the name of its best-scoring marker panel (any panels; control genes from the variable genes) and returns the score table with each cluster's margin over the runner-up. |
| `tl.pseudostructure` | Fits an elastic principal curve ([ElPiGraph](https://github.com/j-bac/elpigraph-python)) through the integrated embedding and projects units onto it. Units beyond the curve's ends go on the extension of the end edge. Marker genes set the direction. The coordinate is scaled to [0, 1]. |
| `tl.test_genes` | Fits a negative-binomial GLM per gene (raw counts, log-library offset, cubic spline in the coordinate) and compares it with an intercept-only model, or with one mean per cluster when `groupby` is given. Genes are called against a decoy with the coordinate shuffled. |
| `tl.fit_curves` | Fitted expression of each gene along the coordinate (log2 counts per 10k): a negative-binomial GAM, one penalised spline per sample, smoothing and dispersion by REML (tradeSeq's model). |
| `tl.compare_groups` | Which genes differ between two groups of samples along the coordinate: gap curves, replicated by every sample pair and larger than mixed-sample splits. |
| `tl.shape_clusters` | Which genes change shape (curves z-scored per sample, so level and amplitude do not count) between two groups of samples, against mixed-sample splits, and Ward clusters of how they change. |
| `pl.curve`, `pl.genes`, `pl.heatmap` | The curve in the fitted embedding or on the UMAP (`basis="X_umap"`), genes along the coordinate per sample, and fitted curves z-scored per gene, ordered by peak or in a given row order. |
| `pl.markers_along`, `pl.labels_along` | Marker genes along the coordinate under a band of labels, and the distribution of each label along it: how well the coordinate orders labels it never saw. |
| `pl.shape_clusters` | The mean shape of each cluster of `tl.shape_clusters`, per sample and per group. |
