"""Plots of the curve and of genes along the pseudostructure coordinate."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData

from .tl import fit_curves


def curve(adata: AnnData, color: str = "pseudostructure", basis: str | None = None, dims: tuple[int, int] = (0, 1),
          cmap: str = "viridis", legend: str = "right", s: float = 3, ax=None):
    """Units in two dimensions of an embedding, with the curve drawn through them.

    By default the embedding is the one the curve was fitted in (``uns['coast']['use_rep']``), whose first two
    dimensions can fold the structure over itself. With ``basis`` (for example ``"X_umap"``), units are drawn in that
    embedding instead, and each curve node at the median position of the units nearest to it in the fitted
    embedding; nodes no unit is nearest to are skipped. This places the curve for display only; the fit is unchanged.

    Categorical colours are scanpy's (``uns[color + '_colors']``) when set; units are drawn in random order so no
    category hides another. ``legend="on data"`` writes each category at the median position of its units. Stars
    mark the two ends of the curve, 0 and 1 of the coordinate.
    """
    from scipy.spatial import cKDTree

    ax = ax or plt.figure(figsize=(4.5, 4)).gca()
    fitted, nodes = adata.obsm[adata.uns["coast"]["use_rep"]], adata.uns["coast"]["nodes"]
    if basis is None:
        X, nodes = fitted[:, list(dims)], nodes[:, list(dims)]
        xlabel, ylabel = f"dim {dims[0] + 1}", f"dim {dims[1] + 1}"
    else:
        X = adata.obsm[basis][:, list(dims)]
        nearest = cKDTree(nodes).query(fitted)[1]
        nodes = np.array([np.median(X[nearest == k], axis=0) for k in range(len(nodes)) if (nearest == k).any()])
        xlabel, ylabel = (f"{basis.removeprefix('X_')} {d + 1}" for d in dims)
    values = adata.obs[color]
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        values = values.astype("category")
        levels = values.cat.categories
        palette = adata.uns.get(f"{color}_colors")
        if palette is None or len(palette) != len(levels):
            palette = [plt.cm.tab20(i % 20) for i in range(len(levels))]
        order = np.random.default_rng(0).permutation(len(values))
        codes = values.cat.codes.to_numpy()[order]
        ax.scatter(*X[order].T, s=s, lw=0, c=[palette[c] for c in codes], rasterized=True)
        for level, colour in zip(levels, palette):
            if legend == "on data":
                inside = (values == level).to_numpy()
                if inside.any():
                    ax.text(*np.median(X[inside], axis=0), str(level), ha="center", va="center", fontsize=8,
                            weight="bold", color=colour, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colour, lw=0.8))
            else:
                ax.scatter([], [], s=3, color=colour, label=str(level))
        if legend != "on data":
            ax.legend(markerscale=4, frameon=False, fontsize=8)
    else:
        points = ax.scatter(*X.T, c=values, s=s, lw=0, cmap=cmap, rasterized=True)
        plt.colorbar(points, ax=ax, label=color, shrink=0.6)
    ax.plot(*nodes.T, "-o", color="black", ms=3, mfc="white", mew=0.6, lw=1)
    for end, text in ((nodes[0], "0"), (nodes[-1], "1")):
        ax.scatter(*end, marker="*", s=120, color="#D7263D", edgecolor="black", lw=0.4, zorder=5)
        ax.annotate(text, end, xytext=(6, -2), textcoords="offset points", fontsize=9, weight="bold")
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax


def genes(adata: AnnData, genes: Sequence[str], x: str = "pseudostructure", sample_key: str | None = None,
          layer: str = "counts", library: str | None = None, n_bins: int = 20, k: int = 10, scale: str = "minmax",
          axes=None):
    """One panel per gene: equal-count bin means (points) and the fitted curve (line), per sample.

    Curves and points cover the 5th to 95th percentile of ``obs[x]`` over all units, the same range for every
    sample (see `coast.tl.fit_curves`). ``scale="minmax"``: counts per 10k scaled to 0-1 by the lowest and highest
    point of the gene's curves; ``scale="log"``: counts per 10k on a log axis.
    """
    genes = list(genes)
    limits = np.percentile(adata.obs[x], [5, 95])
    if axes is None:
        _, axes = plt.subplots(1, len(genes), figsize=(3 * len(genes), 2.6), squeeze=False)
        axes = axes[0]
    curves = 2 ** fit_curves(adata, genes, x=x, sample_key=sample_key, layer=layer, library=library, k=k, limits=limits)
    if sample_key is None:
        samples, palette, curves = [None], [None], pd.concat({"": curves}, names=["sample"])
    else:
        samples = list(adata.obs[sample_key].astype("category").cat.categories)
        palette = adata.uns.get(f"{sample_key}_colors")
        if palette is None or len(palette) != len(samples):
            palette = [None] * len(samples)
    low = curves.groupby(level="gene").min().min(axis=1)
    span = curves.groupby(level="gene").max().max(axis=1) - low
    for sample, colour in zip(samples, palette):
        sub = adata if sample is None else adata[adata.obs[sample_key] == sample]
        counts = sub[:, genes].layers[layer] if layer else sub[:, genes].X
        counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
        lib = sub.obs[library].to_numpy(float) if library else np.asarray(
            (sub.layers[layer] if layer else sub.X).sum(axis=1)).ravel()
        position = sub.obs[x].to_numpy()
        inside = np.flatnonzero((position >= limits[0]) & (position <= limits[1]))
        bins = np.array_split(inside[np.argsort(position[inside])], n_bins)
        bin_x = [position[b].mean() for b in bins]
        for j, (gene, ax) in enumerate(zip(genes, axes)):
            bin_y = np.array([1e4 * counts[b, j].sum() / lib[b].sum() for b in bins])   # bins with no counts: not drawn
            y = curves.loc[(sample or "", gene)].to_numpy()
            if scale == "minmax":
                y, bin_y = (y - low[gene]) / span[gene], (bin_y - low[gene]) / span[gene]
            line, = ax.plot(curves.columns, y, label=sample, color=colour)
            ax.scatter(bin_x, bin_y, s=8, color=line.get_color())
            ax.set(title=gene, xlabel=x)
            if scale == "log":
                ax.set(ylabel="counts per 10k", yscale="log")
            else:
                ax.set(ylabel="scaled expression (0-1)")
    if sample_key is not None:
        axes[-1].legend(frameon=False, fontsize=8)
    return axes


def heatmap(curves: pd.DataFrame, order: Sequence[str] | None = None, ax=None):
    """Fitted curves (genes x positions, from `coast.tl.fit_curves`) z-scored per gene, ordered by peak.

    ``order`` gives the rows instead (for example another panel's genes ordered by peak, so two groups can be read
    row by row).
    """
    curves = curves.dropna(how="all")   # genes without a fit
    z = curves.sub(curves.mean(axis=1), axis=0).div(curves.std(axis=1), axis=0)
    if order is None:
        z = z.iloc[np.argsort(np.nanargmax(z.to_numpy(), axis=1), kind="stable")]
    else:
        z = z.reindex([g for g in order if g in z.index])
    ax = ax or plt.figure(figsize=(4, min(8, max(2, 0.12 * len(z))))).gca()
    image = ax.imshow(z.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                      extent=(z.columns[0], z.columns[-1], len(z), 0), interpolation="nearest")
    if len(z) <= 60:
        ax.set_yticks(np.arange(len(z)) + 0.5, z.index, fontsize=7)
    else:
        ax.set_yticks([])
    ax.set_xlabel(curves.columns.name or "position")
    plt.colorbar(image, ax=ax, label="z")
    return ax


def labels_along(adata: AnnData, label_key: str, order: Sequence[str] | None = None, x: str = "pseudostructure",
                 min_units: int = 20, ax=None):
    """Density of the coordinate within each label, one row per label (top to bottom in ``order``).

    Labels never enter the fit, so this shows how well the coordinate orders them. Colours are scanpy's
    (``uns[label_key + '_colors']``) when set.
    """
    from scipy.stats import gaussian_kde

    labels = adata.obs[label_key].astype("category")
    order = list(order) if order is not None else list(labels.cat.categories)
    palette = dict(zip(labels.cat.categories, adata.uns.get(f"{label_key}_colors", [None] * len(labels.cat.categories))))
    ax = ax or plt.figure(figsize=(3.5, 0.45 * len(order) + 0.6)).gca()
    grid = np.linspace(0, 1, 400)
    position = adata.obs[x].to_numpy(float)
    for i, label in enumerate(order):
        base = len(order) - 1 - i
        ax.axhline(base, color="#DDDDDD", lw=0.5, zorder=0)
        t = position[(labels == label).to_numpy()]
        if len(t) >= min_units:
            y = gaussian_kde(t)(grid)
            y = 0.85 * y / y.max()
            ax.fill_between(grid, base, base + y, color=palette.get(label), alpha=0.45, lw=0)
            ax.plot(grid, base + y, color=palette.get(label), lw=0.9)
    ax.set(xlim=(0, 1), ylim=(0, len(order)), xlabel=x)
    ax.set_yticks(np.arange(len(order))[::-1] + 0.3, order)
    ax.tick_params(axis="y", length=0)
    for side in ("left", "top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _band(labels: np.ndarray, bins: np.ndarray, ok: np.ndarray, min_span: int = 3) -> pd.Series:
    """Label of each bin: the label holding the largest share of its own units there (so a label's size does not
    decide it); runs shorter than ``min_span`` bins take the neighbouring run with the larger share."""
    ct = pd.crosstab(bins, labels).reindex(range(len(ok)), fill_value=0)
    share = ct.div(ct.sum(0).clip(lower=1), axis=1)
    label = share.idxmax(axis=1).where(ok)
    runs = []
    for i, g in label.items():
        if isinstance(g, str) and runs and runs[-1][0] == g and runs[-1][2] == i - 1:
            runs[-1][2] = i
        elif isinstance(g, str):
            runs.append([g, i, i])
    for j, r in enumerate(runs):
        if r[2] - r[1] + 1 < min_span and len(runs) > 1:
            near = {runs[n][0] for n in (j - 1, j + 1) if 0 <= n < len(runs)}
            r[0] = max(near, key=lambda g: share.loc[r[1]:r[2], g].sum())
    for g, first, last in runs:
        label.loc[first:last] = g
    return label


def markers_along(adata: AnnData, markers: Mapping[str, str], label_key: str, x: str = "pseudostructure",
                  layer: str | None = None, n_bins: int = 100, min_units: int = 5, ax=None):
    """Marker genes along the coordinate, under a band of the labels.

    ``markers`` maps each gene to the label it marks (row names "gene (label)"). The coordinate is cut into
    ``n_bins`` equal-width bins; each row is the gene's mean of ``X`` (or ``layer``) per bin, z-scored. The band
    gives each bin the label holding the largest share of that label's own units. Bins with fewer than
    ``min_units`` units are grey.
    """
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    genes = list(markers)
    position = adata.obs[x].to_numpy(float)
    bins = np.minimum((position * n_bins).astype(int), n_bins - 1)
    values = adata[:, genes].layers[layer] if layer else adata[:, genes].X
    values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
    count = np.bincount(bins, minlength=n_bins)
    mean = np.vstack([np.bincount(bins, weights=values[:, j], minlength=n_bins) / np.maximum(count, 1)
                      for j in range(len(genes))])
    mean[:, count < min_units] = np.nan
    z = (mean - np.nanmean(mean, 1, keepdims=True)) / np.nanstd(mean, 1, keepdims=True)
    labels = adata.obs[label_key].astype("category")
    band = _band(labels.astype(str).to_numpy(), bins, count >= min_units)
    levels = list(labels.cat.categories)
    palette = adata.uns.get(f"{label_key}_colors", [plt.cm.tab20(i % 20) for i in range(len(levels))])

    ax = ax or plt.figure(figsize=(7, 0.32 * len(genes) + 1)).gca()
    cmap = LinearSegmentedColormap.from_list("coast", ["#3B4CC0", "#C9D3EE", "#FFFFFF", "#F4B6B0", "#D7263D"])
    cmap.set_bad("#F2F2F2")
    image = ax.imshow(np.clip(z, -3, 3), aspect="auto", interpolation="nearest", cmap=cmap, vmin=-3, vmax=3,
                      extent=(0, 1, len(genes), 0))
    codes = band.map({g: i for i, g in enumerate(levels)}).to_numpy(float)
    top = ax.inset_axes([0, 1.02, 1, 0.9 / len(genes)])
    top.imshow(codes[None], aspect="auto", interpolation="nearest", cmap=ListedColormap(palette),
               vmin=-0.5, vmax=len(levels) - 0.5, extent=(0, 1, 0, 1))
    for g in levels:
        idx = np.flatnonzero(band.to_numpy() == g)
        if len(idx) > 3:
            top.text((np.median(idx) + 0.5) / n_bins, 0.5, g, ha="center", va="center", fontsize=7, color="white",
                     weight="bold")
    top.set_axis_off()
    ax.set_yticks(np.arange(len(genes)) + 0.5, [f"{g} ({markers[g]})" for g in genes], style="italic")
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set(xticks=[0, 0.5, 1], xlabel=x)
    plt.colorbar(image, ax=ax, ticks=[-3, 0, 3], shrink=0.6, label="z")
    return ax


def shape_clusters(table: pd.DataFrame, shapes: pd.DataFrame, groups: Mapping[str, str],
                   colors: Mapping[str, str] | None = None, n_examples: int = 2, axes=None):
    """One panel per cluster of `coast.tl.shape_clusters`: the mean shape of its genes in each sample (thin) and
    each group (thick), titled with its size and its genes with the largest change."""
    clusters = sorted(c for c in table["cluster"].unique() if c > 0)
    if axes is None:
        _, axes = plt.subplots(1, len(clusters), figsize=(3.2 * len(clusters), 2.8), squeeze=False)
        axes = axes[0]
    colors = colors or dict(zip(dict.fromkeys(groups.values()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    x = shapes.columns.to_numpy(float)
    samples = list(shapes.index.get_level_values(0).unique())
    for cluster, ax in zip(clusters, axes):
        members = table.index[table["cluster"] == cluster]
        per_sample = {s: shapes.loc[s].loc[members].mean(0).to_numpy() for s in samples}
        for s in samples:
            ax.plot(x, per_sample[s], color=colors[groups[s]], lw=0.6, alpha=0.7)
        for group in dict.fromkeys(groups.values()):
            ax.plot(x, np.mean([per_sample[s] for s in samples if groups[s] == group], 0), color=colors[group],
                    lw=2, label=group)
        ax.axhline(0, color="#999999", lw=0.5)
        top = table.loc[members].sort_values("rms", ascending=False).index[:n_examples]
        ax.set_title(f"{len(members)} genes", loc="left", fontsize=9)
        ax.set_title(", ".join(top), loc="right", fontsize=8, style="italic")
        ax.set(xlabel=shapes.columns.name or "position")
    axes[0].set_ylabel("mean shape (z)")
    axes[-1].legend(frameon=False, fontsize=8)
    return axes
