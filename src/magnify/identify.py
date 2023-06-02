import re

import numpy as np
import pandas as pd
import scipy
import sklearn.mixture

import magnify.registry as registry


@registry.component("identify_buttons")
def identify_buttons(assay, pinlist, blank=None):
    if blank is None:
        blank = ["", "blank", "BLANK"]

    df = pd.read_csv(pinlist)
    df["Indices"] = df["Indices"].apply(
        lambda s: [int(x) for x in re.sub(r"[\(\)]", "", s).split(",")]
    )
    # Replace blanks with the empty string.
    df["MutantID"] = df["MutantID"].replace(blank, "")
    # Zero-index the indices.
    cols, rows = np.array(df["Indices"].to_list()).T - 1
    names = df["MutantID"].to_numpy(dtype=str, na_value="")
    names_array = np.empty((max(rows) + 1, max(cols) + 1), dtype=names.dtype)
    names_array[rows, cols] = names
    assay = assay.assign_coords(tag=(("mark_row", "mark_col"), names_array))
    assay["valid"] = (
        ("mark_row", "mark_col", "time"),
        np.ones(
            (assay.sizes["mark_row"], assay.sizes["mark_col"], assay.sizes["time"]), dtype=bool
        ),
    )
    return assay


@registry.component("identify_mrbles")
def indentify_mrbles(assay, spectra, num_codes, reference="eu"):
    df = pd.read_csv(spectra)

    ref_idx = df[df["name"] == reference].index[0]
    channels = [c for c in assay.channel.values if c in df.columns]
    sp = df[channels].to_numpy()
    sel = assay.sel(time="0s", channel=channels)
    intensities = sel.roi.where(sel.fg).mean(dim=["roi_x", "roi_y"]).to_numpy()
    volumes = np.linalg.lstsq(sp.T, intensities.T, rcond=None)[0].T
    ratios = volumes / volumes[:, ref_idx : ref_idx + 1]
    assay = assay.assign_coords(ln=("ln", df["name"]))
    assay["ln_vol"] = (("mark", "ln"), volumes)
    assay["ln_ratio"] = (("mark", "ln"), ratios)

    tags = sklearn.cluster.DBSCAN(eps=1e-3, min_samples=5).fit_predict(ratios[:, 1:])
    # tags = model.predict(ratios[:, 1:])
    X = ratios[:, 1:]

    means = np.zeros((num_codes, ratios.shape[1] - 1))
    covs = np.zeros((num_codes, ratios.shape[1] - 1, ratios.shape[1] - 1))
    unique_tags, counts = np.unique(tags[tags != -1], return_counts=True)
    for i, t in enumerate(unique_tags[np.argsort(counts)][-num_codes:]):
        means[i] = np.mean(X[tags == t], axis=0)
        covs[i] = np.cov(X[tags == t], rowvar=False)
    proportions = np.ones(num_codes + 1)
    proportions[-1] = 0.1
    proportions /= proportions.sum()
    lower = np.min(X, axis=0)
    upper = np.max(X, axis=0)
    for i in range(300):
        probs = []
        for k in range(num_codes):
            probs.append(scipy.stats.multivariate_normal.pdf(X, means[k], covs[k]))
        probs.append(np.ones(X.shape[0]) / (upper - lower).prod())
        probs = proportions * np.array(probs).T
        probs = probs / probs.sum(axis=1)[:, np.newaxis]

        means = (
            np.sum(probs[:, :-1, np.newaxis] * X[:, np.newaxis, :], axis=0)
            / np.sum(probs[:, :-1], axis=0)[:, np.newaxis]
        )
        diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        covs = (
            np.sum(
                probs[:, :-1, np.newaxis, np.newaxis] * np.einsum("...i,...j->...ij", diff, diff),
                axis=0,
            )
            / np.sum(probs[:, :-1], axis=0)[:, np.newaxis, np.newaxis]
        )
        proportions = np.sum(probs, axis=0) / X.shape[0]

    tags = np.argmax(probs, axis=1)

    assay = assay.assign_coords(tag=("mark", tags))
    assay["valid"] = (
        ("mark", "time"),
        np.ones((assay.sizes["mark"], assay.sizes["time"]), dtype=bool),
    )

    return assay
