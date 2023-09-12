from __future__ import annotations
import re

import numpy as np
import pandas as pd
import scipy

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
    assay = assay.assign_coords(
        tag=(("mark_row", "mark_col"), names_array),
        valid=(
            ("mark_row", "mark_col", "time"),
            np.ones((names_array.shape[0], names_array.shape[1], assay.sizes["time"]), dtype=bool),
        ),
    )
    return assay


@registry.component("identify_mrbles")
def indentify_mrbles(assay, spectra, codes, reference="eu"):
    # Read in the dataframe of lanthanide spectra and make sure the reference lanthanide is first.
    spectra_df = pd.read_csv(spectra)
    ref_idx = spectra_df[spectra_df["name"] == reference].index[0]
    spectra_df = spectra_df.reindex([ref_idx] + [i for i in range(len(spectra_df)) if i != ref_idx])
    lns = spectra_df["name"].to_list()
    num_lns = len(lns)

    # Read in the dataframe of codes.
    codes_df = pd.read_csv(codes)
    tag_names = codes_df["name"].to_numpy()
    num_codes = len(tag_names)
    # Make sure spectra and codes have consistent lanthanides.
    code_lns = set(codes_df.columns)
    code_lns.remove("name")
    if code_lns != set(lns):
        raise ValueError(f"Lanthanide names in {codes} do not match lanthanide names in {spectra}.")

    # Step 1: Estimate the lanthanide volumes in each bead by solving the linear equation SV = I
    # where S are the reference spectra and I are the intensities of each bead.
    channels = [c for c in assay.channel.values if c in spectra_df.columns]
    sp = spectra_df[channels].to_numpy()
    sel = assay.roi.isel(time=0).sel(channel=channels)
    intensities = sel.where(sel.fg).mean(dim=["roi_x", "roi_y"]).to_numpy()
    volumes = np.linalg.lstsq(sp.T, intensities.T, rcond=None)[0].T
    # We also want the lanthanide ratios with respect to the reference lanthanide.
    ratios = volumes / volumes[:, 0:1]
    # Save the resulting lanthanide volumes and ratios to the assay.
    assay = assay.assign_coords(ln=("ln", lns))
    assay["ln_vol"] = (("mark", "ln"), volumes)
    assay["ln_ratio"] = (("mark", "ln"), ratios)

    # Step 2: Agressively remove outliers to make future processing easier.
    X = ratios[:, 1:]
    # Find the distance to a point that should still be in the same cluster assuming cluster
    # sizes differ by a factor of at most 20 from the mean cluster size.
    n_neighbor = round(len(X) / (20 * num_codes)) + 2
    dist = (
        scipy.spatial.KDTree(X, leafsize=n_neighbor)
        .query(X, k=[n_neighbor], workers=-1)[0]
        .flatten()
    )
    # We care more about excluding all outliers so exclude 5% of points.
    X_r = X[dist <= np.percentile(dist, 95)]

    # Step 3: Find an affine transformation of the code's lanthanide ratios to get a clustering
    # that minimizes the distance between each bead and its closest code.
    code_ratios = codes_df[lns[1:]].to_numpy()

    # We will try to find a good affine transformation by minimizing a function that approximates
    # a per-cluster distance function.
    def loss(theta):
        A = theta[: num_lns - 1]
        p = theta[num_lns - 1 :]
        eps = 1e-8
        dist = np.linalg.norm((A * code_ratios + p)[np.newaxis] - X_r[:, np.newaxis], axis=-1)
        # Logsumexp is a smooth approximation to the max function when eps is small.
        return -eps * np.sum(scipy.special.logsumexp(-dist / eps, axis=-1)) / len(X_r)

    # Minimize the loss only considering affine transforms close to an estimated scaling factor.
    scales = (X_r.max(axis=0) - X_r.min(axis=0)) / (
        code_ratios.max(axis=0) - code_ratios.min(axis=0)
    )
    theta = scipy.optimize.minimize(
        loss,
        x0=np.concatenate((scales, X_r.min(axis=0) - scales * code_ratios.min(axis=0))),
        bounds=np.concatenate(
            (
                (
                    np.column_stack((0.75 * scales, 1.25 * scales)),
                    np.column_stack(
                        (
                            (X_r.min(axis=0) - 1.25 * scales * code_ratios.min(axis=0)),
                            X_r.max(axis=0) - 0.75 * scales * code_ratios.max(axis=0),
                        )
                    ),
                )
            )
        ),
    ).x
    A = theta[: num_lns - 1]
    p = theta[num_lns - 1 :]

    # Cluster points to the closest code.
    tag_idxs = np.argmin(
        np.linalg.norm(X_r[:, np.newaxis] - (A * code_ratios + p)[np.newaxis], axis=-1), axis=1
    )

    # Change the number of codes to be the number of clusters we found.
    nonzero_tags, counts = np.unique(tag_idxs, return_counts=True)
    nonzero_tags = nonzero_tags[counts >= 5]
    num_codes = len(nonzero_tags)

    # Step 4: Perform a better clustering using a Gaussian mixture model initialized with the
    # clustering from step 3. We also add a uniform distribution to the mixture which allows us to
    # exclude outliers less agressively.
    means = np.zeros((num_codes, num_lns - 1))
    covs = np.zeros((num_codes, num_lns - 1, num_lns - 1))
    proportions = np.zeros(num_codes + 1)
    # Initialize the Gaussian components.
    for i, tag_idx in enumerate(nonzero_tags):
        proportions[i] = np.sum(tag_idxs == tag_idx)
        means[i] = np.median(X_r[tag_idxs == tag_idx], axis=0)
        covs[i] = np.cov(X_r[tag_idxs == tag_idx], rowvar=False)
    # Initialize the uniform component.
    proportions[-1] = len(X) - len(X_r)
    proportions /= proportions.sum()
    log_cond_probs = np.empty((len(X), num_codes + 1))
    log_cond_probs[:, -1] = -np.log(X.max(axis=0) - X.min(axis=0)).sum()
    probs = None

    # Run the Expectation-Maximization algorithm.
    for i in range(30):
        # E-step: Compute the probability of each point belonging to each component.
        diff = X[:, np.newaxis, :] - means[np.newaxis, :, :]
        # Work in log space most of the time to avoid numerical issues.
        try:
            log_cond_probs[:, :-1] = (
                -X.shape[1] * np.log(2 * np.pi) / 2
                - 0.5 * np.log(np.linalg.det(covs))
                - 0.5 * np.einsum("...i,...ij,...j->...", diff, np.linalg.inv(covs), diff)
            )
        except np.linalg.LinAlgError:
            print("Warning: Code clustering did not converge.")
            break
        log_probs = np.log(proportions) + log_cond_probs
        log_probs -= scipy.special.logsumexp(log_probs, axis=1)[:, np.newaxis]
        probs = np.exp(log_probs)
        # M-step: Update the parameters of each component.
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
        # Regularize the covariance matrix to avoid numerical issues.
        covs += np.eye(X.shape[1]) * 1e-6
        proportions = np.sum(probs, axis=0) / X.shape[0]

    # Assign each bead a code based on the clustering we just found.
    nonzero_tags = np.append(nonzero_tags, -1)
    tag_names = np.append(tag_names, "outlier")
    if probs is not None:
        tag_idxs = nonzero_tags[np.argmax(probs, axis=1)]
    else:
        tag_idxs = np.argmin(
            np.linalg.norm(X[:, np.newaxis] - (A * code_ratios + p)[np.newaxis], axis=-1), axis=1
        )
    assay = assay.assign_coords(
        tag=("mark", tag_names[tag_idxs]),
    )

    return assay
