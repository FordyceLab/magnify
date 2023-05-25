import re

import numpy as np
import pandas as pd

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
    assay = assay.assign_coords(tag=np.unique(names_array))
    assay = assay.assign_coords(mark_tag=(("mark_row", "mark_col"), names_array))
    assay["valid"] = (
        ("mark_row", "mark_col", "time"),
        np.ones(
            (assay.sizes["mark_row"], assay.sizes["mark_col"], assay.sizes["time"]), dtype=bool
        ),
    )
    return assay


@registry.component("identify_mrbles")
def indentify_mrbles(assay, reference):
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
    assay = assay.assign_coords(tag=np.unique(names_array))
    assay = assay.assign_coords(mark_tag=(("mark_row", "mark_col"), names_array))
    assay["valid"] = (
        ("mark_row", "mark_col", "time"),
        np.ones(
            (assay.sizes["mark_row"], assay.sizes["mark_col"], assay.sizes["time"]), dtype=bool
        ),
    )
    return assay
