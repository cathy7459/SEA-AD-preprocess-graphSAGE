from __future__ import annotations

import pandas as pd


def cell_type_fraction_table(node_metadata: pd.DataFrame, donor_id: str, region: str) -> pd.DataFrame:
    if "major_cell_class" not in node_metadata.columns:
        return pd.DataFrame()
    labels = node_metadata["major_cell_class"].astype("string").fillna("unknown")
    counts = labels.value_counts(normalize=True).rename_axis("major_cell_class")
    out = counts.reset_index(name="fraction")
    out["donor_id"] = donor_id
    out["region"] = region
    return out


def aggregate_patient_features(
    graph_summaries: pd.DataFrame,
    cell_type_fractions: pd.DataFrame,
    mapping_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    graph_wide = (
        graph_summaries.pivot_table(
            index="donor_id",
            columns=["region", "modality"],
            values=["n_nodes", "n_edges", "n_features"],
            fill_value=0,
        )
        if not graph_summaries.empty
        else pd.DataFrame()
    )
    graph_wide.columns = ["__".join(map(str, col)) for col in graph_wide.columns]

    fraction_wide = (
        cell_type_fractions.assign(feature=lambda df: "celltype_fraction__" + df["region"] + "__" + df["major_cell_class"])
        .pivot_table(index="donor_id", columns="feature", values="fraction", fill_value=0)
        if not cell_type_fractions.empty
        else pd.DataFrame()
    )
    pieces = [df for df in [graph_wide, fraction_wide] if not df.empty]
    if mapping_summary is not None and not mapping_summary.empty:
        map_wide = (
            mapping_summary.pivot_table(index="donor_id", columns="region", values="mapped_confidence_mean", fill_value=0)
            .add_prefix("mapped_confidence_mean__")
        )
        pieces.append(map_wide)
    if not pieces:
        return pd.DataFrame(columns=["donor_id"])
    merged = pd.concat(pieces, axis=1).reset_index()
    return merged
