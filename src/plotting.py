from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.io_utils import write_table

sns.set_theme(style="whitegrid", context="talk")


def save_figure(fig, out_base: str | Path) -> list[Path]:
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def violin_by_region(df: pd.DataFrame, value_col: str, out_base: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=df, x="region", y=value_col, ax=ax, inner="quartile", cut=0)
    ax.set_title(title)
    save_figure(fig, out_base)
    write_table(df, Path(out_base).with_suffix(".tsv"))


def boxplot_by_region(df: pd.DataFrame, value_col: str, out_base: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="region", y=value_col, ax=ax)
    ax.set_title(title)
    save_figure(fig, out_base)
    write_table(df, Path(out_base).with_suffix(".tsv"))


def stacked_bar(df: pd.DataFrame, out_base: str | Path, title: str) -> None:
    pivot = df.pivot_table(index=["donor_id", "region"], columns="major_cell_class", values="fraction", fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_title(title)
    ax.set_ylabel("Fraction")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_figure(fig, out_base)
    write_table(df, Path(out_base).with_suffix(".tsv"))


def heatmap(df: pd.DataFrame, index: str, columns: str, values: str, out_base: str | Path, title: str) -> None:
    matrix = df.pivot_table(index=index, columns=columns, values=values, fill_value=0)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(matrix, cmap="viridis", ax=ax)
    ax.set_title(title)
    save_figure(fig, out_base)
    write_table(df, Path(out_base).with_suffix(".tsv"))


def scatter_spatial(df: pd.DataFrame, hue: str, out_base: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=df, x="x", y="y", hue=hue, s=10, linewidth=0, ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_figure(fig, out_base)
    write_table(df, Path(out_base).with_suffix(".tsv"))


def barplot(df: pd.DataFrame, x: str, y: str, out_base: str | Path, title: str, hue: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(title)
    if hue is not None:
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    save_figure(fig, out_base)
    write_table(df, Path(out_base).with_suffix(".tsv"))
