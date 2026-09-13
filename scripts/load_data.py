"""Utilities for loading CSV datasets into pandas DataFrames."""

from pathlib import Path

import pandas as pd


def load_csv(filepath, **read_csv_kwargs):
    """Load a CSV file and return its contents as a pandas DataFrame."""
    csv_path = Path(filepath).expanduser()

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {csv_path}")

    return pd.read_csv(csv_path, **read_csv_kwargs)


def remove_dialogues_below_typicality_threshold(dataframe, threshold=1):
    """Keep complete dialogues whose two typicality scores meet a threshold."""
    score_columns = ["self-typical-confusion", "self-typical-interactions"]
    scores = dataframe[score_columns].apply(pd.to_numeric, errors="coerce")
    passing_rows = scores.ge(threshold).all(axis=1)
    dialogue_ids = dataframe.loc[passing_rows, "dialogue_id"].unique()

    return dataframe[dataframe["dialogue_id"].isin(dialogue_ids)].copy()


def remove_dialogues_with_failed_correctness_and_kc_annotation(dataframe):
    """Remove dialogues where every real turn lacks correctness and KCs."""
    turn_zero = (
        dataframe["turn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["solution", "turn 0", "0"])
    )
    real_turns = dataframe[~turn_zero].copy()
    has_correctness = (
        real_turns["correct"].astype("string").str.strip().str.lower().isin(["true", "false"])
    )
    kcs = real_turns["kcs"].fillna("").astype(str).str.strip()
    has_annotation = has_correctness | ~kcs.isin(["", "[]"])
    dialogue_ids = real_turns.loc[has_annotation, "dialogue_id"].unique()

    return dataframe[dataframe["dialogue_id"].isin(dialogue_ids)].copy()


def remove_turns_without_kcs(dataframe):
    """Remove KC-less real turns while preserving solution/turn-0 rows."""
    turn_zero = (
        dataframe["turn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["solution", "turn 0", "0"])
    )
    kcs = dataframe["kcs"].fillna("").astype(str).str.strip()
    return dataframe[turn_zero | ~kcs.isin(["", "[]"])].copy()


def apply_final_turn_override(dataframe):
    """Override each final real turn using the dialogue's self-correctness."""
    result = dataframe.copy()
    turn_zero = (
        result["turn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["solution", "turn 0", "0"])
    )
    real_turns = result[~turn_zero].copy()
    real_turns["_turn_number"] = pd.to_numeric(
        real_turns["turn"].astype("string").str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    final_indices = (
        real_turns.sort_values(["dialogue_id", "_turn_number"])
        .groupby("dialogue_id", sort=False)
        .tail(1)
        .index
    )
    overrides = {
        "Yes": "True",
        "No": "False",
        "Yes, but I had to reveal the answer": pd.NA,
    }

    for index in final_indices:
        has_correctness = str(result.at[index, "correct"]).strip().lower() in {
            "true",
            "false",
        }
        kcs = str(result.at[index, "kcs"]).strip()
        if not has_correctness or kcs in {"", "[]", "nan", "None"}:
            continue
        outcome = result.at[index, "self-correctness"]
        if outcome in overrides:
            result.at[index, "correct"] = overrides[outcome]

    return result


def remove_turns_without_correctness(dataframe):
    """Remove turns that do not have a True or False correctness label."""
    has_correctness = (
        dataframe["correct"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "false"])
    )
    return dataframe[has_correctness].copy()


def remove_dialogues_with_less_than_two_tagged_turns(dataframe):
    """Keep dialogues with at least two tagged turns, excluding turn 0."""
    turn_zero = (
        dataframe["turn"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["solution", "turn 0", "0"])
    )
    has_correctness = (
        dataframe["correct"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(["true", "false"])
    )
    kcs = dataframe["kcs"].fillna("").astype(str).str.strip()
    tagged_turns = dataframe[~turn_zero & has_correctness & ~kcs.isin(["", "[]"])]
    dialogue_counts = tagged_turns.groupby("dialogue_id").size()
    dialogue_ids = dialogue_counts[dialogue_counts >= 2].index

    return dataframe[dataframe["dialogue_id"].isin(dialogue_ids)].copy()


def load_paper_filtered_data(filepath, threshold=1):
    """Load a CSV and apply all paper-level filters while retaining turn 0."""
    dataframe = load_csv(filepath)
    dataframe = remove_dialogues_below_typicality_threshold(dataframe, threshold)
    dataframe = remove_dialogues_with_failed_correctness_and_kc_annotation(dataframe)
    dataframe = apply_final_turn_override(dataframe)
    dataframe = remove_turns_without_kcs(dataframe)
    dataframe = remove_turns_without_correctness(dataframe)
    dataframe = remove_dialogues_with_less_than_two_tagged_turns(dataframe)

    return dataframe
