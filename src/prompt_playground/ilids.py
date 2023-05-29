import functools

import pandas as pd

from prompt_playground import ILIDS_PATH

TP_FP_SEQUENCES_PATH = (
    ILIDS_PATH / "data" / "handcrafted-metadata" / "tp_fp_sequences.csv"
)


@functools.cache
def sequences_df() -> pd.DataFrame:
    df = pd.read_csv(TP_FP_SEQUENCES_PATH, index_col=0)
    # Only keep relevant columns
    df = df[
        [
            "Classification",
            "Duration",
            "Distance",
            "SubjectApproachType",
            "SubjectDescription",
            "Distraction",
            "Stage",
        ]
    ]

    df["Alarm"] = df["Classification"] == "TP"
    df["y_true"] = df["Alarm"]

    df["category"] = None  # "create" a new column
    df.loc[df["Distraction"].notnull(), "category"] = "Distraction"
    df.loc[~df["Distraction"].notnull(), "category"] = "Background"
    df.loc[df["Alarm"], "category"] = "Alarm"

    return df
