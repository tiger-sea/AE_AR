import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def remove_zero(df: pd.DataFrame) -> pd.DataFrame:
    """
    input: Dataframe including label 0
    output: Dataframe removed 0 and reindexed
    """
    df = df[df["event"] != 0]
    df = df.reset_index()
    df = df.drop("index", axis=1) # remove index column that is made automaticaly at rest_index()
    return df


def scaler(df: pd.DataFrame, method: str="std") -> pd.DataFrame:
    """
    input: Dataframe for scaling and method designation
    output: Scaled dataframe
    
    std (standardize): mean to 0, std to 1
    nrm (normalize): scale to 0 ~ 1
    default is std
    """
    if method == "std":
        scaler = StandardScaler()
    elif method == "nrm":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    
    labels = df["event"]
    columns_name = df.drop("event", axis=1).columns
    df = scaler.fit_transform(df.drop("event", axis=1))
    df = pd.DataFrame(df, columns=columns_name)
    df["event"] = labels
    return df


def create_windows(array: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """
    make sliding window lists
    """
    x = []
    for i in range(0, len(array) - window_size + 1, step_size):
        x.append(array[i:i + window_size])
    x_out = np.array(x)
    return x_out


def synthesize_vectors(x: pd.Series, y: pd.Series, z: pd.Series) -> np.ndarray:
    vectors = np.sqrt(x**2 + y**2 + z**2)
    return vectors


def load_csv(left_files: list, right_files) -> pd.DataFrame:

    left_columns_name = ["L_accX", "L_accY", "L_accZ", "L_bpm", "L_temp", "event"]
    right_columns_name = ["R_accX", "R_accY", "R_accZ", "R_bpm", "R_temp", "event"]

    df_list = []

    # read every file one by one
    for left, right in zip(left_files, right_files):
        # read both hands data
        left_df = pd.read_csv(left, header=None, names=left_columns_name)
        right_df = pd.read_csv(right, header=None, names=right_columns_name)

        # drop "event" column not to duplicate, and unnecessary columns
        left_df = left_df.drop(["L_bpm", "L_temp"], axis=1)
        right_df = right_df.drop(["R_bpm", "R_temp", "event"], axis=1)

        df = pd.concat([left_df, right_df], axis=1)
        df_list.append(df)

    # concatenate dataframes in vertical direction
    df = pd.concat(df_list, axis=0, ignore_index=True)

    # could be helpful
    del df_list, left_df, right_df
    gc.collect()

    # remove label 0
    df = remove_zero(df)

    return df