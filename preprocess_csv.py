import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from params import *
# BASE_DATASET_PATH = r"C:/Users/mlamali.saidsalimo/Downloads/Hand Gesture Recognition (HGR)/asl-signs"
from tqdm import tqdm

FACE_N_LANDMARKS = 468
HAND_N_LANDMARKS = 21
POSE_N_LANDMARKS = 33
ROWS_PER_FRAME = FACE_N_LANDMARKS + HAND_N_LANDMARKS * 2 + POSE_N_LANDMARKS  # 543
NUM_SHARDS = 2
SAVE_PATH = '/tmp/GoogleISLDataset'
BATCH_SIZE = 256

# 1 -> 001
SET_LANDMARKS = set(
    [f"face-{i:03d}" for i in range(FACE_N_LANDMARKS)] + [f"left_hand-{i:03d}" for i in range(HAND_N_LANDMARKS)] + [
        f"pose-{i:03d}" for i in range(POSE_N_LANDMARKS)] + [
        f"right_hand-{i:03d}" for i in range(HAND_N_LANDMARKS)])

# write txt landmark file, each line is a landmark name
landmark_order_path = f"{BASE_DATASET_PATH}/landmark_order.txt"
with open(landmark_order_path, "w") as f:
    text = "\n".join(SET_LANDMARKS)
    f.write(text)

SET_LANDMARKS_TYPES = {"face", "left_hand", "right_hand", "pose"}


def clean_sequences_landmarks_df(df, drop_z=False):
    """Clean sequences landmarks dataframe."""

    # delete z column if exists
    if "z" in df.columns and drop_z:
        df = df.drop(columns=["z"])

    # normalize x and y values
    # scaler = MinMaxScaler()
    # df[["x", "y"]] = scaler.fit_transform(df[["x", "y"]])

    # uniques frames
    frames = df["frame"].unique().tolist()  # [0, 1, 2, ..., 18] or [23, 24, 25, ..., 41]
    frame_to_new_frame = {frame: i for i, frame in enumerate(frames)}  # map frame to new frame

    # rename row_id : 18-face-1 -> 0-face-001, 18-face-002 -> 0-face-002, ..., 19-face-001 -> 1-face-001, ...
    df["row_id"] = df["row_id"].apply(
        lambda
            row_id: f"{frame_to_new_frame[int(row_id.split('-')[0])]}-{row_id.split('-')[1]}-{int(row_id.split('-')[2]):03d}")

    # sort by row_id
    df = df.sort_values("row_id")
    # index by row_id
    df = df.set_index("row_id")

    return df


def verify_sequences_landmarks_df(df: pd.DataFrame) -> bool:
    """Verify sequences landmarks dataframe."""
    # unique row id without frame prefix
    series_row_id = df.index.map(lambda x: "-".join(x.split("-")[1:]))
    row_id_unique_names = series_row_id.unique()
    n_rows = len(df)

    # check if all landmarks are present
    if set(df["type"].unique()) != SET_LANDMARKS_TYPES:
        return False, "Invalid landmarks types"
    if set(row_id_unique_names) != SET_LANDMARKS:
        return False, "Invalid number of landmarks"
    if n_rows % len(SET_LANDMARKS) != 0:
        return False, "Invalid number of rows"

    return True, "Valid dataframe"


def preprocess_sequences_landmarks_df(parquet_path: str, drop_z=False) -> pd.DataFrame:
    # read parquet file
    df = pd.read_parquet(parquet_path)

    # clean dataframe
    df = clean_sequences_landmarks_df(df, drop_z=drop_z)

    # verify dataframe
    is_valid, msg = verify_sequences_landmarks_df(df)
    if not is_valid:
        raise ValueError(f"Invalid dataframe ({parquet_path}): {msg}")

    # keep only ["row_id", "x", "y"] columns
    df = df[["x", "y"]] if drop_z else df[["x", "y", "z"]]

    return df


def split_dataset(df, val_size=0.2, random_state=42, grouping_by_participant=True):
    """Split dataset into train and validation set."""
    if grouping_by_participant:
        df_train = df.groupby("participant_id").apply(lambda x: x.sample(frac=1 - val_size, random_state=random_state))
        df_val = df.drop(df_train.index)
    else:
        df_train, df_val = train_test_split(df, test_size=val_size, random_state=random_state)
    return df_train, df_val


def task(parquet_path: str, output_dirname: str = "train_landmark_files_preprocessed/train", drop_z: bool = True,
         overwrite: bool = False):
    df = preprocess_sequences_landmarks_df(parquet_path, drop_z=drop_z)

    # save cleaned dataframe
    output_path = parquet_path.replace("train_landmark_files", output_dirname)
    # if already exists, skip
    if os.path.exists(output_path) and not overwrite:
        return output_path

    # create all dirs
    parent_dir = os.path.dirname(output_path)  # get parent dir full path
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # df.to_parquet(output_path)
    df.to_csv(output_path.replace(".parquet", ".csv"), index=False)

    # get index of df to series
    df_index = df.index.to_series()
    # save index
    df_index.to_csv(output_path.replace(".parquet", "_index.csv"), index=False)

    return output_path


def main():
    N_WORKERS = 15
    OVERWRITE = False
    DROP_Z = True  # keep only x and y columns
    LIMIT_PARTICIPANTS = 3

    new_train_landmark_files_dirname = f'train_landmark_files_preprocessed_{"xyz" if not DROP_Z else "xy"}'
    for dir_to_create in [f"{new_train_landmark_files_dirname}/train", f"{new_train_landmark_files_dirname}/val"]:
        os.makedirs(f"{BASE_DATASET_PATH}/{dir_to_create}", exist_ok=True)

    print(f"* Number of workers: {N_WORKERS}")
    print(f"* Overwrite: {OVERWRITE}")
    print(f"* Drop z: {DROP_Z}")
    print(f"* Output dir: {BASE_DATASET_PATH}/{new_train_landmark_files_dirname}")

    train_path = f"{BASE_DATASET_PATH}/train.csv"
    df_train_full = pd.read_csv(train_path).sort_values(by=["participant_id", "sign", "sequence_id"]).reset_index(
        drop=True)

    # limit number of participants
    if LIMIT_PARTICIPANTS is not None:
        # get random participants id
        participants_id = df_train_full["participant_id"].unique()
        random.seed(42)
        random.shuffle(participants_id)
        participants_id = participants_id[:LIMIT_PARTICIPANTS]
        df_train_full = df_train_full[df_train_full["participant_id"].isin(participants_id)]
    df_train_full["path"] = df_train_full["path"].apply(lambda parquet_path: f"{BASE_DATASET_PATH}/{parquet_path}")

    df_train, df_val = split_dataset(df_train_full, val_size=0.2, random_state=12, grouping_by_participant=False)

    parquet_paths_train = df_train["path"].tolist()
    # for each path in df_train, preprocess the parquet file and save it in the output_dir_path
    print("Preprocessing train landmark files...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = list(tqdm(executor.map(
            lambda parquet_path: task(parquet_path, output_dirname=f"{new_train_landmark_files_dirname}/train",
                                      overwrite=OVERWRITE, drop_z=DROP_Z), parquet_paths_train),
            total=len(parquet_paths_train)))

    parquet_paths_val = df_val["path"].tolist()
    # for each path in df_val, preprocess the parquet file and save it in the output_dir_path
    print("Preprocessing val landmark files...")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = list(tqdm(executor.map(
            lambda parquet_path: task(parquet_path, output_dirname=f"{new_train_landmark_files_dirname}/val",
                                      overwrite=OVERWRITE, drop_z=DROP_Z), parquet_paths_val),
            total=len(parquet_paths_val)))


if __name__ == "__main__":
    main()
