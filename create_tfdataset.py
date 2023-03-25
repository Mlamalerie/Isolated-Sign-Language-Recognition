import inspect
import random
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, as_completed
from params import *
# BASE_DATASET_PATH = r"C:/Users/mlamali.saidsalimo/Downloads/Hand Gesture Recognition (HGR)/asl-signs"
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Union, Optional, Any

# Constants
FACE_N_LANDMARKS = 468
HAND_N_LANDMARKS = 21
POSE_N_LANDMARKS = 33
ROWS_PER_FRAME = FACE_N_LANDMARKS + HAND_N_LANDMARKS * 2 + POSE_N_LANDMARKS  # 543

# Generation parameters
DATA_COLUMNS = ["x", "y"]
GROUPING_BY_PARTICIPANT = True
LIMIT_PARTICIPANTS = None

VAL_SIZE = 0.1  # 10% of the data will be used for validation
NUM_SHARDS = 2  # divide the dataset into 2 shards. In general [2-10] is a good number
BATCH_SIZE = 256

OVERWRITE = False

SET_LANDMARKS = set(
    [f"face-{i:03d}" for i in range(FACE_N_LANDMARKS)] + [f"left_hand-{i:03d}" for i in range(HAND_N_LANDMARKS)] + [
        f"pose-{i:03d}" for i in range(POSE_N_LANDMARKS)] + [
        f"right_hand-{i:03d}" for i in range(HAND_N_LANDMARKS)])
SET_LANDMARKS_TYPES = {"face", "left_hand", "right_hand", "pose"}


# load the json file
def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def load_relevant_data_subset(pq_path: str, rows_per_frame: int = ROWS_PER_FRAME,
                              data_columns: list = DATA_COLUMNS) -> np.ndarray:
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.astype(np.float32)
    return data.reshape(n_frames, rows_per_frame, len(data_columns))


def split_dataset(df, val_size=0.2, random_state=42, grouping_by_participant=True):
    """Split dataset into train and validation set."""
    if grouping_by_participant:
        participant_ids = df["participant_id"].unique()
        df_trains: List[pd.DataFrame] = []
        df_vals: List[pd.DataFrame] = []
        for participant_id in participant_ids:
            df_participant = df[df["participant_id"] == participant_id]
            df_train, df_val = train_test_split(df_participant, test_size=val_size, random_state=random_state)
            df_trains.append(df_train)
            df_vals.append(df_val)

        df_train = pd.concat(df_trains, ignore_index=True)
        df_val = pd.concat(df_vals, ignore_index=True)

    else:
        df_train, df_val = train_test_split(df, test_size=val_size, random_state=random_state)
    return df_train, df_val


def tf_get_features(ftensor):
    def feat_wrapper(ftensor):
        return load_relevant_data_subset(ftensor.numpy().decode('utf-8'))

    return tf.py_function(
        feat_wrapper,
        [ftensor],
        Tout=tf.float32
    )


def set_shape(x):
    # None dimensions can be of any length
    # ensure_shape will raise an error if the shape is not as expected
    return tf.ensure_shape(x, (None, ROWS_PER_FRAME, len(DATA_COLUMNS)))  # (None, 468, 3)


def create_ds(df: pd.DataFrame, sign_ids: dict, batch_size: int) -> tf.data.Dataset:
    X_ds = tf.data.Dataset.from_tensor_slices(
        df.path.values  # start with a dataset of the parquet paths
    ).map(
        tf_get_features  # load individual sequences
    ).map(
        set_shape  # set and enforce element shape
    ).apply(
        tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size)  # apply batching function
    )

    # load and batch the labels
    y_ds = tf.data.Dataset.from_tensor_slices(
        df.sign.map(sign_ids).values.reshape(-1, 1)
    ).batch(batch_size)

    # zip the features and labels
    return tf.data.Dataset.zip((X_ds, y_ds))


# Sharding could be improved, as the distribution of elements in different shards should optimally be equal.
# Currently, it will be a sample from a uniform distribution because this is simple to implement
def shard_func(*_):
    return tf.random.uniform(shape=[], maxval=NUM_SHARDS, dtype=tf.int64)


# generate metadata file, with parameters
def generate_metadata_file(output_dir: str, sign_ids: dict, batch_size: int, data_columns: list, num_shards: int,
                           val_size: float, len_train: int, len_val: int,
                           grouping_by_participant: bool, limit_participants: int):
    # Créez le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Enregistrez les paramètres du dataset dans un fichier JSON
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump({
            "batch_size": batch_size,
            "sign_ids": sign_ids,
            "data_columns": data_columns,
            "num_shards": num_shards,
            "len_train": len_train,
            "len_val": len_val,
            "val_size": val_size,
            "grouping_by_participant": grouping_by_participant,
            "limit_participants": limit_participants,
        }, f)

    print("> Le fichier de métadonnées a été enregistré.")


def get_participant_ids(dirpath: str, limit_participants: int = None):
    participant_ids = os.listdir(dirpath)
    random.seed(42)
    random.shuffle(participant_ids)
    if limit_participants:
        participant_ids = participant_ids[:limit_participants]
    return [int(participant_id) for participant_id in participant_ids]


def main():
    new_landmark_files_dirname = f'landmark_files_preprocessed__{"".join(DATA_COLUMNS)}__ids_{LIMIT_PARTICIPANTS or "all"}__val_{VAL_SIZE}'
    new_landmark_files_dirpath = f"{BASE_DATASET_PATH}/{new_landmark_files_dirname}"
    print("-" * 80)
    print(f"* Output dir: {new_landmark_files_dirpath}")
    print(f"* Limit participants: {LIMIT_PARTICIPANTS}")
    print(f"* Data columns: {DATA_COLUMNS}")
    print(f"* Batch size: {BATCH_SIZE}")
    print(f"* Validation size: {VAL_SIZE}")
    print(f"* Number of shards: {NUM_SHARDS}")
    print("-" * 80)

    # sign to prediction index map
    sign_to_prediction_index_map_path = f"{BASE_DATASET_PATH}/sign_to_prediction_index_map.json"
    s2p_map = load_json_file(sign_to_prediction_index_map_path)

    # load train.csv
    train_path = f"{BASE_DATASET_PATH}/train.csv"
    df_train_full = pd.read_csv(train_path).sort_values(by=["participant_id", "sign", "sequence_id"]).reset_index(
        drop=True)

    # limit number of participants, get random participants id
    participant_ids = get_participant_ids(f"{BASE_DATASET_PATH}/train_landmark_files",
                                          limit_participants=LIMIT_PARTICIPANTS)

    df_train_full = df_train_full[df_train_full["participant_id"].isin(participant_ids)]

    # add base path to parquet files
    df_train_full["path"] = df_train_full["path"].apply(lambda parquet_path: f"{BASE_DATASET_PATH}/{parquet_path}")

    # split dataset
    df_train, df_val = split_dataset(df_train_full, val_size=VAL_SIZE, random_state=12,
                                     grouping_by_participant=GROUPING_BY_PARTICIPANT)
    print(f"> Train dataset size: {len(df_train)}")
    print(f"> Validation dataset size: {len(df_val)}")
    print(f"> Number of classes: {len(s2p_map)}")
    print("-" * 80)

    # train dataset
    train_ds = create_ds(df_train, s2p_map, batch_size=BATCH_SIZE)
    train_example = next(iter(train_ds))
    print(f"> train example shape: {train_example[0].shape}")
    ## save train dataset
    save_train_ds_path = f"{new_landmark_files_dirpath}/trainDataset"
    train_ds.prefetch(tf.data.AUTOTUNE).save(save_train_ds_path, shard_func=shard_func)  # save dataset
    train_ds_cardinality = tf.data.experimental.cardinality(train_ds)
    train_ds_size = train_ds_cardinality.numpy() * BATCH_SIZE
    print(f">>> Train tf.dataset saved. Cardinality: {train_ds_cardinality}, size: {train_ds_size}.")

    # validation dataset
    val_ds = create_ds(df_val, s2p_map, batch_size=BATCH_SIZE)
    val_example: tf.data.Dataset = next(iter(val_ds))
    print(f"> validation example shape: {val_example[0].shape}")
    ## save validation dataset
    save_val_ds_path = f"{new_landmark_files_dirpath}/valDataset"
    val_ds.prefetch(tf.data.AUTOTUNE).save(save_val_ds_path, shard_func=shard_func)
    val_ds_cardinality = tf.data.experimental.cardinality(val_ds)
    val_ds_size = val_ds_cardinality.numpy() * BATCH_SIZE
    print(f">>> Validation tf.dataset saved. Cardinality: {val_ds_cardinality}, size: {val_ds_size}.")

    # generate metadata file
    generate_metadata_file(new_landmark_files_dirpath, sign_ids=s2p_map, batch_size=BATCH_SIZE,
                           data_columns=DATA_COLUMNS, num_shards=NUM_SHARDS, val_size=VAL_SIZE, len_train=len(df_train), len_val=len(df_val),
                           grouping_by_participant=GROUPING_BY_PARTICIPANT, limit_participants=LIMIT_PARTICIPANTS)


if __name__ == "__main__":
    main()
