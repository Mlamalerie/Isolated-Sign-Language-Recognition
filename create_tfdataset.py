import random
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

FACE_N_LANDMARKS = 468
HAND_N_LANDMARKS = 21
POSE_N_LANDMARKS = 33
ROWS_PER_FRAME = FACE_N_LANDMARKS + HAND_N_LANDMARKS * 2 + POSE_N_LANDMARKS  # 543

NUM_SHARDS = 2
BATCH_SIZE = 256
DATA_COLUMNS = ["x", "y"]

N_WORKERS = 15
OVERWRITE = False
DROP_Z = True  # keep only x and y columns
LIMIT_PARTICIPANTS = 10

SET_LANDMARKS = set(
    [f"face-{i:03d}" for i in range(FACE_N_LANDMARKS)] + [f"left_hand-{i:03d}" for i in range(HAND_N_LANDMARKS)] + [
        f"pose-{i:03d}" for i in range(POSE_N_LANDMARKS)] + [
        f"right_hand-{i:03d}" for i in range(HAND_N_LANDMARKS)])
SET_LANDMARKS_TYPES = {"face", "left_hand", "right_hand", "pose"}


# load the json file
def load_json_file(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return json.load(f)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def to_serialized_example(df: pd.DataFrame, sign: str, participant_id: int, sequence_id: int):
    """Convert dataframe to tfrecord example."""
    # create example
    example = tf.train.Example(features=tf.train.Features(feature={
        "sign": _bytes_feature(sign.encode()),
        "participant_id": _int64_feature(participant_id),
        "sequence_id": _int64_feature(sequence_id),
        "landmarks": _bytes_feature(df.to_msgpack(compress="zlib")),
    }))
    return example.SerializeToString()


def load_relevant_data_subset(pq_path: str, data_columns: list = DATA_COLUMNS) -> np.ndarray:
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.astype(np.float32)
    return data.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))


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
    if not set(df["type"].unique()) == SET_LANDMARKS_TYPES:
        return False, "Invalid landmarks types"
    if not set(row_id_unique_names) == SET_LANDMARKS:
        return False, "Invalid number of landmarks"
    if not n_rows % len(SET_LANDMARKS) == 0:
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
        participant_ids = df["participant_id"].unique()
        df_trains : List[pd.DataFrame] = []
        df_vals : List[pd.DataFrame] = []
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
    return tf.ensure_shape(x, (None, ROWS_PER_FRAME, len(DATA_COLUMNS))) # (None, 468, 3)


def create_ds(df: pd.DataFrame, sign_ids: dict, batch_size: int):
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

# save dataset to tfrecord
def save_dataset(ds: tf.data.Dataset, output_dir: str):
    # Créez le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Créez un itérateur pour parcourir le dataset
    iterator = iter(ds)


    # Parcourez le dataset et enregistrez chaque élément dans un fichier TFRecord
    file_index = 0
    while True:
        try:
            # Obtenez le prochain élément du dataset
            example = next(iterator)

            # Générez le nom de fichier pour cet exemple
            filename = f"{output_dir}/data_{file_index}.tfrecord"

            # Enregistrez l'exemple dans le fichier
            with tf.io.TFRecordWriter(filename) as writer:
                writer.write(example.numpy())

            # Incrémentez l'index de fichier
            file_index += 1
        except StopIteration:
            break

    print(f"Le dataset a été enregistré dans {file_index} fichiers TFRecord.")


# Sharding could be improved, as the distribution of elements in different shards should optimally be equal.
# Currently, it will be a sample from a uniform distribution because this is simple to implement
def shard_func(*_):
    return tf.random.uniform(shape=[], maxval=NUM_SHARDS, dtype=tf.int64)

def main():

    new_train_landmark_files_dirname = f'train_landmark_files_preprocessed_{"xyz" if not DROP_Z else "xy"}_{LIMIT_PARTICIPANTS if LIMIT_PARTICIPANTS else "all"}'

    for dir_to_create in [f"{new_train_landmark_files_dirname}/train", f"{new_train_landmark_files_dirname}/val"]:
        os.makedirs(f"{BASE_DATASET_PATH}/{dir_to_create}", exist_ok=True)

    print(f"* Number of workers: {N_WORKERS}")
    print(f"* Overwrite: {OVERWRITE}")
    print(f"* Drop z: {DROP_Z}")
    print(f"* Output dir: {BASE_DATASET_PATH}/{new_train_landmark_files_dirname}")

    sign_to_prediction_index_map_path = f"{BASE_DATASET_PATH}/sign_to_prediction_index_map.json"
    s2p_map = load_json_file(sign_to_prediction_index_map_path)
    p2s_map = {v: k for k, v in s2p_map.items()}

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

    # train dataset
    train_ds = create_ds(df_train, s2p_map, batch_size=BATCH_SIZE)
    save_train_ds_path = f"{BASE_DATASET_PATH}/{new_train_landmark_files_dirname}/trainDataset"
    train_ds.prefetch(tf.data.AUTOTUNE).save(save_train_ds_path, shard_func=shard_func) # save dataset
    print(f"> Train dataset saved to {save_train_ds_path} with {len(parquet_paths_train)} elements.")

    # validation dataset
    val_ds = create_ds(df_val, s2p_map, batch_size=BATCH_SIZE)
    save_val_ds_path = f"{BASE_DATASET_PATH}/{new_train_landmark_files_dirname}/valDataset"
    val_ds.prefetch(tf.data.AUTOTUNE).save(save_val_ds_path, shard_func=shard_func) # save dataset
    print(f"> Validation dataset saved to {save_val_ds_path} with {len(df_val)} elements.")


if __name__ == "__main__":
    main()
