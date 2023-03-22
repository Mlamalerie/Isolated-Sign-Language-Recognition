from params import *
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import mediapipe as mp
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import numpy as np
import random
from threading import current_thread
from threading import get_ident
from threading import get_native_id


def draw_frame(frame, df_landmark, sign, fig):
    lms = [lm for _, lm in df_landmark.query("frame == @frame").groupby("type")]
    artists = []
    # add sup title
    fig.suptitle(f"Sign: '{sign}' - Frame: {frame}", fontsize=10)
    for idx_lm, lm in enumerate(lms):
        ax = fig.add_subplot(2, 2, idx_lm + 1)
        first_row_type = lm.type.values[0]
        title = f"landmarks: {first_row_type}"

        lm["y_"] = lm["y"] * -1  # invert y axis

        if first_row_type == "face":
            ax.set_title(title)
            artist = ax.scatter(lm["x"], lm["y_"])
            artists.append(artist)

        elif first_row_type in ["left_hand", "right_hand", "pose"]:
            ax.set_title(title)
            artist = ax.scatter(lm["x"], lm["y_"])
            artists.append(artist)
            connections = mp.solutions.hands.HAND_CONNECTIONS if not first_row_type == "pose" else mp.solutions.pose.POSE_CONNECTIONS
            for connection in connections:
                point_a = connection[0]
                point_b = connection[1]
                x1, y1 = lm.query("landmark_index == @point_a")[["x", "y_"]].values[0]
                x2, y2 = lm.query("landmark_index == @point_b")[["x", "y_"]].values[0]
                artist = ax.plot([x1, x2], [y1, y2], color="grey")[0]
                artists.append(artist)
                # add landmark index
                if not np.isnan(x1) and not np.isnan(y1):
                    artist = ax.text(x1, y1, point_a, fontsize=7, ha="right", va="bottom")
                    artists.append(artist)

        else:
            print("Unknown type")
    return artists


def make_animation(df_train: pd.DataFrame, sign: str, participant_id: int, sequence_id_idx: int = 0, fps: int = 60,
                   output_dir_path: str = None, overwrite=False) -> None:
    """ Make animation from the landmark dataframe.

    Args:
        df_train (pd.DataFrame): dataframe with the landmarks, containing the columns "sign", "participant_id", "sequence_id" and "path"
        sign (str): sign name
        participant_id (str): participant id
        fps (int, optional): fps of the animation. Defaults to 60.
    """

    # worker thread details
    #thread = current_thread()
    #print(f'> Worker thread: name={thread.name}...', end=' ')

    # check if sign is valid
    if sign not in df_train.sign.unique().tolist():
        raise ValueError(f"sign {sign} is not valid")
    # check if participant_id is valid
    if participant_id not in df_train.participant_id.unique().tolist():
        raise ValueError(f"participant_id {participant_id} is not valid")

    rows_sequences = df_train.query("sign == @sign and participant_id == @participant_id")
    if sequence_id_idx >= len(rows_sequences):
        raise ValueError(f"RangeError: sequence_id_idx {sequence_id_idx} is out of range {len(rows_sequences)}")

    row_sequence = rows_sequences.iloc[sequence_id_idx]
    sequence_id = row_sequence.sequence_id

    output_dir_path = output_dir_path or os.path.join(os.getcwd(), "animations")
    output_dir_path = os.path.join(output_dir_path, sign)
    gif_file_name = f"{sign}_{participant_id}_{sequence_id}_{fps}.gif"
    output_full_path_gif = os.path.join(output_dir_path, gif_file_name)

    if os.path.exists(output_full_path_gif) and not overwrite:
        # print(f"animation already exists")
        return output_full_path_gif

    df_landmark_sequence = pd.read_parquet(f"{BASE_DATASET_PATH}/{row_sequence.path}")

    # * ANIMATION *

    frames_list = df_landmark_sequence.frame.unique().tolist()  # list of frames

    fig = plt.figure(figsize=(10, 10))
    # draw_frame(example_landmark,fig, edges, sign="Shhh", frame=frame_focused)

    draw_frame_bis = partial(draw_frame, df_landmark=df_landmark_sequence, fig=fig, sign=sign)

    try:
        ani = animation.FuncAnimation(fig, draw_frame_bis, frames=frames_list, interval=1000 / fps,
                                      blit=True, repeat=False)
    except Exception as e:
        print(f"Error in animation: {type(e)} - {e}")
        return None

    # check if this dir exists
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # save animation
    ani.save(output_full_path_gif, writer="pillow", fps=fps)

    plt.close(fig)

    #print(f"animation saved at {output_full_path_gif}/")

    return output_full_path_gif


def main_seq():
    train_path = f"{BASE_DATASET_PATH}/train.csv"
    df_train = pd.read_csv(train_path).sort_values(by=["participant_id", "sign", "sequence_id"]).reset_index(drop=True)

    print(f"Number of rows (files): {df_train.shape[0]}")
    print(f"Columns: {df_train.columns}")

    signs_to_gif: list = df_train.sign.unique().tolist()

    participants: list = df_train.participant_id.unique().tolist()
    random.shuffle(participants)

    participants_to_gif: list = participants[:5]
    print(f"Participants: {participants_to_gif}")
    MAX_SEQUENCES_PER_PARTICIPANT = 2

    # use multithreading and check completed tasks with tqdm
    # (sign, participant_id, idx_sequence, fps,output_dir, overwrite)
    args = [(sign, participant_id, idx_sequence, 60, None, False) for sign in signs_to_gif for participant_id in
            participants_to_gif for idx_sequence in range(MAX_SEQUENCES_PER_PARTICIPANT)]

    for arg in tqdm(args):
        print(arg)
        make_animation(df_train, *arg)


def main_parallelized():
    train_path = f"{BASE_DATASET_PATH}/train.csv"
    df_train = pd.read_csv(train_path).sort_values(by=["participant_id", "sign", "sequence_id"]).reset_index(drop=True)

    print(f"Number of rows (files): {df_train.shape[0]}")
    print(f"Columns: {df_train.columns}")

    signs_to_gif: list = df_train.sign.unique().tolist()

    participants: list = df_train.participant_id.unique().tolist()
    random.shuffle(participants)

    participants_to_gif: list = participants[:2]
    print(f"Participants: {participants_to_gif}")
    MAX_SEQUENCES_PER_PARTICIPANT = 10

    # create a list of arguments to pass to the make_animation function
    args = [(sign, participant_id, idx_sequence, 60, None, False) for sign in signs_to_gif for participant_id in
            participants_to_gif for idx_sequence in range(MAX_SEQUENCES_PER_PARTICIPANT)]

    # main thread details
    #thread = current_thread()
    #print(f'Main thread: name={thread.name}, idnet={get_ident()}, id={get_native_id()}')

    N_WORKERS = 10
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        # submit tasks to executor
        results = list(tqdm(executor.map(lambda x: make_animation(df_train, *x), args), total=len(args)))


if __name__ == "__main__":
    # set seed
    random.seed(1234)
    main_parallelized()
