# Isolated Sign Language Recognition

Kaggle Competition: [link](https://www.kaggle.com/competitions/asl-signs/overview)

## Goal of the Competition

The goal of this competition is to classify isolated American Sign Language (ASL) signs. You will create a TensorFlow
Lite model trained on labeled landmark data extracted using the MediaPipe Holistic Solution.

## Dataset Description

Deaf children are often born to hearing parents who do not know sign language. Your challenge in this competition is to
help identify signs made in processed videos, which will support the development of mobile apps to help teach parents
sign language so they can communicate with their Deaf children.

This competition requires submissions to be made in the form of TensorFlow Lite models. You are welcome to train your
model using the framework of your choice as long as you convert the model checkpoint into the tflite format prior to
submission. Please see the evaluation page for details.

### Files

**train_landmark_files/[participant_id]/[sequence_id].parquet** The landmark data. The landmarks were extracted from raw
videos with the [MediaPipe holistic model](https://google.github.io/mediapipe/solutions/holistic.html). Not all of the
frames necessarily had visible hands or hands that could be
detected by the model.

Landmark data should not be used to identify or re-identify an individual. Landmark data is not intended to enable any
form of identity recognition or store any unique biometric identification.

- `frame` - The frame number in the raw video.
- `row_id` - A unique identifier for the row.
- `type` - The type of landmark. One of `['face', 'left_hand', 'pose', 'right_hand']`.
- `landmark_index` - The landmark index number. Details of the hand landmark locations can be
  found [here](https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model).
- `[x/y/z]` - The normalized spatial coordinates of the landmark. These are the only columns that will be provided to
  your submitted model for inference. The MediaPipe model is not fully trained to predict depth so you may wish to ignore the
  z values.

_**train.csv**_

- `path` - The path to the landmark file.
- `participant_id` - A unique identifier for the data contributor.
- `sequence_id` - A unique identifier for the landmark sequence.
- `sign` - The label for the landmark sequence.

## Author

- Mlamali SAID SALIMO

