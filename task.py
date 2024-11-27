import os
import keras
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from time import localtime, strftime


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
        print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Loaded Federated Dataset. Type: {type(fds)}")
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Created Partitions. Type: {type(partition)}")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    print(f"{strftime("%Y-%m-%d %H:%M:%S", localtime())} Dataset Format. \nX_Train Type: {type(x_train)} Length: {len(x_train)}")
    print(f"Dataset Format. \nY_Train Type: {type(y_train)} Length: {len(y_train)}")
    print(f"Dataset Format. \nX_Test Type: {type(x_test)} Length: {len(x_test)}")
    print(f"Dataset Format. \nY_Test Type: {type(y_test)} Length: {len(y_test)}")

    return x_train, y_train, x_test, y_test