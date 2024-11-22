from numpy.random import shuffle
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import ao_core as ao
import ao_arch as ar
from time import time
from torchvision import extension, transforms, datasets
from pathlib import Path


def download_data():
    datasets.MNIST("../data", train=True, download=True)
    datasets.MNIST("../data", train=False, download=True)


def prepare_data():
    filenames = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]

    folder = "./data/MNIST/raw/"

    # if not Path(folder + filenames[0]).exists():
    #     _prepare_for_epochs()
    # if not Path(folder + filenames[1]).exists():
    #     _prepare_for_epochs()
    # if not Path(folder + filenames[2]).exists():
    #     _prepare_for_epochs()
    # if not Path(folder + filenames[3]).exists():
    #     _prepare_for_epochs()
    for name in filenames:
        if not Path(folder + name).exists():
            download_data()

    train_in = np.fromfile(folder + filenames[0], dtype="uint8", offset=16)
    train_in = train_in.reshape([60000, 28, 28])
    train_labels = np.fromfile(folder + filenames[1], dtype="uint8", offset=8)
    test_in = np.fromfile(folder + filenames[2], dtype="uint8", offset=16)
    test_in = test_in.reshape([10000, 28, 28])
    test_labels = np.fromfile(folder + filenames[3], dtype="uint8", offset=8)
    return (train_in, train_labels), (test_in, test_labels)


def setup_agent():
    description = "Basic MNIST"

    arch_i = [8 for _ in range(28 * 28)]
    arch_z = [4]
    arch_c = []

    connector_function = "rand_conn"
    connector_params = [4000, 1500, (8 * 28 * 28), 4]

    arch = ar.Arch(
        arch_i, arch_z, arch_c, connector_function, connector_params, description
    )
    agent = ao.Agent(arch, save_meta=False, _steps=1200000)
    return agent


def downsample(arr, down=200):
    f = np.vectorize(lambda x: 1 if x >= down else 0)
    return f(arr)


def proc_input(image):
    if image.ndim == 1:
        return np.array(
            [np.array(list(format(pixel, "08b")), dtype=np.uint8) for pixel in image]
        )
    else:
        return np.array([proc_input(sub_array) for sub_array in image])


def label_transform(x):
    label_to_binary = np.zeros([10, 4], dtype="int8")
    for i in np.arange(10):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)
    return label_to_binary[x]


def process_data():
    # data = unpickle()
    (mnist_train, mnist_train_labels), (mnist_val, mnist_val_labels) = prepare_data()
    # mnist_train = downsample(mnist_train)
    # mnist_val = downsample(mnist_val)
    mnist_train = proc_input(mnist_train)
    mnist_val = proc_input(mnist_val)
    mnist_train_z = proc_labels(mnist_train_labels)
    mnist_val_z = proc_labels(mnist_val_labels)
    return (mnist_train, mnist_train_z), (mnist_val, mnist_val_z)


def proc_labels(labels):
    label_to_binary = np.zeros([10, 4], dtype="int8")
    for i in np.arange(10):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)
    labels_z = np.zeros([labels.size, 4])
    for i in np.arange(labels.size):
        labels_z[i] = label_to_binary[labels[i]]
        # labels_z[i] = label_transform(labels[i])
    return labels_z


def paired_shuffle(arr_1, arr_2):
    assert len(arr_1) == len(arr_2)
    p = np.random.permutation(len(arr_1))
    return arr_1[p], arr_2[p]


def run_val_ao(agent, mnist_inputs, mnist_z):
    correct_count = 0
    for j in tqdm.trange(len(mnist_inputs)):
        agent.reset_state()
        # print(x[j])
        output = -1
        for _ in range(3):
            output = agent.next_state(
                mnist_inputs[j].reshape(8 * 28 * 28),
                DD=False,
                Hamming=True,
                unsequenced=True,
            )
        # print(output)
        if np.array_equal(output, mnist_z[j]):
            correct_count += 1

    # return epoch_accs
    num_items = len(mnist_inputs)
    return correct_count / num_items


def mnist_test_ao(n_epochs=3):

    details = {
        "epoch_accs": [],
        "epoch_times": [0],
        "epoch_samples": [0],
    }
    accs = []
    errors = []
    (mnist_train, mnist_train_z), (mnist_val, mnist_val_z) = process_data()

    agent = setup_agent()

    for epoch in range(n_epochs):
        mnist_train, mnist_train_z = paired_shuffle(mnist_train, mnist_train_z)
        mnist_val, mnist_val_z = paired_shuffle(mnist_val, mnist_val_z)
        # epoch_accs = []

        # calculate epoch accuracy for current training level, starting with no training
        epoch_accs = run_val_ao(agent, mnist_val, mnist_val_z)
        print(f"Epoch {epoch} Accuracy: {epoch_accs}")
        details["epoch_accs"].append(epoch_accs)

        accs.append(epoch_accs)

        # print(f"Epoch {epoch} Accuracy: {np.mean(epoch_accs)}")
        # details["epoch_accs"].append(np.mean(epoch_accs))

        # accs.extend(epoch_accs)

        start = time()
        reshaped = mnist_train.reshape(len(mnist_train), 8 * 28 * 28)
        agent.next_state_batch(reshaped, mnist_train_z, unsequenced=True)
        epoch_time = time() - start
        print(f"Epoch {epoch} Time: {epoch_time}")

        details["epoch_times"].append(epoch_time)
        details["epoch_samples"].append(len(mnist_train))

    mnist_val, mnist_val_z = paired_shuffle(mnist_val, mnist_val_z)
    epoch_accs = run_val_ao(agent, mnist_val, mnist_val_z)
    print(f"Final Accuracy: {epoch_accs}")
    details["epoch_accs"].append(epoch_accs)
    accs.append(epoch_accs)

    # print(f"Final Accuracy: {np.mean(epoch_accs)}")
    # details["epoch_accs"].append(np.mean(epoch_accs))
    # accs.extend(epoch_accs)

    return accs, errors, -1, details


if __name__ == "__main__":
    n_epochs = 3
    # batch_size = 256
    # batch_size = 1000
    _, _, _, details = mnist_test_ao(n_epochs)
    print(details)
