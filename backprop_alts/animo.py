from numpy.random import shuffle
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import ao_core as ao
import ao_arch as ar
from time import time
from torchvision import transforms
from utils import _prepare_for_epochs


def setup_agent():
    description = "Basic MNIST"

    arch_i = [28 * 28]
    arch_z = [4]
    arch_c = []

    connector_function = "rand_conn"
    connector_params = [392, 261, 784, 4]

    arch = ar.Arch(
        arch_i, arch_z, arch_c, connector_function, connector_params, description
    )
    agent = ao.Agent(arch, save_meta=False, _steps=1100000)
    return agent


def downsample(arr, down=200):
    f = np.vectorize(lambda x: 1 if x >= down else 0)
    return f(arr)


def label_transform(x):
    label_to_binary = np.zeros([10, 4], dtype="int8")
    for i in np.arange(10):
        label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)
    return label_to_binary[x]


def proc_labels(labels):
    # label_to_binary = np.zeros([10, 4], dtype="int8")
    # for i in np.arange(10):
    #     label_to_binary[i] = np.array(list(np.binary_repr(i, 4)), dtype=int)
    labels_z = np.zeros([labels.size, 4])
    for i in np.arange(labels.size):
        # labels_z[i] = label_to_binary[labels[i]]
        labels_z[i] = label_transform(labels[i])
    return labels_z


def run_val_ao(agent, val_loader):
    correct_count = 0
    for i, (x, z) in tqdm.tqdm(enumerate(val_loader), "Validating: "):
        for j in range(len(x)):
            agent.reset_state()
            # print(x[j])
            output = agent.next_state(
                x[j].numpy().reshape(28 * 28), DD=False, Hamming=True, unsequenced=True
            )
            if np.array_equal(output, z[j]):
                correct_count += 1

    # return epoch_accs
    return correct_count


def mnist_test_ao(batch_size=256, n_epochs=3, n_labels=10):

    details = {
        "epoch_accs": [],
        "epoch_times": [0],
        "epoch_samples": [0],
    }
    mnist, mnist_val, accs, errors = _prepare_for_epochs()
    # replace transforms to work with an ao agent
    input_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: np.array(x)),
            transforms.Lambda(lambda x: downsample(x)),
        ]
    )
    mnist.transform = input_transform
    label_lambda = transforms.Lambda(lambda x: label_transform(x))
    mnist.target_transform = label_lambda
    mnist_val.transform = input_transform
    mnist.target_transform = label_lambda

    agent = setup_agent()

    for epoch in range(n_epochs):
        train_loader = DataLoader(mnist, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=True)
        # epoch_accs = []

        epoch_accs = run_val_ao(agent, val_loader)
        print(f"Epoch {epoch} Accuracy: {epoch_accs}")

        # print(f"Epoch {epoch} Accuracy: {np.mean(epoch_accs)}")
        # details["epoch_accs"].append(np.mean(epoch_accs))

        # accs.extend(epoch_accs)

        start = time()
        # TODO: train agent here
        for i, (x, z) in tqdm.tqdm(enumerate(train_loader), "Training: "):
            reshaped = x.numpy().reshape(len(x), 28 * 28)
            agent.next_state_batch(reshaped, z, unsequenced=True)
        epoch_time = time() - start
        print(f"Epoch {epoch} Time: {epoch_time}")

        # calculate epoch accuracy for current training level, starting with no training

        pass

    epoch_accs = []
    epoch_accs = run_val_ao(agent, mnist_val)
    print(f"Final Accuracy: {epoch_accs}")
    # print(f"Final Accuracy: {np.mean(epoch_accs)}")
    # details["epoch_accs"].append(np.mean(epoch_accs))
    # accs.extend(epoch_accs)

    return accs, errors, -1, details


if __name__ == "__main__":
    n_epochs = 3
    batch_size = 256
    mnist_test_ao(batch_size, n_epochs)
