import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def preprocessing_extraction(raw_data):

    # Remove area cone where shouldn´t be
    data = raw_data[raw_data[:, 2] > -0.05]
    data = data[data[:, 2] < 0.15]

    intesity = data[:, 3]
    data = data[:, :3]

    return data, intesity


def plot_2D(xyz, labels, core_samples_mask):
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = xyz[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = xyz[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.show()

def run(raw_data, eps, minSamples):
    xyz, intensity = preprocessing_extraction(raw_data)

    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=minSamples).fit(xyz)
    labels = db.labels_

    n_noise = list(labels).count(-1)

    cloud = np.zeros((len(labels) - n_noise, 4), dtype=np.float)

    cloud[:, :3] = xyz[labels != -1]
    cloud[:, 3] = intensity[labels != -1]
    labels = labels[labels != -1]

    return cloud, labels

