import numpy as np


def k_means(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    new_centroids = np.zeros((k, 4))
    clusters = [[] for _ in range(k)]

    while not np.all(centroids == new_centroids):
        distances = np.sqrt(
            np.sum((data - centroids[:, np.newaxis]) ** 2, axis=2)
        )
        labels = np.argmin(distances, axis=0)

        new_centroids = np.array(
            [data[labels == i].mean(axis=0) for i in range(k)]
        )

        centroids = new_centroids

    for i, label in enumerate(labels):
        clusters[label].append(data[i])

    return (centroids, labels + 1, clusters)


def k_means_predict(data, centroids):
    distances = np.sqrt(
        np.sum(np.square(data - centroids[:, np.newaxis]), axis=2)
    )
    labels = np.argmin(distances, axis=0)
    clusters = [[] for _ in range(len(centroids))]

    for i, label in enumerate(labels):
        clusters[label].append(data[i])

    return (distances, labels + 1, clusters)


def knn(test_data, train_data, k):
    distances = np.sqrt(
        np.sum(
            np.square(
                test_data[:, np.newaxis, :] - train_data[np.newaxis, :, :]
            ),
            axis=2,
        )
    )

    return np.argpartition(distances, k, axis=1)[:, :k]
