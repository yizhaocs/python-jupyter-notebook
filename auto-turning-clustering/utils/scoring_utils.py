import sklearn


def silhouette_score(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        score = sklearn.metrics.silhouette_score(X, cluster_labels)
    return score


def calinski_harabasz_score(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        score = sklearn.metrics.calinski_harabasz_score(X, cluster_labels)
    return score


def davies_bouldin_score(estimator, X):
    estimator.fit(X)
    cluster_labels = estimator.labels_
    num_labels = len(set(cluster_labels))
    num_samples = X.shape[0]
    if num_labels == 1 or num_labels == num_samples:
        return -1
    else:
        score = sklearn.metrics.davies_bouldin_score(X, cluster_labels)
    return score