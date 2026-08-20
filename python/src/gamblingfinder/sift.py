import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 739841

# Approximates methodology from Li, Longxi et al. “Identifying Gambling and Porn Websites with Image Recognition.” Pacific Rim Conference on Multimedia (2017).
# Note that SIFT keypoints rather than SURF keypoints are used, due to easier installation.
# SIFT keypoint descriptors are obtained with the following code (only 10% of keypoints are retained as per Li et al.):
# import cv2
# import numpy as np
# image = cv2.imread(
#     screenshot_file,
#     cv2.IMREAD_GRAYSCALE
# )
# sift = cv2.SIFT_create()
# keypoints, descriptors = sift.detectAndCompute(image, None)
# descriptors = descriptors.astype(np.uint8)
# num_rows = descriptors.shape[0]
# num_samples = max(1, int(num_rows * 0.1))
# sampled_indices = np.random.choice(num_rows, num_samples, replace=False)
# sampled_descriptors = descriptors[sampled_indices]

def descriptors_cluster_centers(descriptors_series: pd.Series) -> pd.DataFrame:
    descriptors = np.vstack(
        descriptors_series.apply(np.vstack)
    )
    print(descriptors.shape)

    kmeans = KMeans(n_clusters=2000, random_state=SEED).fit(descriptors)
    centers = kmeans.cluster_centers_.tolist()

    result = pd.DataFrame({
        "cluster_id": range(len(centers)),
        "center": centers
    })

    return result


def descriptors_to_histogram(descriptors: np.ndarray, sift_clusters: np.ndarray) -> list[int]:
    closest_cluster_indices = pairwise_distances_argmin(descriptors, sift_clusters)
    histogram = np.bincount(closest_cluster_indices, minlength=len(sift_clusters))
    result = histogram.tolist()
    assert len(result) == len(sift_clusters)
    return result


def train_test_classifier(X_train, y_train, X_test):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", random_state=SEED, probability=True))
    ])
    clf.fit(X_train, y_train)
    joblib.dump(clf, "data/svc_sift_gambling.joblib")

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    return y_pred_proba


def predict_from_descriptors(descriptors_series: pd.Series, sift_clusters: np.ndarray):
    histograms = descriptors_series.apply(
        lambda descriptors: descriptors_to_histogram(np.array(descriptors.tolist()), sift_clusters)
    )

    X = histograms.tolist()
    clf = joblib.load("data/svc_sift_gambling.joblib")
    y_pred_proba = clf.predict_proba(X)[:, 1]

    return y_pred_proba
