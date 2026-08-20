from gamblingfinder.sift import descriptors_to_histogram
import numpy as np


def model(dbt, _session):
    sift_clusters = dbt.ref("scr_sift_clusters")
    sift_clusters = np.array(sift_clusters.sort("cluster_id").df()["center"].tolist())

    ground_truth = dbt.ref("stg_scr_ground_truth")
    ground_truth = ground_truth.project(
        "sample_id, label, is_train, sift_keypoint_descriptors"
    ).df()

    sift_histograms = ground_truth["sift_keypoint_descriptors"].apply(
        lambda descriptors: descriptors_to_histogram(
            np.array(descriptors.tolist()), sift_clusters
        )
    )

    ground_truth.drop(columns=["sift_keypoint_descriptors"], inplace=True)
    ground_truth["sift_histogram"] = sift_histograms
    return ground_truth
