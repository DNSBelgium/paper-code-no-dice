from gamblingfinder.sift import predict_from_descriptors
import numpy as np

def model(dbt, _session):
    _ = dbt.ref("scr_sift_train_test")

    sift_clusters = dbt.ref("scr_sift_clusters")
    sift_clusters = np.array(sift_clusters.sort("cluster_id").df()["center"].tolist())

    illegal_outside_be = dbt.ref("stg_scr_illegal_outside_be")
    illegal_outside_be = illegal_outside_be.project("sample_id, sift_keypoint_descriptors").df()

    y_pred = predict_from_descriptors(illegal_outside_be["sift_keypoint_descriptors"], sift_clusters)

    illegal_outside_be.drop(columns=["sift_keypoint_descriptors"], inplace=True)
    illegal_outside_be["prediction"] = y_pred

    print("Recall: {:.2%}".format((y_pred >= 0.5).sum() / len(y_pred)))

    return illegal_outside_be
