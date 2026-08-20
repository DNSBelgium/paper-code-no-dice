from gamblingfinder.sift import descriptors_cluster_centers

def model(dbt, _session):
    train = dbt.ref("stg_scr_train")
    descriptors = train.project("sift_keypoint_descriptors").df()["sift_keypoint_descriptors"]
    centers = descriptors_cluster_centers(descriptors)
    return centers
