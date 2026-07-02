{{
    config(materialized="view")
}}

from {{ ref("stg_doc2vec_ground_truth") }}
where not is_train
