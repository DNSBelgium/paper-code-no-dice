{{
    config(materialized="view")
}}

from {{ ref("stg_doc2vec_ground_truth") }}
where is_train
