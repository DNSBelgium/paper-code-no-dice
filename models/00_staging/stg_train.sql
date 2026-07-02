{{
    config(materialized="view")
}}

from {{ ref("stg_ground_truth") }}
where is_train
