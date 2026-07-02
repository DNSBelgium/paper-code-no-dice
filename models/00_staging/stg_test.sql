{{
    config(materialized="view")
}}

from {{ ref("stg_ground_truth") }}
where not is_train
