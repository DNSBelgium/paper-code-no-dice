{{
    config(materialized="view")
}}

from {{ ref("stg_scr_ground_truth") }}
where is_train
