{{
    config(materialized="table")
}}

with

test_data as (
    from {{ ref("stg_test") }}
),

gambling_preds as (
    from {{ ref("lr_g_train_test") }}
),

linksite_preds as (
    from {{ ref("lr_l_train_test") }}
),

chained as (
    select
        test_data.sample_id,
        test_data.label,
        if(
            gambling_preds.prediction >= 0.5,
            if(
                linksite_preds.prediction >= 0.5,
                2,
                1
            ),
            0
        ) as prediction
    from test_data
    join gambling_preds using (sample_id)
    left join linksite_preds using (sample_id)
)

from chained
