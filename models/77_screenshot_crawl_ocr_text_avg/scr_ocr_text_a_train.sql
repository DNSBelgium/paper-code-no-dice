{{
    config(materialized="view")
}}

with

train as (
    from {{ ref("stg_scr_train") }}
),

avg_emb as (
    select
        sample_id,
        label,
        {% for i in range(1, 385) %}
            (f_ocr_emb_{{ i }} + f_emb_{{ i }}) / 2 as f_avg_emb_{{ i }},
        {% endfor %}
    from train
)

from avg_emb
