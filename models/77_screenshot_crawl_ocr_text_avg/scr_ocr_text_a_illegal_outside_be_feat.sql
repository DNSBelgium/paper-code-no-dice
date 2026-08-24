{{
    config(materialized="view")
}}

with

features as (
    from {{ ref("stg_scr_illegal_outside_be") }}
),

avg_emb as (
    select
        sample_id,
        {% for i in range(1, 385) %}
            (f_ocr_emb_{{ i }} + f_emb_{{ i }}) / 2 as f_avg_emb_{{ i }},
        {% endfor %}
    from features
)

from avg_emb
