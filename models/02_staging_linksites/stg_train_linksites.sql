{{
    config(materialized="view")
}}

select * replace (label - 1 as label)
from {{ ref("stg_train") }}
where label > 0
