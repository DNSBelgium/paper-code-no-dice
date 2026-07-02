{{
    config(materialized="view")
}}

select * replace (label - 1 as label)
from {{ ref("stg_test") }}
where label > 0
