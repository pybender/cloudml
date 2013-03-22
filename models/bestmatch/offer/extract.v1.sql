-- Auxiliary functions
create or replace function json_get(input text, path text)
  returns text
as $$
  import json
  from jsonpath import jsonpath
  try:
    input_json = json.loads(input)
    m = jsonpath(input_json, path)
    return ','.join(map(str, m)) if m else None
  except Exception as e:
    return None
$$ language plpythonu;

drop type if exists contractor cascade;
create type contractor as (
  dev_eng_skill float,
  dev_adj_score float,
  dev_portfolio_items_count integer,
  tsexams numeric[],
  skills text[],
  dev_country text,
  dev_region text,
  dev_total_hours float
);

create or replace function parse_contractor(contractor_info text)
  returns contractor
as $$
  import json
  from jsonpath import jsonpath
  try:
    input_json = json.loads(contractor_info)
  except Exception as e:
    input_json = dict() 
  def false_to_array(m): return [] if not m else m
  contractor = dict()
  contractor['dev_eng_skill'] = input_json.get('dev_eng_skill', 0)
  contractor['dev_adj_score'] = input_json.get('dev_adj_score', 0)
  contractor['dev_portfolio_items_count'] = input_json.get('dev_porfolio_items_count', 0)
  contractor['tsexams'] = false_to_array(jsonpath(input_json, '$.tsexams.*.ts_ref'))
  contractor['skills'] = false_to_array(jsonpath(input_json, '$.skills.*.skl_name'))
  contractor['dev_country'] = input_json.get('dev_country', None)
  contractor['dev_region'] = input_json.get('dev_region', None)
  contractor['dev_total_hours'] = input_json.get('dev_total_hours', 0)
  return contractor
$$ language plpythonu;

create or replace function my_array_intersect(anyarray, anyarray)
  returns anyarray
  language sql
as $function$
  select array(
    select unnest($1)
    intersect
    select unnest($2)
  );
$function$;

-- Dataset selection
create table match.bestmatch_dataset as
select
  case when ja.offermc_comp_reply = 'accept' then 1 else 0 end as made_offer,
  qi.application,
--  qi.employer_info, 
--  qi.contractor_info, 
--  qi.agency_info,

  oo."Opening Title" as opening_title,
  oo."Job Description" as opening_description,
  oo."Required Skills" as opening_skills,
  o.type as opening_type,
  o.relatedjobcategory,
  jc."Level2" as opening_category,
  jc."Segment" as opening_segment,
  oo."Amount" as opening_budget,
  o.is_fjp,
  o.pref_cnt,

  ja.is_ic,
  ja.amount as bid_amount,
  ja.hr_charge_rate as bid_rate,
 
  oo."PrefAvailableHoursPerWeek" as pref_available_hours,
  oo."PrefEnglishSkill" as pref_english,
  oo."PrefFeedbackScore" as pref_feedback,
  oo."PrefGroup" as pref_group,
  oo."PrefHasPortfolio" as pref_has_portfolio,
  oo."PrefHourlyRateMin" as pref_rate_min,
  oo."PrefHourlyRateMax" as pref_rate_max,
  oo."PrefLocationRegion" as pref_region_id,
  r."Region" as pref_region,
  oo."PrefTest" as pref_test,
  oo."PrefoDeskHours" as pref_odesk_hours,
  so.candidate_type_pref as pref_candidate_type,

  (con).dev_eng_skill >= oo."PrefEnglishSkill" as matches_pref_english,
  (con).dev_adj_score >= oo."PrefFeedbackScore" as matches_pref_feedback,
  not oo."PrefHasPortfolio" or 
    oo."PrefHasPortfolio" and (con).dev_portfolio_items_count > 0 as matches_pref_portfolio,
  oo."PrefHourlyRateMin" is null or 
    oo."PrefHourlyRateMin" = 0 and oo."PrefHourlyRateMax" = 0 or  
    ja.hr_charge_rate >= oo."PrefHourlyRateMin" and ja.hr_charge_rate <= oo."PrefHourlyRateMax" as matches_pref_rate,
  oo."PrefLocationRegion" = 0 or r."Region" = (con).dev_region  as matches_pref_region,
  oo."PrefTest" = 0 or my_array_intersect((con).tsexams, array[oo."PrefTest"]) is not null as matches_pref_test,
  (con).dev_total_hours >= oo."PrefoDeskHours" as matches_pref_odesk_hours,
  so.candidate_type_pref = 'all' or ja.is_ic and so.candidate_type_pref = 'individuals' or not ja.is_ic and so.candidate_type_pref = 'agencies' as matches_pref_ic_ac,

  json_get(employer_info, '$.op_country') || ',' || (con).dev_country as employer_contractor_countries,
  array_to_string(my_array_intersect((con).skills, string_to_array(oo."Required Skills", ',')), ',') as matched_skills,
  coalesce(array_upper(my_array_intersect((con).skills, string_to_array(oo."Required Skills", ',')), 1), 0)::float / array_upper(string_to_array(oo."Required Skills", ','), 1) as matched_skills_ratio,
  qi.file_provenance_date
from 
  (
    select 
      *, parse_contractor(contractor_info) as con
    from 
      match.ja_quick_info qi
    where 
      qi.file_provenance_date between '2012-10-01' and '2012-11-01'
  ) qi join
  agg.b_opening o on qi.opening = o.opening join 
  "oDesk DB"."Openings" oo on oo."Record ID#" = o.opening join
  "oDesk DB"."JobCategories" jc on jc."Record ID#" = o.relatedjobcategory join
  agg.b_job_application ja on qi.application = ja.application left outer join
  sdb.openings so on so."Record ID#" = o.opening left outer join
  "oDesk DB"."Regions" r on r."Record ID#" = oo."PrefLocationRegion"
where 
  o.include_in_stats
distributed randomly
;
