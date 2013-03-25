-- Dataset selection
create table match.bestmatch_dataset as
select
  case when ja.offermc_comp_reply = 'accept' then 1 else 0 end as made_offer,
  qi.employer_info, 
  qi.contractor_info, 
  qi.agency_info,

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
  so.candidate_type_pref as pref_candidate_type
from 
  match.ja_quick_info qi join
  agg.b_opening o on qi.opening = o.opening join 
  "oDesk DB"."Openings" oo on oo."Record ID#" = o.opening join
  "oDesk DB"."JobCategories" jc on jc."Record ID#" = o.relatedjobcategory join
  agg.b_job_application ja on qi.application = ja.application left outer join
  sdb.openings so on so."Record ID#" = o.opening left outer join
  "oDesk DB"."Regions" r on r."Record ID#" = oo."PrefLocationRegion"
where 
  o.include_in_stats and qi.file_provenance_date between '%(start_date)s' and '%(end_date)s' 
;