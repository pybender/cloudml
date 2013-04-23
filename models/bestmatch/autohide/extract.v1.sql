select
  oja."Status",
  oja."Related Reason",
  r."Reason",
  count(*)
from
  "oDesk DB"."JobApplications" oja join
  agg.b_opening o on oja."Related Opening" = o.opening left outer join
  "oDesk DB"."Reasons" r on oja."Related Reason" = r."Record ID#"
where
  oja."Status" in ('Cancelled', 'Rejected') and 
  oja."CreatedType" = 'Professional' and
  o.include_in_stats = true and
  to_timestamp(oja."Date Created"/1000) between '2012-01-01' and '2013-01-01'
group by
  1, 2, 3
order by
  1, 4 desc;

create temporary table autohide_tmp as
select
  qi.application,
  qi.opening,
  case 
    when r."Record ID#" in (147, 133, 134, 137, 151, 159) and oja."Status" = 'Rejected' then true
    else false
  end as rejected,
   
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
  am."Message" as cover_letter,

  oo."PrefAvailableHoursPerWeek" as pref_available_hours,
  oo."PrefEnglishSkill" as pref_english,
  oo."PrefFeedbackScore" as pref_feedback,
  oo."PrefGroup" as pref_group,
  oo."PrefHasPortfolio" as pref_has_portfolio,
  oo."PrefHourlyRateMin" as pref_rate_min,
  oo."PrefHourlyRateMax" as pref_rate_max,
  oo."PrefLocationRegion" as pref_region_id,
  reg."Region" as pref_region,
  oo."PrefTest" as pref_test,
  oo."PrefoDeskHours" as pref_odesk_hours,
  so.candidate_type_pref as pref_candidate_type
  
from
  "oDesk DB"."JobApplications" oja join
  agg.b_job_application ja on oja."Record ID#"::int = ja.application::int join
  match.ja_quick_info qi on qi.application::int = ja.application::int join
  agg.b_opening o on qi.opening::int = o.opening::int join 
  "oDesk DB"."Openings" oo on oo."Record ID#"::int = o.opening::int join
  "oDesk DB"."JobCategories" jc on jc."Record ID#"::int = o.relatedjobcategory::int join
  "oDesk DB"."Reasons" r on oja."Related Reason"::int = r."Record ID#"::int left outer join
  "oDesk DB"."AssignmentMessages" am on am."Record ID#"::int = oja."Related AssignmentMessage for Candidate Referral Cover Letter"::int left outer join
  sdb.openings so on so."Record ID#"::int = o.opening::int left outer join
  "oDesk DB"."Regions" reg on reg."Record ID#"::int = oo."PrefLocationRegion"::int
where
  oja."CreatedType" = 'Professional' and
  o.include_in_stats = true and
  qi.file_provenance_date between '%(start_date)s' and '%(end_date)s';

select 
  autohide_tmp.*,
  qi.employer_info, 
  qi.contractor_info
from 
  autohide_tmp join match.ja_quick_info qi on autohide_tmp.application::int = qi.application::int
where
  qi.file_provenance_date between '%(start_date)s' and '%(end_date)s';



--  qi.file_provenance_date between '%(start_date)' and '%(end_date)';



--select * from "oDesk DB".date_dim limit 10;