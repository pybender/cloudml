import datetime
import json


PARAMS = {'start': '2012-12-03', 'end': '2012-12-04'}

ROW = {
    "dev_is_looking": "1",
    "application": '555',
    "contractor_info": {
        "dev_is_looking": "1",
        "dev_is_looking_week": "1",
        "dev_active_interviews": "5",
        "dev_availability": "30",
    },
    "employer_info": {
        "op_timezone": "UTC+10:00",
        "op_country_tz": "",
        "op_country": "Philippines"
    },
    "float_field": "5.3",
    "int_field": "7",
    "boolean": 1,
    "json_field": {"Communication": "5.0",
                   "Skills": "5.0",
                   "Deadlines": "5.0",
                   "Availability": "5.0",
                   "Quality": "5.0",
                   "Cooperation": "5.0"},
    "json_jsonpath_field": [{"val_0": "Professional and experienced person"}],
    "list_field": ["hello", "hi"],
    "store": {"book": [{"author": "Nigel"}, {"author": "Evelyn"}]},
    "say_hello": "hello hi pruvit",
    "words": "Words, words, words",
    "date": "Jun 1 2014  1:33PM",
}
