# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import calendar
from datetime import datetime

from base import FeatureTypeBase, FeatureTypeInstanceBase


class DateFeatureTypeInstance(FeatureTypeInstanceBase):

    def transform(self, value):
        """
        Convert date to UNIX timestamp.

        value: string
            the value to convert

        Note:
            Feature should containing the 'pattern' in parameters.
        """
        params = self.active_params()
        if value is None:
            return self._default_value

        try:
            return calendar.timegm(
                datetime.strptime(value, params['pattern']).timetuple())
        except (ValueError, TypeError):
            return self._default_value


class DateFeatureType(FeatureTypeBase):
    instance = DateFeatureTypeInstance
    required_params = ['pattern']
    default_value = 946684800  # Default is Jan 1st, 2000
