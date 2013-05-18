from datetime import datetime
import calendar

from base import FeatureTypeBase, FeatureTypeInstanceBase


class DateFeatureTypeInstance(FeatureTypeInstanceBase):

    def transform(self, value):
        """
        Convert date to UNIX timestamp.

        Keyword arguments:
        value -- the value to convert
        params -- params containing the pattern

        """
        params = self.active_params()
        if value is None:
            return self.default_value
        try:
            return calendar.timegm(
                datetime.strptime(value, params['pattern']).timetuple())
        except ValueError:
            pass
        except TypeError:
            pass
        return self.default_value


class DateFeatureType(FeatureTypeBase):
    instance = DateFeatureTypeInstance
    required_params = ['pattern']
    # Default is Jan 1st, 2000
    default = 946684800
