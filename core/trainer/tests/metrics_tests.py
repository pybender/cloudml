# Author: Nikolay Melnik <nmelnik@upwork.com>

import unittest

from core.trainer.metrics import Metrics, ClassificationModelMetrics, \
    RegressionModelMetrics
from core.trainer.exceptions import SchemaException


class MetricsMeTest(unittest.TestCase):
    def test_factory(self):
        metrics = Metrics.factory("classification")
        self.assertEquals(type(metrics), ClassificationModelMetrics)

        metrics = Metrics.factory("regression")
        self.assertEquals(type(metrics), RegressionModelMetrics)

        with self.assertRaisesRegexp(
                SchemaException, "invalid model type isn't supported"):
            Metrics.factory("invalid")
