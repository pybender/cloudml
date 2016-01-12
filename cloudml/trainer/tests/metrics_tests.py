# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest

from cloudml.trainer.metrics import Metrics, ClassificationModelMetrics, \
    RegressionModelMetrics
from cloudml.trainer.exceptions import SchemaException


class MetricsMeTest(unittest.TestCase):
    def test_factory(self):
        metrics = Metrics.factory("classification")
        self.assertEquals(type(metrics), ClassificationModelMetrics)

        metrics = Metrics.factory("regression")
        self.assertEquals(type(metrics), RegressionModelMetrics)

        with self.assertRaisesRegexp(
                SchemaException, "invalid model type isn't supported"):
            Metrics.factory("invalid")
