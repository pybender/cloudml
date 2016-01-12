"""
Unittests for inline predict section classes.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
from lxml import objectify

from cloudml.importhandler.predict import Predict
from cloudml.importhandler.exceptions import ImportHandlerException


class TestPredict(unittest.TestCase):
    TAG = objectify.fromstring("""
        <predict>
          <model name="rank" value="BestMatch.v31">
          </model>
          <model name="autohide" value="BestMatch.v31">
              <weight label="true" value="1.23543"/>
              <weight label="false" value="1.0"/>
          </model>
          <result>
              <label model="rank" />
              <probability model="rank" label="true" />
          </result>
    </predict>
    """)
    INVALID_WEIGTH = objectify.fromstring("""
        <predict>
          <model name="rank" value="BestMatch.v31">
              <weight label="true" value="1;23543"/>
              <weight label="false" value="1.0"/>
          </model>
          <result>
              <label model="rank" />
              <probability model="rank" label="true" />
          </result>
    </predict>
    """)
    INVALID_MODEL_TAG = objectify.fromstring("""
        <predict>
          <model name="rank" />
          <result>
              <label model="rank" />
              <probability model="rank" label="true" />
          </result>
    </predict>
    """)

    def test_params_validation(self):
        predict = Predict(self.TAG)

        prob = predict.result.probability
        self.assertEquals(prob.model, "rank")
        self.assertEquals(prob.script, None)
        self.assertEquals(prob.label, "true")

        lab = predict.result.label
        self.assertEquals(lab.model, "rank")
        self.assertEquals(lab.script, None)

        self.assertEquals(len(predict.models), 2)

    def test_predict_declaration(self):
        with self.assertRaisesRegexp(ImportHandlerException,
                                     "Either value or script attribute need to"
                                     " be defined for predict model rank"):
            Predict(self.INVALID_MODEL_TAG)

        with self.assertRaisesRegexp(ImportHandlerException,
                                     "Invalid predict model weight: 1;23543."
                                     "Should be a float value."):
            Predict(self.INVALID_WEIGTH)
