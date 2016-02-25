"""
Unittests for python scripts manager class.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest

from cloudml.importhandler.scripts import ScriptManager, Script
from cloudml.importhandler.exceptions import ImportHandlerException, \
    LocalScriptNotFoundException
from lxml import objectify
import os
from cloudml.tests.test_utils import StreamPill
import boto3


class ScriptManagerTest(unittest.TestCase):
    def setUp(self):
        self.manager = ScriptManager()

    def test_script(self):
        self.assertEqual(self.manager._exec('1+2'), 3)

    def test_manager(self):
        self.manager.add_python("""def intToBoolean(a):
            return a == 1
        """)
        self.assertEqual(self.manager._exec('intToBoolean(1)'), True)

    def test_match_1888(self):
        def check(script):
            print script
            self.manager.add_python(script)
            self.assertEqual(self.manager._exec('stripSpecial("abc<")'), "abc")

        SCRIPT1 = """
pattern='[<>]+'
def stripSpecial(value):
    import re
    return re.sub(pattern, \" \", value).strip()
"""
        SCRIPT2 = """
import re
def stripSpecial(value):
    return re.sub(\"[<>]+\", \" \", value).strip()
"""
        check(SCRIPT1)
        check(SCRIPT2)

    def test_invalid_script(self):
        self.assertRaises(
            ImportHandlerException,
            self.manager.add_python, '1dsf++2')
        self.assertRaises(
            ImportHandlerException,
            self.manager.execute_function, '1dsf++2', 2)
        self.assertRaises(
            ImportHandlerException,
            self.manager._exec, '1dsf++2')

    def test_exec_row_data(self):
        res = self.manager.execute_function(
            '#{value} + #{data.result.x}[0]', 3,
            local_vars={'data.result.x': (5, 1)})
        self.assertEquals(res, 8)


class ScriptTest(unittest.TestCase):
    BASE_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../testdata'))
    PLACEBO_RESPONSES_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'placebo_responses/script/'))
    EMPTY_SRC = objectify.fromstring(
        """<script src="" />"""
    )
    EMPTY_ALL = objectify.fromstring(
        """<script />"""
    )
    TEXT = objectify.fromstring(
        """<script><![CDATA[1+1]]></script>"""
    )
    LOCAL_SCRIPT_CORRECT = objectify.fromstring(
        """<script src="%s" />""" % os.path.join(BASE_DIR, "local_script.py")
    )
    LOCAL_SCRIPT_INCORRECT = objectify.fromstring(
        """<script src="%s" />""" % os.path.join(BASE_DIR, "local_script1.py")
    )
    PRIORITY_SCRIPT = objectify.fromstring(
        """<script src="%s"><![CDATA[2+2]]></script>""" %
        os.path.join(BASE_DIR, "local_script.py")
    )
    AMAZON_CORRECT = objectify.fromstring(
        """<script src="amazon_script.py" />"""
    )
    AMAZON_INCORRECT = objectify.fromstring(
        """<script src="amazon_script1.py" />"""
    )

    def setUp(self):
        super(ScriptTest, self).setUp()
        self.pill = StreamPill(debug=True)
        self.session = boto3.session.Session()
        boto3.DEFAULT_SESSION = self.session

    def test_empty_values(self):
        script = Script(self.EMPTY_SRC)
        self.assertEqual('', script.get_script_str())
        self.assertEqual('', script.src)
        self.assertEqual(None, script.text)

        script = Script(self.EMPTY_ALL)
        self.assertEqual('', script.get_script_str())
        self.assertEqual(None, script.src)
        self.assertEqual(None, script.text)

    def test_text_exists(self):
        script = Script(self.TEXT)
        self.assertEqual('1+1', script.get_script_str())
        self.assertEqual(None, script.src)
        self.assertEqual('1+1', script.text)

    def test_local_file(self):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR, 'incorrect'))
        self.pill.playback()

        script = Script(self.LOCAL_SCRIPT_INCORRECT)
        self.assertRaises(LocalScriptNotFoundException,
                          script._process_local_file)
        self.assertRaises(ImportHandlerException, script.get_script_str)
        script = Script(self.LOCAL_SCRIPT_CORRECT)
        self.assertEqual('def always99(a):\n    return 99',
                         script.get_script_str())
        self.assertEqual(os.path.join(self.BASE_DIR, "local_script.py"),
                         script.src)
        self.assertEqual(None, script.text)

        script = Script(self.PRIORITY_SCRIPT)
        self.assertEqual('def always99(a):\n    return 99',
                         script.get_script_str())
        self.assertEqual(os.path.join(self.BASE_DIR, "local_script.py"),
                         script.src)
        self.assertEqual('2+2', script.text)

    def test_incorrect_amazon_file(self):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR, 'incorrect'))
        self.pill.playback()

        script = Script(self.AMAZON_INCORRECT)
        self.assertRaises(ImportHandlerException, script.get_script_str)

    def test_correct_amazon_file(self):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR, 'correct'))
        self.pill.playback()
        script = Script(self.AMAZON_CORRECT)
        self.assertEqual(None, script.text)
        self.assertEqual("amazon_script.py", script.src)
        self.assertEqual("3+5", script.get_script_str())
