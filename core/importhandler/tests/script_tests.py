"""
Unittests for python scripts manager class.
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>

import unittest

from core.importhandler.scripts import ScriptManager, prepare_context
from core.importhandler.exceptions import ImportHandlerException


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
