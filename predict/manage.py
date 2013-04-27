import os.path
import inspect

from flask.ext.script import Manager, Shell
from flask.ext.script import Command, Option

from api import app

def _make_context():
    return dict(app=app)

manager = Manager(app)

class Test(Command):
    """Run app tests."""

    def run(self):
        import nose
        nose.run(argv=['', '--exclude-dir=core'])

manager.add_command('test', Test())
manager.add_command("shell", Shell(make_context=_make_context))

if __name__ == "__main__":
    manager.run()
