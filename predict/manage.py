from flask.ext.script import Manager, Shell

from api import app

def _make_context():
    return dict(app=app)

manager = Manager(app)
manager.add_command("shell", Shell(make_context=_make_context))

if __name__ == "__main__":
    manager.run()
