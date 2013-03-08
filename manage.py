from flask.ext.script import Manager, Command
from flask.ext.alembic import ManageMigrations
from api import db, app


class CreateDB(Command):
    """Manage alembic migrations"""
    capture_all_args = True

    def run(self, args):
        db.create_all()

manager = Manager(app)
manager.add_command("migrate", ManageMigrations())
manager.add_command("createdb", CreateDB())

if __name__ == "__main__":
    manager.run()
