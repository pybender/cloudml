import os
import sys

from distutils.core import setup
from distutils.core import Command
from unittest import TextTestRunner, TestLoader
from glob import glob
from os.path import splitext, basename, join as pjoin


PROJECT_BASE_DIR = ''
TEST_PATHS = ['tests']

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]


install_requires = parse_requirements('requirements.txt')

def read_version_string():
    sys.path.insert(0, pjoin(os.getcwd()))
    from cloudml import __version__
    version = __version__
    sys.path.pop(0)
    return version


class TestCommand(Command):
    description = 'run test suite'
    user_options = []

    def initialize_options(self):
        THIS_DIR = os.path.abspath(os.path.split(__file__)[0])
        sys.path.insert(0, THIS_DIR)
        for test_path in TEST_PATHS:
            sys.path.insert(0, pjoin(THIS_DIR, test_path))
        self._dir = os.getcwd()

    def finalize_options(self):
        pass

    def run(self):
        status = self._run_tests()
        sys.exit(status)

    def _run_tests(self):
        testfiles = []
        for test_path in TEST_PATHS:
            for t in glob(pjoin(self._dir, test_path, '*_tests.py')):
                testfiles.append('.'.join(
                    [test_path.replace('/', '.'), splitext(basename(t))[0]]))

        tests = TestLoader().loadTestsFromNames(testfiles)

        t = TextTestRunner(verbosity=2)
        res = t.run(tests)
        return not res.wasSuccessful()


class NoseCommand(Command):
    description = 'run test suite using nose tests and generate report'
    user_options = []

    def initialize_options(self):
        THIS_DIR = os.path.abspath(os.path.split(__file__)[0])
        sys.path.insert(0, THIS_DIR)
        for test_path in TEST_PATHS:
            sys.path.insert(0, pjoin(THIS_DIR, test_path))
        self._dir = os.getcwd()

    def finalize_options(self):
        pass

    def run(self):
        status = self._run_tests()
        sys.exit(status)

    def _run_tests(self):
        import nose
        if not os.path.exists('target'):
            os.mkdir('target')
        nose.run(argv=['', '--verbose',
                       '--with-xunit', '--xunit-file=target/nosetests.xml',
                       '--with-doctest'])


class CoverageCommand(Command):
    description = 'run test suite and generate coverage report'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import coverage
        cov = coverage.coverage(config_file='.coveragerc')
        cov.start()

        tc = NoseCommand(self.distribution)
        tc._run_tests()

        cov.stop()
        cov.html_report(directory='./target/coverage/html')
        cov.xml_report(outfile='./target/coverage/coverage.xml')
        cov.save()


class SonarCommand(Command):
    description = 'generate sonar report'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        cc = CoverageCommand(self.distribution)
        cc.run()
        os.system(
            'sonar-runner'
            ' -Dproject.settings=scripts/sonar-project.properties'
        )

setup(
    name='cloudml',
    version=read_version_string(),
    description='Machine learning as a service scipy-trainer',
    author='Ioannis Foukarakis',
    author_email='ifoukarakis@odesk.com',
    packages=[
        'core',
    ],
    package_dir={
        'cloudml': 'core'
    },
    url='http://www.upwork.com',
    cmdclass={
        'test': NoseCommand,
        'coverage': CoverageCommand,
        'sonar': SonarCommand
    },
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Match team',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=install_requires,
    test_requires=['nose', 'coverage', 'moto==0.3.3', 'mock==1.0.1']
)
