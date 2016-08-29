import os
import sys

from setuptools import setup, find_packages
from distutils.core import Command
from unittest import TextTestRunner, TestLoader
from glob import glob
from os.path import splitext, basename, isfile, join as pjoin


PROJECT_BASE_DIR = ''
TEST_PATHS = ['tests']


def parse_requirements(requirements):
    install_requires = []
    dependency_links = []
    if os.path.isfile(requirements):
        with open(requirements) as f:
            for l in f:
                if not l.strip('\n') or l.startswith('#'):
                    continue
                if l.startswith('-'):
                    dependency_links.append(l.strip('\n').replace('-e ', ''))
                else:
                    install_requires.append(l.strip('\n'))
        install_requires.reverse()
    return install_requires, dependency_links



install_requires, dependency_links = parse_requirements('requirements.txt')


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


setup(
    name='cloudml',
    version=read_version_string(),
    description='Machine learning as a service scipy-trainer',
    author='Panagiotis Papadimitriou, Ioannis Foukarakis \
Nikolay Melnik',
    author_email='papadimitriou@cloud.upwork.com, \
ifoukarakis@cloud.upwork.com, nmelnik@cloud.upwork.com',
    maintainer='Nikolay Melnik',
    maintainer_email='nmelnik@cloud.upwork.com',
    packages=find_packages(),
    include_package_data=True,
    scripts=['trainer.py', 'predictor.py',
               'transformer.py', 'importhandler.py'],
    url='http://www.upwork.com',
    cmdclass={
        'test': NoseCommand,
        'coverage': CoverageCommand
    },
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Information Technology',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
    setup_requires=["numpy==1.10.04"],
    install_requires=install_requires,
    dependency_links=dependency_links,
    tests_require=['nose', 'coverage', 'moto==0.3.3', 'mock==1.0.1']
)
