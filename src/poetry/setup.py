from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'trax'
]

setup(
    name='poetry',
    version='0.1',
    author = '',
    author_email = '',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Human Phrase Identification problem',
    requires=[]
)