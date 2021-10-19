from setuptools import setup, find_packages

DISTNAME = 'Mechanical-Analysis'
DESCRIPTION = 'Mechanical-Analysis: some more crazy mess Daan made'
MAINTAINER = 'Daan Boltje'
MAINTAINER_EMAIL = 'boltje@delmic.com'
LICENSE = 'LICENSE'
README = 'README.md'
URL = 'https://github.com/hoogenboom-group/Mechanical-Analysis'
VERSION = '0.1.dev'
PACKAGES = ['manalysis']
INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'scikit-image',
    'tqdm',
    'joblib',
]

if __name__ == '__main__':

    setup(name=DISTNAME,
          version=VERSION,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          packages=PACKAGES,
          include_package_data=True,
          url=URL,
          license=LICENSE,
          description=DESCRIPTION,
          long_description=open(README).read(),
          install_requires=INSTALL_REQUIRES)
