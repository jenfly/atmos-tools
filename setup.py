from setuptools import setup

setup(name='atmos',
      version='0.1',
      description='Handy utilities for atmospheric science',
      url='https://github.com/jenfly/atmos-tools',
      author='Jennifer Walker',
      author_email='jenfly@gmail.com',
      license='MIT',
      packages=['atmos'],
      install_requires=[
          'basemap',
          'matplotlib',
          'numpy',
          'pandas',
          'PyPDF2',
          'scipy',
          'xarray',
      ],
      include_package_data=True,
      zip_safe=False,
)
