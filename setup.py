from setuptools import setup, find_packages

setup(
    name =         'ti_lib',
    version =      '0.0.1',
    packages =     find_packages(),
    description =  'Technicla Indicator Lib',
    author =       'Dmitry Klimenko',
    author_email = 'klimenko.dnk@gmail.com',
    url =           '',
    zip_safe =      True,  
    install_requires = [
        'setuptools>=69.0.0',
        'numpy>=1.26',        
        'numba>=0.60.0',
    ],
    classifiers = [
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires = '>=3.12',
)