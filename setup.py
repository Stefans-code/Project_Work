# coding=utf-8
from setuptools import setup

setup(
    name='pyphysio',
    packages=['pyphysio',
              'pyphysio.loaders',
              'pyphysio.specialized',
              'pyphysio.indicators',
              'pyphysio.generators'],
    package_data={'tests': ['data/*']},
    version='3.0',
    description='Python library for physiological signals analysis (IBI & HRV, ECG, BVP, EDA, RESP...)',
    author='MPBA FBK',
    author_email='andrea.bizzego@unitn.it',
    url='https://github.com/MPBA/pyphysio',
    keywords=['eda', 'gsr', 'ecg', 'bvp', 'signal', 'analysis', 'physiological', 'pyhrv', 'hrv'],
    classifiers=[
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pycwt',
        'dask',
        'netcdf4'
    ],
    requires=[
        'csaps',
        'pywavelets',
        'nilearn',
        'pytest',
    ],
)

print("")
print("")
print("")
print("----------------------------------")
print("|                                |")
print("|  Thanks for using 'pyphysio'!  |")
print("|                                |")
print("----------------------------------")
print("")
print("Remember to cite pyphysio in your publications:")
print("Bizzego et al. (2019) 'pyphysio: A physiological signal processing library for data science approaches in physiology', SoftwareX")
print("")
print("----------------------------------")
