# coding=utf-8
from setuptools import setup
#import conda_build.bdist_conda

setup(
    name='pyphysio',
    packages=['pyphysio',
              'pyphysio.loaders',
              'pyphysio.specialized.activity',
              'pyphysio.specialized.eda',
              'pyphysio.specialized.eeg',
              'pyphysio.specialized.emg',
              'pyphysio.specialized.fnirs',
              'pyphysio.specialized.heart',
              'pyphysio.specialized.resp',
              'pyphysio.indicators',
              'pyphysio.generators'],
    package_data={'pyphysio': ['test_data/*']},
    # data_files={'test_data_out': ['info_medical', 'medical.txt.bz2']},
    version='3.1.10',
    description='Python library for physiological signals analysis (IBI & HRV, ECG, BVP, EDA, RESP, fNIRS, ...)',
    author='a.bizzego',
    author_email='andrea.bizzego@unitn.it',
    url='https://gitlab.com/a.bizzego/pyphysio',
    keywords=['eda', 'gsr', 'ecg', 'bvp', 'fnirs', 'signal', 
              'analysis', 'physiological', 'psychopysiology', 'neuroscience'],
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
        'dask',
        'netcdf4',
        'xarray',
        'pywavelets',
        'bottleneck'
    ],
    requires=[
        'h5py'
        'pytorch',
        'csaps',
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
