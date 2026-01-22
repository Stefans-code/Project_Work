# coding=utf-8
from setuptools import setup
# from .pyphysio import version
#import conda_build.bdist_conda

version = '4.4.0'

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
    package_data={'pyphysio': ['test_data/*', 'specialized/fnirs/_dlweights/*']},
    # data_files={'test_data_out': ['info_medical', 'medical.txt.bz2']},
    version=version,
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
        'bottleneck',
        'pyxdf',
        'csaps',
        'nilearn',
        'h5py',
        'pytest'
    ],
    requires=[        
        'pytorch',
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
