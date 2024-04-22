import setuptools

verstr = "0.0.1"

setuptools.setup(
    name="gw_lensing",
    version=verstr,
    author="Jose María Ezquiaga",
    author_email="jose.ezquiaga@nbi.ku.dk",
    description="Exploring gravitational wave lensing of gravitational waves.",
    packages=[
        "gw_lensing",
        #"gw_lensing.bayesian_inference",
        "gw_lensing.cosmology",
        "gw_lensing.detectors",
        "gw_lensing.gw_population",
        "gw_lensing.gw_rates",
        "gw_lensing.utils",
        'gw_lensing.detectors.pw_network',
        'gw_lensing.detectors.sensitivity_curves',
    ],
    package_data = {
        'gw_lensing.detectors.sensitivity_curves': ['*.txt'],
        'gw_lensing.detectors.pw_network': ['*.txt'],
                   },
    include_package_data=True,

    install_requires=[
        "numpyro",
    ],

    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
    python_requires='>=3.7',
)
