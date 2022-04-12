from setuptools import setup

setup(
    name='tes',
    url='http://www.quantum.info',
    author='Geoff Gillett and Leonardo Assis',
    author_email='uqlassis@uq.edu.au',
    # package_dir={'tes': ''},
    py_modules=[
                'tes.base', 'tes.data', 'tes.filesets', 'tes.maps', 'tes.mca', 'tes.protocol', 'tes.registers', 'tes.calibration', 'tes.counts', 'tes.data_acquisition', 'tes.folder_management', 'tes.mca_control', 'tes.traces'
                ],
    install_requires=['matplotlib', 'numpy', 'scipy', 'lmfit',
              'pyzmq', 'pyserial', 'pyyaml', 'numba'],
    version='0.2.0',
    description="Package for communicating with the hardware processor "
                "connected to Transition Edge Sensors (TES) at "
                "Quantum Technology Lab (QTLab) - UQ."
    
)
