from setuptools import setup
import os
setup(
    name='pymulti',
    version='0.1.0',
    packages=['PYMULTI-FOR-INUFSION-SIMULATION'],
    install_requires=[
        # 依赖库的列表
        'os',
        'shutil',
        'subprocess',
        're>=2.2.1',
        'warnings',
        'skopt>=0.9.0',
    ],
)
