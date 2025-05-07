# OSS/setup.py
# 각 패키지 폴더에 __init__.py 파일을 추가하여 패키지로 인식되도록 합니다.
# pip install -e . 명령어로 설치할 수 있습니다.
from setuptools import setup, find_packages

setup(
    name='oss_RecSys',
    version='0.1',
    packages=find_packages(include=[
        'model01_content_based',
        'model02_yelp_NCF',
        'model03_id_NCF',
        'model04_demographic_filtering',
    ]),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'wandb',
    ],
)
