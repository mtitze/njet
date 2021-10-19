# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

def get_long_description():
    with open(f'./README.md', encoding='utf8') as fp:
        return fp.read()
        
setup(
    long_description=get_long_description(),
    name='njet',
    version='0.1.4',
    description='Lightweight automatic differentiation package for higher-order differentiation.',
    long_description_content_type='text/markdown',
    python_requires='<3.11,>=3.8',
    project_urls={
        "homepage": "https://njet.readthedocs.io/en/latest/index.html",
        "repository": "https://github.com/mtitze/njet"
    },
    author='Malte Titze',
    author_email='mtitze@users.noreply.github.com',
    license='GPL-3.0-or-later',
    keywords='AD jet njet automatic differentiation',
    packages=['njet'],
    package_dir={"": "."},
    package_data={},
    install_requires=['numpy==1.*,>=1.21.2', 'sympy==1.*,>=1.8.0'],
    extras_require={"dev": ["ipykernel==6.*,>=6.4.1"]},
)
