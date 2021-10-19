import toml
import os

tomldict = toml.load(f'{os.path.dirname(__file__)}/../pyproject.toml')
__version__ = tomldict['tool']['poetry']['version']
