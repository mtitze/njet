'''    
    njet: A leightweight automatic differentiation library for 
          higher-order automatic differentiation
    
    Copyright (C) 2021, 2022, 2023 by Malte Titze

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from ._version import __version__
from .ad import jet, derive, taylor_coefficients
from .jet import jetpoly
