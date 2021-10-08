'''    
    AD-Lib: Automatic Differentiation Library
    Copyright (C) 2021  Malte Titze

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

from .functions import exp, log
from .ad import jet as jet_source

class jet(jet_source):    
    def __pow__(self, other):
        if not isinstance(other, jet):
            other = jet(value=other) # n.b. convert from ad.py would convert to 'jet_source', not jet'.

        if other.order == 0:
            result = jet_source.__pow__(self, other)
        else:
            '''
            General exponentiation, using exp and log functions.
            '''
            result = exp(other*log(self))
            result.graph = [(2, '**'), self.graph, other.graph]
        return result
    
    def __rpow__(self, other):
        if not isinstance(other, jet):
            other = jet(value=other) # n.b. convert from ad.py would convert to 'jet_source', not jet'.
        return other**self
