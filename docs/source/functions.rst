Functions and jets
==================

The njet module can derive functions which are built from elementary functions by means of
basic operations (e.g. addition, multiplication, composition, component projection, exponentiation, ...).

In the following we list those elementary functions which are currently supported to be
used with jets. If you like to include your custom elementary function which can not yet be
found in this list, you have to build it by yourself or drop me a message so that I may add it to the repo
(take a look at the njet.functions script for examples).

These more 'elementary' functions are characterized by the fact that their derivatives are known exactly
and can easily be computed up to any order.

.. automodule:: njet.functions
    :members:
    :undoc-members:
    :exclude-members: get_function, get_package_name, jetfunc

