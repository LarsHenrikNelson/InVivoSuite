# __init__.py - Module setup for PyPL2
#
# (c) 2016 Plexon, Inc., Dallas, Texas
# www.plexon.com
#
# This software is provided as-is, without any warranty.
# You are free to modify or share this file, provided that the above
# copyright notice is kept intact.

#__init__.py serves three purposes:
#   1) Let's Python know that the .py files in this directory are importable modules.
#   2) Sets up classes and functions in the pypl2lib and pypl2api modules to be easy to
#      access. For example, without importing the pl2_ad function from pypl2api in 
#      __init__.py, you would have to import pypl2 in your script like this:
#           from pypl2.pypl2api import pl2_ad
#      instead of like this:
#           from pypl2 import pl2_ad
#      It's a minor convenience, but improves readability.
#   3) Explicitly states which classes and functions in PyPL2 are meant to be public 
#      parts of the API.

from .pypl2lib import PL2FileInfo, PL2AnalogChannelInfo, PL2SpikeChannelInfo, PL2DigitalChannelInfo, PyPL2FileReader
from .pypl2api import pl2_ad, pl2_spikes, pl2_events, pl2_info, pl2_comments

__author__ = 'Chris Heydrick (chris@plexon.com)'
__version__ = '1.3.0'

# I'll put thanks to community bug fixers and feature implementers here
__with_thanks_to__ = ['With thanks to Roland Ferger with the Pena Lab for demonstrating how to port to Python 3']

# 5/24/2016 CH
# Added 64-bit .dll support, incremented to 1.1.0
# 8/27/2018 CH
# pl2_ad, pl2_spikes, and pl2_events are now properly closing the .pl2 file when done
# Fixed pl2_ad not correctly handling when ad channel first timestamp is 0
# Incremented to 1.1.1
# 3/8/2019 CH
# Ported to Python 3 - no longer works with Python 2
# Incremented to 1.2.0
# 3/16/2021 CH
# Added functions for getting comments from OmniPlex 1.20+
# Incremented to 1.3.0