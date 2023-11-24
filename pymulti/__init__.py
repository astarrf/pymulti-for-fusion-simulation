"""
Multi-1D/Multi-2D/Multi-3D automaic control.
本库由上海交通大学物理与天文学院开发，用于自动化控制等离子体模拟程序Multi-1D/Multi-2D/Multi-3D的运行。
本库仅用于学术交流，不得用于商业用途。
"""

import os 
import shutil
import subprocess
import re
import warnings
import skopt
from enum import Enum
import struct

VERSION = '0.1.0'