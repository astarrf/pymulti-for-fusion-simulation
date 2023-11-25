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


class Multi_Program(Enum):
    multi_1d = "multi_1d"
    multi_2d = "multi_2d"
    multi_3d = "multi_3d"


def init(program: str, bashrc_path: str):
    try:
        os.system(f'source {bashrc_path}')  # 执行source命令，加载bashrc配置
    except Exception as e:
        if program == Multi_Program.multi_1d:
            pass
        elif program == Multi_Program.multi_2d:
            os.system(
                'export MULTI=/lustre/home/acct-phydci/phydci-user0/2023ConeAngle/MULTI-2D/')
            os.system(
                'export PATH=/lustre/home/acct-phydci/phydci-user0/2023ConeAngle/MULTI-2D/r94/boot-3.1:$PATH')
        elif program == Multi_Program.multi_3d:
            pass
    finally:
        os.system('echo $MULTI')  # 打印环境变量MULTI的值
