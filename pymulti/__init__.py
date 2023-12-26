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


def init_one(program: str, program_path: str = None):
    if program_path is None:
        program_path = '/lustre/home/acct-phydci/phydci-user0/2023ConeAngle'
    if program == Multi_Program.multi_1d:
        d = 1
    elif program == Multi_Program.multi_2d:
        d = 2
    elif program == Multi_Program.multi_3d:
        d = 3
    os.system(f'export MULTI={program_path}/MULTI-{d}D/')
    os.system(f'export PATH={program_path}/MULTI-{d}D/r94/boot-3.1:$PATH')


def init(program: str, bashrc_path: str = None, program_path: str = None):
    if bashrc_path is None:
        init_one(program, program_path)
    else:
        try:
            os.system(f'source {bashrc_path}')  # 执行source命令，加载bashrc配置
        except Exception as e:
            init_one(program, program_path)
    os.system('echo $MULTI')  # 打印环境变量MULTI的值


def findallcases(path: str, rule: str = None):
    """
    从路径中找到所有的case
    :param path: 路径
    :return: 所有的case
    """
    files = [file for file in os.listdir(path)]
    my_files = []
    for file in files:
        flag = re.search(rule, file)
        if flag != None:
            my_files.append(file)
    return my_files
