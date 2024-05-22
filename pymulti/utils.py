"""
Multi-1D/Multi-2D/Multi-3D automaic control.
本库由上海交通大学物理与天文学院开发，用于自动化控制等离子体模拟程序Multi-1D/Multi-2D/Multi-3D的运行。
本库仅用于学术交流，不得用于商业用途。
"""

import threading
import os
import re
from enum import Enum
from .CaseIO import Cases
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def findAllCases(path: str, rule: str = None):
    files = [file for file in os.listdir(path)]
    my_files = []
    for file in files:
        if re.search(rule, file) != None:
            my_files.append(file)
    print(len(my_files))
    return my_files


def getAllReward(judge_func, CaseDir: str, source_path: str, output_path: str, file_path: str, rule: str = None):
    my_files = findAllCases(file_path, rule)
    # 初始化变量
    rwd = []
    param1 = []
    param2 = []
    cnt = 1
    n_jobs = 5
    file_lock = threading.Lock()

    tot = len(my_files)
    print(tot)

    # 定义处理函数

    def process_case(file):
        my_case = Cases(Multi_Program.multi_3d,
                        CaseDir=CaseDir,
                        source_path=source_path,
                        target_path=f'/{file}',
                        replace_list=None)
        return my_case

    def judge_case(case, index):
        print(f'start:{index}')
        return judge_func(case), case.file_path

    # 创建并提交任务
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # 创建case并获取参数
        futures = {executor.submit(process_case, file)
                                   : file for file in my_files}

        # 收集case和参数
        cases = []
        for future in as_completed(futures):
            try:
                my_case = future.result()
                cases.append(my_case)
                print(f'{cnt}/{tot}', my_case.file_path)
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")
            cnt += 1

        # 创建并提交judge任务
        judge_futures = {executor.submit(
            judge_case, case, index): case for index, case in enumerate(cases)}

        # 收集judge结果并写入文件
        for future in as_completed(judge_futures):
            try:
                reward, file = future.result()
                rwd.append(reward)
                with file_lock:
                    with open(output_path, "a") as f:
                        f.write(f'{file} {reward}\n')
            except Exception as e:
                print(f"Error judging case {judge_futures[future]}: {e}")

    print("Processing complete.")
    print(param1)
    print(param2)
    print(rwd)
