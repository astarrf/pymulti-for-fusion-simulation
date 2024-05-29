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


# 保存原始的print函数，以便稍后调用它。
rewrite_print = print

# 定义新的print函数。


def print(*arg):
    # 首先，调用原始的print函数将内容打印到控制台。
    rewrite_print(*arg)

    # 如果日志文件所在的目录不存在，则创建一个目录。
    output_dir = "log_file"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开（或创建）日志文件并将内容写入其中。
    log_name = 'log.txt'
    filename = os.path.join(output_dir, log_name)
    with open(filename, "a") as f:
        f.write(str(arg) + "\n")
    # rewrite_print(*arg, file=open(filename, "a"))


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
    '''
    初始化环境变量
    program: 
        Multi_Program类型，指定要初始化的程序
    bashrc_path: 
        str类型，指定bashrc文件路径
    program_path: 
        str类型，指定程序路径
    '''
    if bashrc_path is None:
        init_one(program, program_path)
    else:
        try:
            os.system(f'source {bashrc_path}')  # 执行source命令，加载bashrc配置
        except Exception as e:
            init_one(program, program_path)
    os.system('echo $MULTI')  # 打印环境变量MULTI的值


def findAllCases(path: str, rule: str = None) -> list:
    '''
    寻找指定路径下的所有文件
    path: 
        str类型，指定要查找的路径
    rule:
        str类型，指定查找规则
    '''
    files = [file for file in os.listdir(path)]
    my_files = []
    for file in files:
        if re.search(rule, file) != None:
            my_files.append(file)
    print(f'number of files found: {len(my_files)}')
    return my_files


def getAllReward(judge_func, CaseDir: str, source_path: str, output_path: str, cases_path: str, rule: str = None, n_jobs: int = 5) -> list:
    '''
    获取所有文件的reward
    judge_func: 
        function类型，指定评价函数
    CaseDir:
        str类型，指定case路径
    source_path:
        str类型，指定source路径
    output_path:
        str类型，指定输出文件路径
    cases_path:
        str类型，指定cases路径
    rule:
        str类型，指定查找规则
    n_jobs:
        int类型，指定并行任务数
    '''
    my_files = findAllCases(cases_path, rule)
    # 初始化变量
    rwd = []
    cnt = 1
    file_lock = threading.Lock()

    tot = len(my_files)
    print(tot)

    # 定义处理函数

    def process_case(file) -> Cases:
        my_case = Cases(Multi_Program.multi_3d,
                        CaseDir=CaseDir,
                        source_path=source_path,
                        target_path=f'/{file}',
                        replace_list=None)
        return my_case

    def judge_case(case: Cases, index):
        print(f'start:{index}')
        return judge_func(case), case.file_path

    # 创建并提交任务
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        # 创建case并获取参数
        futures = {executor.submit(process_case, file)                   : file for file in my_files}

        # 收集case和参数
        cases = []
        for future in as_completed(futures):
            try:
                my_case = future.result()
                cases.append(my_case)
                print(f'{cnt}/{tot}', my_case.file_path)
            except Exception as e:
                raise Exception(
                    f"Error processing file {futures[future]}: {e}")
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
                raise Exception(
                    f"Error judging case {judge_futures[future]}: {e}")

    print("Processing complete.")
    return rwd
