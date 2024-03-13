# 导入必要的库
# 导入库的方法：import pymulti.BayesOptimizer
import os
import skopt
import numpy as np
import CaseIO as io
from time import sleep
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class BayesOptimizer():
    def __init__(self, program: str, test_name: str, CaseDir: str, source_path: str, file_name: str, tag: str, features: list, func, prefix: list = [], suffix: list = [], sleep_T=10):
        """
        Bayes优化器的初始化函数。

        参数：
        - program: 选择程序
        - test_name: 测试名称
        - features: 特征列表
        - CaseDir: 案例目录
        - source_path: 源代码路径
        - file_name: 读取优化标准的文件路径
        - tag: 读取优化标准的标签
        - func: 目标函数
        - prefix: 前缀列表
        - suffix: 后缀列表
        - sleep_T: 每次检查文件是否生成的时间间隔
        """
        self.program = program
        self.test_name = test_name
        self.features = features
        self.CaseDir = CaseDir
        self.source_path = source_path
        self.file_name = file_name
        self.tag = tag
        self.func = func
        self.prefix = prefix
        self.suffix = suffix
        self.sleep_T = sleep_T

    def run(self, dimensions: list, x0=None, y0=None, n_calls=100, random_state=None, n_jobs: int = 1, do_delta_stop: bool = False, delta: float = 0.01, print_step: bool = False):
        """
        运行Bayes优化器。

        参数：
        - dimensions: 参数维度
        - x0: 初始参数
        - y0: 初始函数值
        - n_calls: 最大调用次数
        - random_state: 随机种子（没有测试）
        - n_jobs: 进程池最大数量
        - do_delta_stop: 是否使用最优值与上一个最优值差小于delta判断停止
        - delta: 停止条件
        - print_step: 是否打印每一步的结果

        返回：
        - res: 优化结果
        """
        x = []  # 用于存储x参数
        y = []  # 用于存储y参数
        if x0 is None:
            x0 = []
        if y0 is None:
            y0 = []
        for xx in x0:
            x.append(xx)
        for yy in y0:
            y.append(yy)

        rng = skopt.utils.check_random_state(random_state)  # 随机种子
        space = skopt.utils.normalize_dimensions(dimensions)  # 参数维度
        xi = 0.01
        kappa = 1.96
        base_estimator = skopt.utils.cook_estimator(
            "GP", space=space, random_state=rng.randint(0, np.iinfo(np.int32).max),
            noise="gaussian")  # 用于拟合目标函数的模型
        acq_func_kwargs = {"xi": xi, "kappa": kappa}  # 用于计算下一个点的函数值
        res = skopt.Optimizer(
            dimensions=dimensions,
            base_estimator=base_estimator,
            acq_func="gp_hedge",
            acq_func_kwargs=acq_func_kwargs,
            random_state=random_state)  # 优化器
        # 如果有初始点，先计算初始点的函数值
        if x0:
            res.tell(x0, y0, fit=True)
            best_y = np.min(res.yi)
        else:
            best_y = 1e10
        # 使用进程池计算下一个点的函数值
        with ThreadPoolExecutor(n_jobs) as executor:
            for i in range(n_calls):
                x_trys = res.ask(n_points=n_jobs)
                future_results = {executor.submit(
                    self.__bofunc_, xtmp): xtmp for xtmp in x_trys}
                y_trys = []
                for future in as_completed(future_results):
                    y_trys.append(future.result())
                res.tell(x_trys, y_trys, fit=True)
                for x_try in x_trys:
                    x.append(x_try)
                for y_try in y_trys:
                    y.append(y_try)
                res.x = x
                res.y = y
                # 如果使用delta停止，当优化小于delta时停止
                if do_delta_stop:
                    if best_y - np.min(res.yi) < delta:
                        break
                    else:
                        best_y = np.min(res.yi)
                # 打印每一步的结果
                if print_step:
                    for i in range(len(x_trys)):
                        print("最新尝试的参数：", x_trys[i])
                        print("最新尝试的函数值：", y_trys[i])
        index = np.argmin(res.y)  # 最优参数的索引
        x_ans = res.x[index]  # 最优参数
        y_ans = res.y[index]  # 最优函数值
        res.x_ans = x_ans  # 最优参数
        res.y_ans = y_ans  # 最优函数值
        return res

    def __bofunc_(self, x):
        """
        Bayes优化的目标函数。

        参数：
        - x: 参数

        返回：
        - reward: 目标函数值
        """
        new_x = []
        for i, xx in enumerate(x):
            pre, suf = '', ''
            if i < len(self.prefix):
                pre = self.prefix[i]
            if i < len(self.suffix):
                suf = self.suffix[i]
            xx = f'{pre}{xx}{suf}'
            new_x.append(xx)
        replace_list = io.merge_feature(self.features, new_x)
        target_path_name = f'/{self.test_name}'
        now = datetime.now()
        time = now.strftime("%Y%m%d%H%M")
        target_path_name = f'{target_path_name}_{time}'
        target_path_name = target_path_name[:32]
        print(self.CaseDir, self.source_path, target_path_name)
        case = io.Cases(self.program, self.CaseDir, self.source_path,
                        target_path_name, replace_list)
        case.new_case()
        case.run()
        case_file_name = f'{self.CaseDir}/{target_path_name}_3DM/{self.file_name}'
        print(case_file_name)
        running = True
        while running:
            running = not (os.path.isfile(case_file_name))
            sleep(self.sleep_T)
        reward = self.func(case, self.file_name, self.tag)
        return reward


class Traverser():
    def __init__(self, program: str, test_name: str, bashrc_path: str, CaseDir: str, source_path: str, traverse_list: list, prefix: list = [], suffix: list = []):
        """
        遍历器的初始化函数。
        参数：
        - program: 选择程序
        - test_name: 测试名称
        - bashrc_path: bashrc文件路径
        - features: 特征列表
        - CaseDir: 案例目录
        - source_path: 源代码路径
        - traverse_list: 遍历列表:[[feature1,[val1_1,val2_1,...]],[feature2,[val1_2,val2_2,...]],...]
        - prefix: 前缀列表
        - suffix: 后缀列表
        """
        self.program = program
        self.test_name = test_name
        self.bashrc_path = bashrc_path
        self.CaseDir = CaseDir
        self.source_path = source_path
        self.traverse_list = traverse_list
        self.feature, self.feature_range = self.__traverse_list_reshape_(
            self.traverse_list)
        self.prefix = prefix
        self.suffix = suffix

    def __traverse_list_reshape_(self, list_2d: list):
        """
        将二维列表转换为遍历列表,输出一个变量表和一个二维遍历表
        [[feature1,[val1_1,val2_1,...]],[feature2,numpy.array],...]-->[[val1_1,val1_2,...],[val1_1,val2_2,...],...]
        """
        param_ranges = [list_2d[i][1] for i in range(len(list_2d))]
        param_features = [list_2d[i][0] for i in range(len(list_2d))]
        param_grids = np.meshgrid(*param_ranges)
        param_combinations = np.column_stack(
            [grid.ravel() for grid in param_grids])
        return param_features, param_combinations

    def run(self, print_step: bool = False):
        """
        运行遍历器。
        """
        case_list = []
        for i in range(len(self.feature_range)):
            x = self.feature_range[i]
            new_x = []
            for j, xx in enumerate(x):
                pre, suf = '', ''
                if j < len(self.prefix):
                    pre = self.prefix[j]
                if j < len(self.suffix):
                    suf = self.suffix[j]
                xx = f'{pre}{xx}{suf}'
            new_x.append(xx)
            replace_list = io.merge_feature(
                self.feature, new_x)
            target_path_name = f'/{self.test_name}'
            now = datetime.now()
            time = now.strftime("%Y%m%d%H%M%S")
            target_path_name = f'{target_path_name}_{time}'
            target_path_name = target_path_name[:24]
            print(self.CaseDir, self.source_path, target_path_name)
            case = io.Cases(self.program, self.CaseDir, self.source_path,
                            target_path_name, replace_list)
            case.new_case()
            case.run()
            case_list.append(case)
            if print_step:
                msg = ''
                for r_list in replace_list:
                    msg += f'{r_list[0]}={r_list[1]},'
                print(f'第{i+1}次遍历的参数：{msg}')
        return case_list
