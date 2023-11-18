# 导入必要的库
import os
import skopt
from . import CaseIO as io  # 假设这里是相对导入

class BayesOptimizer():
    def __init__(self, test_name: str, bashrc_path: str, features: list, CaseDir: str, source_path: str, func):
        """
        Bayes优化器的初始化函数。

        参数：
        - test_name: 测试名称
        - bashrc_path: bashrc文件路径
        - features: 特征列表
        - CaseDir: 案例目录
        - source_path: 源代码路径
        - func: 目标函数
        """
        self.test_name = test_name
        self.bashrc_path = bashrc_path
        self.features = features
        self.CaseDir = CaseDir
        self.source_path = source_path
        self.func = func
        os.system(f'source {self.bashrc_path}')  # 执行source命令，加载bashrc配置
        os.system('echo $MULTI')  # 打印环境变量MULTI的值

    def run(self, dimensions: list, delta: float = 1e-3, print_step: bool = False, n_calls: int = 5000, random_state: int = 0):
        """
        运行Bayes优化器。

        参数：
        - dimensions: 参数的范围
        - delta: 收敛判定的阈值
        - print_step: 是否打印每步的结果
        - n_calls: 最大迭代次数
        - random_state: 随机数生成器的种子

        返回：
        - res: 优化结果
        """
        # 定义回调函数，每次迭代后打印结果
        def onstep(res):
            print("最新尝试的参数：", res.x)
            print("最新尝试的函数值：", res.fun)
        # 定义回调函数，当函数值的改变小于0.01时停止优化
        delta_stop = skopt.callbacks.DeltaYStopper(delta)
        if print_step:
            res = skopt.gp_minimize(self.__bofunc_, dimensions, n_calls=n_calls, callback=[onstep, delta_stop], random_state=random_state)
        else:
            res = skopt.gp_minimize(self.__bofunc_, dimensions, n_calls=n_calls, callback=delta_stop, random_state=random_state)
        return res

    def __bofunc_(self, x):
        """
        Bayes优化的目标函数。

        参数：
        - x: 参数

        返回：
        - reward: 目标函数值
        """
        replace_list = io.merge_feature(self.features, x)
        target_path_name = f'{self.CaseDir}/{self.test_name}'
        for r_list in replace_list:
            target_path_name += f'_{r_list[0]}{r_list[1]}'
        target_path = os.path.abspath(f'{self.CaseDir}/{self.test_name}_')
        case = io.Cases(self.CaseDir, self.source_path, target_path, replace_list)
        case_p = case.run()
        case_p.wait()
        data = case.data_get()
        reward = self.func(data)
        return reward
