import pymulti as pm
import pymulti.Optimizer as opt
import pymulti.CaseIO as io
import numpy as np


def my_func(case: io.Cases, file_name, tag):
    data = case.get_data_tag(file_name, tag)
    reward = np.var(data)
    return reward


pm.init('multi_3d')
my_opt = opt.BayesOptimizer(program='multi_3d',
                            test_name='bo_test',
                            CaseDir='',
                            source_path='',
                            file_name='',
                            tag='',
                            features=['x', 'y', 'z'],
                            func=my_func)
