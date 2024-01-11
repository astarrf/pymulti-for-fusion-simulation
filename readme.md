# PyMulti

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI Version](https://img.shields.io/pypi/v/your-library-name.svg)](https://pypi.org/project/your-library-name/)
[![Python Versions](https://img.shields.io/pypi/pyversions/your-library-name.svg)](https://pypi.org/project/your-library-name/)
[![Build Status](https://travis-ci.org/your-username/your-library-name.svg?branch=master)](https://travis-ci.org/your-username/your-library-name)
[![Coverage Status](https://coveralls.io/repos/github/your-username/your-library-name/badge.svg?branch=master)](https://coveralls.io/github/your-username/your-library-name?branch=master)

Short description of your library.

## Features

- Easier new cases spawnning
- Batch process of all cases in one programme
- Easier data processing
- Suitable for Multi 1D/2D/3D

## Installation

Copy the whole directory to your own directory,
Use ```import pymulti``` to import the package

### PyMulti.CaseIO
A module for controlling a single case, from initializing a new case to get any data.

A demo for using the module is:
```python
import pymulti as pm
import pymulti.CaseIO as io
import numpy as np

pm.init('multi_3d')
my_case=io.Cases(program=Multi_Program.multi_3d, 
                       CaseDir='My/Case/Path', 
                       source_path='My/Source/Path', 
                       target_path='My/Target/Path',
                       replace_list=[["feature$0","value0"],
                                     ["feature$1","value1"],])
my_case.new_case()
my_case.run()
```

A demo of using the ```pymulti.optimizer``` is:
```python
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
                            test_name='BO_test',
                            CaseDir='My/Case/Path',
                            source_path='My/Source/Path',
                            file_name='My/File/Name',
                            tag='MyTag',
                            features=['x', 'y', 'z'],
                            func=my_func)
res = my_opt.run(dimensions=[(-10.0, 10.0), (-20.0, 20.0),],
                 delta=1e-3,
                 print_step=False,
                 do_delta_stop=True,
                 n_calls=5000,
                 random_state=0)
ans = res.x
```