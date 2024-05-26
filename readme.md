# PyMulti Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

PyMulti is a script for user-friendly control of simulation program **MULTI**. There are tools to generate new cases with modified parameters as you want, based on given model case. Several evaluating functions can quantify the performance of your cases.  Some more advanced optimization for specific goals are also included to optimize parameters automatically. Visualization tools will also be available in the future.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [CaseIO Module](#caseio-module)
  - [BayesOptimizer Module](#bayesoptimizer-module)
- [Modules](#modules)
  - [CaseIO](#caseio)
  - [BayesOptimizer](#bayesoptimizer)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the `pymulti` library, clone the repository and install the required dependencies:

```bash
git clone https://github.com/astarrf/pymulti-for-fusion-simulation.git
cd pymulti
pip install -r requirements.txt
```

## Usage

### CaseIO Module

The `CaseIO` module provides functionality to manage and manipulate case files for various multi-program simulations.

#### Example

```python
from pymulti import init, Cases
import numpy as np

init('multi_3d')
my_case=Cases(program=Multi_Program.multi_3d, 
                       CaseDir='My/Case/Path', 
                       source_path='My/Source/Path', 
                       target_path='My/Target/Path',
                       replace_list=[["feature$0","value0"],
                                     ["feature$1","value1"],])
my_case.new_case()
my_case.run()
```

### BayesOptimizer Module

The `BayesOptimizer` module provides functionality to perform Bayesian optimization for multi-program simulations.

#### Example

```python
import pymulti as pm
import numpy as np


def my_func(case: pm.Cases, file_name, tag):
    data = case.get_data_tag(file_name, tag)
    reward = np.var(data)
    return reward

pm.init('multi_3d')
my_opt = pm.BayesOptimizer(program='multi_3d',
                            test_name='BO_test',
                            CaseDir='My/Case/Path',
                            source_path='My/Source/Path',
                            file_name='My/File/Name',
                            tag='MyTag',
                            features=['x', 'y', 'z'],
                            prefix=['prefix1','prefix2'],
                            suffix=['suffix1','suffix2'],
                            func=my_func)
res = my_opt.run(dimensions=[(-10.0, 10.0), (-10.0, 10.0),],
                 print_step=True,
                 do_delta_stop=False,
                 n_calls=150,
                 random_state=0,
                 n_jobs=8)
ans = res.x
```

## Modules

### CaseIO

The `CaseIO` module includes the following functionalities:

- `merge_feature(list1: list, list2: list) -> list`: Merges two lists into a list of feature-value pairs.
- `Cases` class: Manages case files with methods to copy, replace, and plot data.

#### Cases Class

**Initialization**

```python
Cases(program, CaseDir, source_path, target_path, replace_list=None, file_path='/User.r')
```

**Methods**

- `copy()`: Copies files from the source to the target directory.
- `replace()`: Replaces specified features with new values in the target files.
- `plotRhoTP()`: Plots the density and temperature over time.
- `get_coordinate(filename)`: Retrieves the coordinates from a file.

### BayesOptimizer

The `BayesOptimizer` module includes the following functionalities:

- `BayesOptimizer` class: Performs Bayesian optimization for multi-program simulations.

#### BayesOptimizer Class

**Initialization**

```python
BayesOptimizer(program, test_name, CaseDir, source_path, file_name, tag, features, func, prefix=[], suffix=[], sleep_T=1, run_time_count=2000)
```

**Methods**

- `run(dimensions, x0=None, y0=None, n_calls=100, random_state=None, n_jobs=5, do_delta_stop=False, delta=0.01, print_step=False, train_log=True, log_path=None)`: Runs the Bayesian optimizer.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.
```
Feel free to customize the sections and content further to better fit the specifics and additional functionalities of your library.