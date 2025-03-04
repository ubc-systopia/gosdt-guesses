# Fast Sparse Decision Tree Optimization via Reference Ensembles

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
![example workflow](https://github.com/ubc-systopia/gosdt-guesses/actions/workflows/main.yml/badge.svg)

**This project is the cannonical implementation of the following papers**:
- McTavish, H., Zhong, C., Achermann, R., Karimalis, I., Chen, J., Rudin, C., & Seltzer, M. (2022). Fast Sparse Decision Tree Optimization via Reference Ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 36(9), 9604-9613. https://doi.org/10.1609/aaai.v36i9.21194
- Jimmy Lin, Chudi Zhong, Diane Hu, Cynthia Rudin, and Margo Seltzer. 2020. Generalized and scalable optimal sparse decision trees. In Proceedings of the 37th International Conference on Machine Learning (ICML'20), Vol. 119. JMLR.org, Article 571, 6150–6160.

A [scikit-learn](https://scikit-learn.org) compatible library for generating Optimal Sparse Decision Trees.
It is a direct competitor of CART[[3](#related-work)] and C4.5[[6](#related-work)], as well as DL8.5[[1](#related-work)], BinOct[[7](#related-work)], and OSDT[[4](#related-work)].
Its advantage over CART and C4.5 is that the trees are globally optimized, not constructed just from the top down. 
This makes it slower than CART, but it provides better solutions. 
On the other hand, it tends to be faster than other optimal decision tree methods because it uses bounds to limit the search space, and uses a black box model (a boosted decision tree) to “guess” information about the optimal tree.
It takes only seconds or a few minutes on most datasets.

To make it run faster, please use the options to limit the depth of the tree, and increase the regularization parameter above 0.02.
If you run the algorithm without a depth constraint or set the regularization too small, it will run more slowly.

This work builds on a number of innovations for scalable construction of optimal tree-based classifiers: Scalable Bayesian Rule Lists[[8](#related-work)], CORELS[[2](#related-work)], OSDT[[4](#related-work)].

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Example](#example)
- [Frequently Asked Questions](#faq)
- [How to build the project](#how-to-build-the-project)
- [Project Versioning](#project-versioning)
- [Project Structure](#project-structure)
- [Debugging](#debugging)
- [Related Work](#related-work)

## Installation

GOSDT is available on [PyPI](https://pypi.org/project/gosdt/) and can thus be easily installed using pip.

```bash
pip3 install gosdt
```

Note: Our x86_64 wheels all use modern ISA extensions such AVX to perform fast bitmap operations. 
If you're running on an older system where that's not possible, we recommend that you build from source following the [instructions below](#how-to-build-the-project).

## Configuration

When initializing a `GOSDTClassifier` object, the following hyperparameters can be specified: 

    regularization : float, default=0.05
        The regularization penalty incurred for each leaf in the model. We 
        highly recommend setting the regularization to a value larger than 
        1 / (# of samples). A small regularization will lead to a longer 
        training time. If a smaller regularization (than 1 / (# of samples)) is
        preferredm you mus set the parameter `allow_small_reg` to True, which
        by default is False.

    allow_small_reg : bool, default=False
        Boolean flag for allowing a regularization that's less than 1 / (# of samples).
        If False the effective regularization is bounded below by 1 / (# of samples).

    depth_budget : int | None, default=None
        Sets the maximum tree depth for a solution model, counting a tree with just 
        the root node as a tree of depth 0 

    time_limit: int | None, default=None
        A time limit (in seconds) upon which the algorithm will terminate. If
        the time limit is reached without a solution being found, the algorithm will terminate with an error.

    balance: bool, default=False
        A boolean flag enabling overriding the sample importance by equalizing the importance of each present class.

    cancellation: bool, default=True
        A boolean flag enabling the propagation of task cancellations up the dependency graph.

    look_ahead: bool, default=True
        A boolean flag enabling the one-step look-ahead bound implemented via scopes.

    similar_support: bool, default=True
        A boolean flag enabling the similar support bound implemented via a distance index.

    rule_list: bool, default=False
        A boolean flag enabling rule-list constraints on models.

    non_binary: bool, default=False
        A boolean flag enabling non-binary model trees. 
        #todo(Ilias: Our tree parser does not currently handle this flag)        

    diagnostics: bool, default=False
        A boolean flag enabling printing of diagnostic traces when an error is encountered.
        This is intended for debugging the C++ logic and is not intended for end-user use.

    model_limit: int, default=1
        The maximum number of optimal models to extract, in the case of multiple optima.
        
    debug: bool, default=False
        A boolean flag that enables saving the state of the optimization, so that it can be
        inspected or ran again in the future. This is intended for debugging the C++ logic and 
        is not intended for end-user use.

When calling `fit`, the following arguments are available: 

        X : array-like of shape (n_samples, n_features)
            The training input samples. Boolean values are expected.

        y : array-like of shape (n_samples,)
            The target values. The target values can be binary or multiclass.
        
        input_features : array-like of shape (n_features,) | None, default=None
            The feature names for the input data. If None, the feature names will be set to ["x0", "x1", ...].
            
        y_ref : array-like of shape (n_samples,) | None, default=None
            Theese represent the predictions made by some blackbox model, that will be used to guide optimization.
            The reference labels can be binary or multiclass, but must have the same classes and shape as y.

        cost_matrix : array-like of shape (n_classes, n_classes) | None, default=None
            The cost matrix for the optimization. If None, a cost matrix will be created based on 
            the number of classes and whether a balanced cost matrix is requested.

## Example

This is a classification example using GOSDT with threshold guessing, lower bound guessing and a depth limit.
Additional examples and notebooks are available in the [`examples/`](./examples/) folder.

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from gosdt import ThresholdGuessBinarizer, GOSDTClassifier

# Parameters
GBDT_N_EST = 40
GBDT_MAX_DEPTH = 1
REGULARIZATION = 0.001
SIMILAR_SUPPORT = False
DEPTH_BUDGET = 6
TIME_LIMIT = 60
VERBOSE = True

# Read the dataset
df = pd.read_csv("datasets/compas.csv", sep=",")
X, y = df.iloc[:, :-1], df.iloc[:, -1]
h = df.columns[:-1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)
print("X train shape:{}, X test shape:{}".format(X_train.shape, X_test.shape))

# Step 1: Guess Thresholds
X_train = pd.DataFrame(X_train, columns=h)
X_test = pd.DataFrame(X_test, columns=h)
enc = ThresholdGuessBinarizer(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=2021)
enc.set_output(transform="pandas")
X_train_guessed = enc.fit_transform(X_train, y_train)
X_test_guessed = enc.transform(X_test)
print(f"After guessing, X train shape:{X_train_guessed.shape}, X test shape:{X_test_guessed.shape}")
print("train set column names == test set column names: {list(X_train_guessed.columns)==list(X_test_guessed.columns)}")

# Step 2: Guess Lower Bounds
enc = GradientBoostingClassifier(n_estimators=GBDT_N_EST, max_depth=GBDT_MAX_DEPTH, random_state=42)
enc.fit(X_train_guessed, y_train)
warm_labels = enc.predict(X_train_guessed)

# Step 3: Train the GOSDT classifier
clf = GOSDTClassifier(regularization=REGULARIZATION, similar_support=SIMILAR_SUPPORT, time_limit=TIME_LIMIT, depth_budget=DEPTH_BUDGET, verbose=VERBOSE) 
clf.fit(X_train_guessed, y_train, y_ref=warm_labels)

# Step 4: Evaluate the model
print("Evaluating the model, extracting tree and scores", flush=True)


print(f"Model training time: {clf.result_.time}")
print(f"Training accuracy: {clf.score(X_train_guessed, y_train)}")
print(f"Test accuracy: {clf.score(X_test_guessed, y_test)}")
```


## FAQ

- **Does GOSDT (implicitly) restrict the depth of the resulting tree?**

    As of 2022, GOSDT With Guesses can now restrict the depth of the resulting tree both implicitly and explicitly. Our primary sparsity constraint is the regularization parameter (lambda) which is used to penalize the number of leaves. As lambda becomes smaller, the generated trees will have more leaves, but the number of leaves doesn't guarantee what depth a tree has since GOSDT generates trees of any shape. As of our 2022 AAAI paper, though, we've allowed users to restrict the depth of the tree. This provides more control over the tree's shape and reduces the runtime. However, the depth constraint is not a substitute for having a nonzero regularization parameter! Our algorithm achieves a better sparsity-accuracy tradeoff, and saves time, with a well-chosen lambda.

- **Why does GOSDT run for a long time when the regularization parameter (lambda) is set to zero?**

    The running time depends on the dataset itself and the regularization parameter (lambda). In general, setting lambda to 0 will make the running time longer. Setting lambda to 0 is kind of deactivating the branch-and-bound in GOSDT. In other words, we are kind of using brute force to search over the whole space without effective pruning, though dynamic programming can help for computational reuse. In GOSDT, we compare the difference between the upper and lower bound of a subproblem with lambda to determine whether this subproblem needs to be further split. If lambda=0, we can always split a subproblem. Therefore, it will take more time to run. It usually does not make sense to set lambda smaller than 1/n, where n is the number of samples.

- **Is there a way to limit the size of the produced tree?**

    Regularization parameter (lambda) is used to limit the size of the produced tree (specifically, in GOSDT, it limits the number of leaves of the produced tree). We usually set lambda to [0.1, 0.05, 0.01, 0.005, 0.001], but the value really depends on the dataset. One thing that might be helpful is considering how many samples should be captured by each leaf node. Suppose you want each leaf node to contain at least 10 samples. Then setting the regularization parameter to 10/n is reasonable. In general, the larger the value of lambda is, the sparser a tree you will get.

- **In general, how does GOSDT set the regularization parameter?**

    GOSDT aims to find an optimal tree that minimizes the training loss with a penalty on the number of leaves. The mathematical description is min loss+lambda*# of leaves. When we run GOSDT, we usually set lambda to different non-zero values and usually not smaller than 1/n. On page 31 Appendix I.6 in our ICML paper, we provide detailed information about the configuration we used to run accuracy vs. sparsity experiments.

## How to build the project

### Step 1: Install required development tools

GOSDT uses `CMake` as its default cross-platform build system and `Ninja` as the default generator for parallel builds.
GOSDT relies on `pybind11` for Python bindings and `scikit-build-core` as its Python meta build system for the generation of wheel files.
GOSDT also uses `delocate`, `auditwheel` and `delvewheel` to copy all required 3rd-party dynamic libraries into the wheel archive (Each of these different tools are used for the three OS platforms we support: macOS, Linux, and Windows).

Note that `delocate`, `auditwheel` and `delvewheel` are only only needed if building wheels for deployment to PyPI.

**macOS**:
We rely on the [brew](https://brew.sh/) package manager to install 3rd-party libraries and dependencies.

```sh
brew install cmake ninja pkg-config
pip3 install --upgrade scikit-build-core pybind11 delocate
```

**Ubuntu:**

```sh
sudo apt install -y cmake ninja-build pkg-config patchelf
pip3 install --upgrade scikit-build-core pybind11 auditwheel
```

**Windows:**
Please make sure that you launch Powershell as Admin.

**Step 1.1.:** Install Chocolatey

In addition to Windows Package Manager (a.k.a. `winget`), [Chocolatey](https://chocolatey.org/) is used to install tools that are not yet provided by `winget`.
Please follow this [guide](https://chocolatey.org/install#individual) or use the following commands to install Chocolatey.

```ps1
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

**Step 1.2:** Install vcpkg

GOSDT requires the C++ package manager `vcpkg` to install all necessary C and C++ libraries on Windows.
Please follow this [guide](https://vcpkg.io/en/getting-started.html) or use the following commands to install `vcpkg` to `C:\vcpkg`.

```ps1
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
.\vcpkg\bootstrap-vcpkg.bat
```

Once you have installed `vcpkg`, for example, to `C:\vcpkg`, you need to...
- Update your `PATH` variable to include `C:\vcpkg`.
- Add a new environment variable `VCPKG_INSTALLATION_ROOT` with a value of `C:\vcpkg`.

The following Powershell script modifies the system environment permanently.
In other words, all users can see these two new variables.

```ps1
$vcpkg = "C:\vcpkg"
$old = (Get-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH).path
$new = "$old;$vcpkg"
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $new
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name VCPKG_INSTALLATION_ROOT -Value $vcpkg
```

You can verify whether the new variable `VCPKG_INSTALLATION_ROOT` is set properly by typing the following command in Powershell:
Note that you may need to restart your terminal or reboot your computer to apply the changes.

```ps1
$ENV:VCPKG_INSTALLATION_ROOT
```

**Step 1.3:** Install required development tools

```ps1
winget install Kitware.CMake
choco install -y ninja
choco install -y pkgconfiglite
pip3 install --upgrade scikit-build
pip3 install --upgrade delvewheel
```

If `choco install -y pkgconfiglite` reports SSL/TLS-related errors, please use the following commands to install it manually.

```ps1
Invoke-WebRequest -UserAgent "Wget" -Uri "http://downloads.sourceforge.net/project/pkgconfiglite/0.28-1/pkg-config-lite-0.28-1_bin-win32.zip" -OutFile "C:\pkgconfig.zip"
Expand-Archive "C:\pkgconfig.zip" -DestinationPath "C:\"
Copy-Item "C:\pkg-config-lite-0.28-1\bin\pkg-config.exe" "C:\Windows"
Remove-Item "C:\pkg-config-lite-0.28-1" -Force -Recurse
Remove-Item "C:\pkgconfig.zip"
```

### Step 2: Install required 3rd-party libraries

GOSDT relies on `IntelTBB` for concurrent data structures, and `GMP` for fast bitwise operations.

**macOS:**

```bash
brew install tbb gmp
```

**Ubuntu:**

```bash
sudo apt install -y libtbb-dev libgmp-dev
```

**Windows:**

```ps1
vcpkg install tbb:x64-windows
vcpkg install gmp:x64-windows
```

### Step 3: Build the project

There are two main methods for building the project:

- for local use (and development).
- wheel generation for distribution and deployment to PYPI.

The following assumes that your working directory is the project root.

#### Method 1: Local use/development and debugging

Note that the following command also produces the `gosdt_cli` target that can be used to hook `gdb` or other debuggers/sanitizers into the GOSDT C++ implementation.
The project build artifacts can be found in the `pyproject-build` directory. 

```bash
pip3 install .
```

#### Method 2: Wheel generation

**macOS**:

```zsh
pip3 wheel --no-deps . -w dist/
delocate-wheel -w dist -v dist/gosdt-*.whl
```

**Windows:**

```ps1
pip3 wheel --no-deps . -w dist/
python3 -m delvewheel repair --no-mangle-all --add-path "$ENV:VCPKG_INSTALLATION_ROOT\installed\x64-windows\bin" dist/gosdt-1.0.5-cp310-cp310-win_amd64.whl -w dist
```

**Ubuntu:**

We are using Ubuntu as the host, but manylinux wheel building relies on Docker.
For this specific target we recommend using the `cibuildwheel` tool which hides some of the painful details related to wheel building for manylinux.

```bash
pipx run cibuildwheel
```

## Project Versioning

This project uses the `setuptools_scm` tool to perform project versioning for the built Python library.
This tool uses git commit tags to pick a reasonable version number.
In the odd case where one wishes to build this project while removing all references to the git repo, there is a patch for [`pyproject.toml`](./pyproject.toml) included at [`scripts/non_git.patch`](./scripts/non_git.patch)

## Project Structure

This repository contains the following directories:

- **datasets**: Datasets that are used to generate plots and figures from the papers and run examples.
- **examples**: Contains sample code and notebooks demonstrating how to use the gosdt library.
- **scripts/build**: Scripts used while building the python library.
- **scripts/gosdt**: Scripts to replicate the plots and figures from the ICML 2020 paper[[5](#related-work)].
- **scripts/gosdt-guesses**: Scripts to replicate the plots and figures from the AAAI 2022 paper[[9](#related-work)]
- **src**: Implementation of the gosdt Python and C++ library.
- **tests**: pytest tests for the Python library.

## Debugging

Unfortunately, there is no super clear way to run `gdb` on the final `python` version of the project, for this we've added the `debug` configuration option to the `GOSDTClassifier` class.
Enabling this flag will create a `debug_(current time)` directory that is populated with a serialized copy of the `Dataset` and `Configuration` classes as well as some other useful debugging information.

We additionaly provide the `gosdt_cli` target which is a C++ terminal application that takes as an argument the path to a `debug_*` directory and runs the GOSDT algorithm on it.
This enables the developper to use standard C++ tools to debug the algorithm, such as debuggers or sanitizers.

An example that uses the debug flag can be found at [`examples/debug.py`](examples/debug.py)

## Related Work

[1] Aglin, G.; Nijssen, S.; and Schaus, P. 2020. Learning optimal decision trees using caching branch-and-bound search. In _AAAI Conference on Artificial Intelligence_, volume 34, 3146–3153.

[2] Angelino, E.; Larus-Stone, N.; Alabi, D.; Seltzer, M.; and Rudin, C. 2018. Learning Certifiably Optimal Rule Lists for Categorical Data. _Journal of Machine Learning Research_, 18(234): 1–78.

[3] Breiman, L.; Friedman, J.; Stone, C. J.; and Olshen, R. A. 1984. _Classification and Regression Trees_. CRC press.

[4] Hu, X.; Rudin, C.; and Seltzer, M. 2019. Optimal sparse decision trees. In _Advances in Neural Information Processing Systems_, 7267–7275.

[5] Lin, J.; Zhong, C.; Hu, D.; Rudin, C.; and Seltzer, M. 2020. Generalized and scalable optimal sparse decision trees. In _International Conference on Machine Learning (ICML)_, 6150–6160.

[6] Quinlan, J. R. 1993. C4.5: _Programs for Machine Learning_. Morgan Kaufmann

[7] Verwer, S.; and Zhang, Y. 2019. Learning optimal classification trees using a binary linear program formulation. In _AAAI
Conference on Artificial Intelligence_, volume 33, 1625–1632.

[8] Yang, H., Rudin, C., & Seltzer, M. (2017, July). Scalable Bayesian rule lists. In _International Conference on Machine Learning (ICML)_ (pp. 3921-3930). PMLR.

[9] McTavish, H., Zhong, C., Achermann, R., Karimalis, I., Chen, J., Rudin, C., & Seltzer, M. (2022). Fast Sparse Decision Tree Optimization via Reference Ensembles. _Proceedings of the AAAI Conference on Artificial Intelligence_, 36(9), 9604-9613.
