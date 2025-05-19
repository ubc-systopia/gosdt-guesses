# Python standard library imports
import json

# External imports
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from typing import Optional

# GOSDT imports
from ._libgosdt import BoolMatrix, FloatMatrix, Configuration, Dataset, gosdt_fit, Status
from ._tree import Tree


class GOSDTClassifier(ClassifierMixin, BaseEstimator):
    f"""
    Decision Tree Classifier that produces optimal sparce decision trees.

    For more information on the method used please refer to the following papers:
    - "Generalized and Scalable Optimal Sparse Decision Trees"
       https://doi.org/10.48550/arXiv.2006.08690
    - "Fast Sparse Decision Tree Optimization via Reference Ensembles"
       https://doi.org/10.1609/aaai.v36i9.21194

    Attributes
    __________
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
        
    debug: bool, default=False
        A boolean flag that enables saving the state of the optimization, so that it can be
        inspected or ran again in the future. This is intended for debugging the C++ logic and 
        is not intended for end-user use.

    Examples
    --------
    A minimal example with the well known Iris dataset.

    >>> from gosdt import ThresholdGuessBinarizer, GOSDTClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True, as_frame=True)
    >>> X_bin = ThresholdGuessBinarizer().fit_transform(X, y)
    >>> clf = GOSDTClassifier(regularization=0.1, verbose=True)
    >>> clf.fit(X_bin, y)
    """

    def __init__(
            self,
            regularization: float = 0.05,
            allow_small_reg: bool = False,
            depth_budget: Optional[int] = None,
            time_limit: Optional[int] = None,
            balance: bool = False,
            cancellation: bool = True,
            look_ahead: bool = True,
            similar_support: bool = True,
            rule_list: bool = False,
            non_binary: bool = False,
            diagnostics: bool = False,
            uncertainty_tolerance: float = 0,
            upperbound_guess: Optional[float] = None,
            model_limit: int = 1,
            worker_limit: int = 1,
            verbose: bool = False,
            debug: bool = False,
    ):
        self.regularization = regularization
        if regularization < 0:
            raise ValueError("regularization must be non-negative")

        self.allow_small_reg = allow_small_reg

        self.depth_budget = depth_budget
        if depth_budget is not None and depth_budget < 0:
            raise ValueError("depth_budget must be non-negative")

        self.time_limit = time_limit
        if time_limit is not None and time_limit < 0:
            raise ValueError("time_limit must be non-negative")

        self.balance = balance
        self.cancellation = cancellation
        self.look_ahead = look_ahead
        self.similar_support = similar_support
        self.rule_list = rule_list
        self.non_binary = non_binary
        self.diagnostics = diagnostics
        self.verbose = verbose
        self.debug = debug

        self.uncertainty_tolerance = uncertainty_tolerance
        if uncertainty_tolerance < 0:
            raise ValueError("uncertainty_tolerance must be non-negative")

        self.upperbound_guess = upperbound_guess
        if upperbound_guess is not None and (upperbound_guess < 0 or upperbound_guess > 1):
            raise ValueError("upperbound_guess must be between 0 and 1")

        self.model_limit = model_limit
        if model_limit < 0:
            raise ValueError("model_limit must be non-negative")

        self.worker_limit = worker_limit
        if worker_limit < 0:
            raise ValueError("worker_limit must be non-negative")
        if worker_limit > 1:
            print("[WARNING] A worker_limit > 1 has been chosen."
                  " The current release of GOSDT displays some deadlocking issues."
                  " Use this setting at your own risk.")



    def fit(self, X, y, y_ref=None, input_features=None, cost_matrix=None, feature_map=None):
        """
        Fit the GOSDTClassifier to X, y.

        Parameters
        ----------
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

        feature_map : list of sets of ints | None, default=None
            The feature map is a list that maps original feature indices to the sets of binarized features that
            represent them. If None, the feature map will be set to the trivial feature map. This is not currently
            being used, but could in the future be used to produce N-ary trees.

        Raises
        ------
        Value Error
            If the number of feature names does not match the number of features, or if the length of the y_ref vector
            does not match the length of the y vector, or if the classes in y_ref do not match the classes in y.
            Additionally if X is not boolean, or if the number of features in the feature_map does not match the number
            of features in X.
        
        TimeoutError
            If the optimization does not finish within the time limit. In this case the partial result is preserved, but
            is not provably optimal.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the feature names
        self.feature_names_ = input_features
        if hasattr(X, "columns"):
            self.feature_names_ = X.columns

        # Check that X and y have the correct shape
        X, y = check_X_y(X, y)
        X = check_array(X, ensure_2d=True, dtype=bool)
        y = check_array(y, ensure_2d=False, dtype=None)
        (n, m) = X.shape

        # Continue with feature names validation now that we know that X is an arraylike (numpy array).
        if self.feature_names_ is None:
            self.feature_names_ = [f"x{i}" for i in range(X.shape[1])]
        elif len(self.feature_names_) != X.shape[1]:
            raise ValueError(
                f"Number of feature names ({len(self.feature_names_)}) does not match number of features ({X.shape[1]}).")


        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Store dataset information
        self.n_samples_, self.n_features_ = X.shape
        self.n_classes_ = len(self.classes_)

        # If y_ref is provided, check that it has the correct shape
        if y_ref is not None:
            if len(y_ref) != len(y):
                raise ValueError(
                    f"y_ref must have the same length as y, but has length {len(y_ref)}.")

            y_ref = check_array(y_ref, ensure_2d=False, dtype=None)
            if set(unique_labels(y_ref)) != set(self.classes_):
                raise ValueError(
                    f"y_ref must have the same classes as y ({self.classes_}), but has classes: {unique_labels(y_ref)}.")
            self.reference_LB = True # set flag to indicate using reference LB (needed for correct scope behaviour)
        else:
            self.reference_LB = False

        # Check that X is boolean
        if not np.all(np.logical_or(X == 0, X == 1)):
            raise ValueError(
                f"X must be boolean, but contains values other than 0 and 1."
            )

        # Check that regularization is large enough to be effective based on the number of samples:
        if self.regularization < 1.0 / n:
            print(f"[WARNING] A regularization was chosen that is less than 1 / (# of samples) = {1.0 / n}."
                  f" This may lead to a longer training time if not adjusted.")
            if not self.allow_small_reg:
                print(f"[WARNING] Regularization increased to 1 / (# of samples) = {1.0 / n}."
                      f" If you would like to continue with your chosen regularization ({self.regularization}),"
                      f" please set allow_small_reg=True.")
                self.regularization = 1.0 / n

        # Validate the feature_map
        if feature_map is None:
            # The trivial feature_map
            feature_map = [set([i]) for i in range(self.n_features_)]

        feature_map_size = sum([len(s) for s in feature_map])
        if feature_map_size != self.n_features_:
            raise ValueError(
                f"Number of features in the feature_map ({feature_map_size}) does not match number of features ({self.n_features_}).")
        
                
        # If y only has one class, we return the trivial model that alway predicts that class.
        if len(self.classes_) == 1:

            # We forge a result object that has the same structure as the one returned by the C++ code.
            #
            # For some reason, I'm having some trouble instantiatin a proper gosdt::Result object here, but a SimpleNamespace
            # should suffice for our use. This could cause issues in the future if we need the actual C++ objects! 
            # TODO(Ilias: change this to a proper gosdt::Result object if possible.)
            from types import SimpleNamespace
            self.result_ = SimpleNamespace()
            self.result_.model = json.dumps([{"prediction": 0, "loss": 0}])
            self.result_.graph_size = 0
            self.result_.queue_size = 0
            self.result_.n_iterations = 0
            self.result_.lowerbound = 0
            self.result_.upperbound = 0
            self.result_.model_loss = 0
            self.result_.time = 0
            self.result_.status = Status.UNINITIALIZED

            # Write a tree object that will always predict the single class.
            json_result = json.loads(self.result_.model)
            assert len(json_result) == 1, "The forged model should only have one tree."
            self.trees_ = [Tree(json_result[0], self.feature_names_, self.n_classes_, self.classes_)]
    
            return self

        # Binarize the y vector and create a cost matrix if one was not provided
        y_bin = np.ndarray(shape=(len(y), len(self.classes_)), dtype=bool)
        for i in range(len(y)):
            y_bin[i] = self.classes_ == y[i]

        # Create a Boolean matrix from X and y
        input_matrix = self.__create_input_matrix(X, y)

        # Create a cost Float matrix
        cost_matrix = self.__create_cost_matrix(y_bin, cost_matrix)

        # Create the Configuration object.
        self.__create_configuration()

        # Create the Dataset
        dataset: Dataset
        reference_labels = None
        if y_ref is not None:
            reference_labels = self.__create_reference_labels(y_ref)
            dataset = Dataset(self.config_, input_matrix,
                              cost_matrix, feature_map, reference_labels)
        else:
            dataset = Dataset(self.config_, input_matrix,
                              cost_matrix, feature_map)
            
        # If the debug flag is set, we save the state of the optimization, so that it can be
        # inspected or ran again in the future.
        if self.debug:
            self.__save_debug_state(X, y, y_ref, dataset)            

        # Call the GOSDT C++ logic
        self.result_ = gosdt_fit(dataset)

        # Check the gosdt result status and report errors as needed.
        if self.result_.status == Status.UNINITIALIZED:
            raise RuntimeError("[ERROR] Optimization never started.")
        elif self.result_.status == Status.FALSE_CONVERGENCE:
            raise RuntimeError("[ERROR] Optimization shows false convergence, no model was found.")
        elif self.result_.status == Status.NON_CONVERGENCE:
            print("[WARNING] Optimization did not converge, the result may not be optimal.")
        elif self.result_.status == Status.TIMEOUT:
            print("[WARNING] Optimization did not finish within the time limit, the result may not be optimal.")
        
        # Parse the result into a list of Tree objects
        json_result = json.loads(self.result_.model)
        self.trees_ = [Tree(res, self.feature_names_, self.n_classes_, self.classes_)
                       for res in json_result]

        return self

    def get_result(self):
        # Check if fit has been called.
        check_is_fitted(self, ['result_', 'trees_'])

        return {
            "models_string": self.result_.model,
            "graph_size": self.result_.graph_size,
            "n_iterations": self.result_.n_iterations,
            "lower_bound": self.result_.lowerbound,
            "upper_bound": self.result_.upperbound,
            "model_loss": self.result_.model_loss,
            "time": self.result_.time,
            "status": self.result_.status,
        }

    def predict(self, X, model_number=0):
        """
        Predict the class for each sample in X.s
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Boolean values are expected.
            
        model_number : int, default=0
            The model number to use for prediction. If multiple models were extracted, 
            this parameter can be used to select a specific model.
        
        Raises
        ------
        ValueError
            If the model_number is greater than the number of models extracted or the set model_limit parameter.
        
        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.s
        """

        # Check if fit has been called.
        check_is_fitted(self, ['result_', 'trees_'])

        if model_number > 0 and self.model_limit == 1:
            raise ValueError(
                f"Requesting a prediction from model {model_number} but the configuration option (model_limit) is set to 1.")

        if model_number >= len(self.trees_):
            raise ValueError(
                f"Requesting a prediction from model {model_number} but only {len(self.trees_)} models were extracted.")

        # Validate X
        X = check_array(X, ensure_2d=True, force_all_finite=True)

        return self.trees_[model_number].predict(X)

    def predict_proba(self, X, model_number=0):
        # TODO: Note that there's two ways to do this:
        # 1. Use the predict_proba function over all extracted models and then average to get a probability distribution.
        # 2. Use the predict_proba function over one specific chosen model.
        if model_number > 0 and self.model_limit == 1:
            raise ValueError(
                f"Requesting a prediction from model {model_number} but the configuration option (model_limit) is set to 1.")

        if model_number >= len(self.trees_):
            raise ValueError(
                f"Requesting a prediction from model {model_number} but only {len(self.trees_)} models were extracted.")

        # Validate X
        X = check_array(X, ensure_2d=True, dtype=bool)

        return self.trees_[model_number].predict_proba(X)

    def __create_configuration(self):
        self.config_ = Configuration()
        self.config_.regularization = self.regularization
        # Here we compute the off-by-one translation of depth_budget
        if self.depth_budget is not None:
            self.config_.depth_budget = self.depth_budget + 1
        else:
            # Set unlimited depth_budget
            self.config_.depth_budget = 0
        if self.time_limit is not None:
            self.config_.time_limit = self.time_limit
        self.config_.cancellation = self.cancellation
        self.config_.look_ahead = self.look_ahead
        self.config_.similar_support = self.similar_support
        self.config_.rule_list = self.rule_list
        self.config_.non_binary = self.non_binary
        self.config_.diagnostics = self.diagnostics
        self.config_.verbose = self.verbose
        if self.upperbound_guess is not None:
            self.config_.upperbound_guess = self.upperbound_guess
        self.config_.model_limit = self.model_limit
        self.config_.worker_limit = self.worker_limit
        self.config_.reference_LB = self.reference_LB

    def __create_cost_matrix(self, y, custom_cost_matrix: None) -> FloatMatrix:
        cost_matrix = custom_cost_matrix
        if cost_matrix is not None:
            # custom cost matrix validation
            cost_matrix = check_array(cost_matrix, ensure_2d=True, dtype=float)
            if cost_matrix.shape != (self.n_classes_, self.n_classes_):
                raise ValueError(
                    f"cost_matrix must be a square matrix of size number of classes ({self.n_classes_}).")
        else:
            # Create a cost matrix with the same number of rows and columns as the number of classes
            cost_matrix = np.ones((self.n_classes_, self.n_classes_))
            # Set the diagonal to 0
            np.fill_diagonal(cost_matrix, 0)
            if self.balance:
                # Set the off-diagonal to 1 / (class count * self.n_classes_)
                class_counts = np.sum(y, axis=0)
                for i in range(self.n_classes_):
                    for j in range(self.n_classes_):
                        if i != j:
                            cost_matrix[i, j] = 1.0 / \
                                (class_counts[j] * self.n_classes_)

            else:
                # Set the off-diagonal to 1 / self.n_samples_
                cost_matrix[cost_matrix == 1] = 1.0 / self.n_samples_

        # Create Float matrix from cost matrix
        fm = FloatMatrix(self.n_classes_, self.n_classes_)
        fm_np_array = np.array(fm, copy=False)
        np.copyto(fm_np_array, cost_matrix)
        return fm

    def __create_reference_labels(self, y_ref) -> BoolMatrix:
        # Trasform y_ref into a binary matrix
        y_ref_bin = self.label_binarizer_.transform(y_ref) > 0.5
        (n, m) = y_ref_bin.shape

        # In the special case where y_ref only had two types we need to add the second row that's equal to it's negation:
        if m == 1:
            y_ref_bin = np.concatenate((~y_ref_bin, y_ref_bin), axis=1)
            m = 2

        # Create a Boolean matrix from the numpy array
        bm = BoolMatrix(n, m)
        bm_np_array = np.array(bm, copy=False)
        np.copyto(bm_np_array, y_ref_bin)

        return bm

    def __create_input_matrix(self, X, y) -> BoolMatrix:
        # Trasform y into a binary matrix
        self.label_binarizer_ = LabelBinarizer()
        y_bin = self.label_binarizer_.fit_transform(y) > 0.5

        # In the special case where y only had two types we need to add the second row that's equal to it's negation:
        if y_bin.shape[1] == 1:
            y_bin = np.concatenate((~y_bin, y_bin), axis=1)

        # Collate X_bin and y_bin as a numpy array
        Xy_bin = np.concatenate((X, y_bin), axis=1)
        (n, m) = Xy_bin.shape

        # Create a Boolean matrix from the numpy array
        bm = BoolMatrix(n, m)
        bm_np_array = np.array(bm, copy=False)
        np.copyto(bm_np_array, Xy_bin)

        return bm

    def __save_debug_state(self, X, y, y_ref, dataset):
        # Create a folder to save the state
        import os
        import time
        dbg_folder = f"debug_{int(time.time())}"
        os.makedirs(dbg_folder, exist_ok=False)
        
        np.savetxt(f"{dbg_folder}/X.csv", X, delimiter=",")
        np.savetxt(f"{dbg_folder}/y.csv", y, delimiter=",")
        if y_ref is not None:
            np.savetxt(f"{dbg_folder}/y_ref.csv", y_ref, delimiter=",")
        np.savetxt(f"{dbg_folder}/feature_names.csv", np.array(self.feature_names_), delimiter=",", fmt="%s")
                
        # Save the dataset to a file.
        self.config_.save(f"{dbg_folder}/config.json")
        dataset.save(f"{dbg_folder}/dataset.bin")
