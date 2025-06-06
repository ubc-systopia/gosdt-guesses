# External imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_X_y
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_array, check_is_fitted
    
class ThresholdGuessBinarizer(TransformerMixin, BaseEstimator):
    f"""
    Encode numerical features as a one-hot numeric array. 
    The encoding is based on the thresholds found by a Gradient Boosting Classifier.

    For more information on the method used please refer to the following paper:
    "Fast Sparse Decision Tree Optimization via Reference Ensembles"
    https://doi.org/10.1609/aaai.v36i9.21194
    
    Parameteres
    ----------
    learning_rate : float, default=0.1
        The chosen learning rate for the Gradient Boosting Classifier.
        For more information please refer to the documentation of the Gradient Boosting Classifier 
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

    n_estimators : int, default=100
        The number of estimators for the Gradient Boosting Classifier.
        For more information please refer to the documentation of the Gradient Boosting Classifier 
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

    max_depth : int, default=3
        The maximum depth of the trees for the Gradient Boosting Classifier.
        For more information please refer to the documentation of the Gradient Boosting Classifier 
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

    random_state : int, default=0
        The random state for the Gradient Boosting Classifier.
        For more information please refer to the documentation of the Gradient Boosting Classifier 
        (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

    column_elimination : bool, default=True
        Whether to perform column elimination or not. If True, the algorithm will
        iteratively remove the least important feature until the score of the model
        decreases. This is done to reduce the number of features in the final model.
    
    Attributes
    ----------
    n_features_in_ : int
        The number of features passed to :meth:`fit`.

    n_features_out_ : int
        The number of features generated by :meth:`transform`.

    feature_names_in_ : list of str
        The names of the features passed to :meth:`fit` with the `columns` parameter.
        If `X` is a pandas DataFrame, the column names are used. Otherwise, the column
        names are a list of strings of the form `x0`, `x1`, ..., `x(n_features_in_-1)`.

    thresholds_ : list of tuples
        The thresholds found by the Gradient Boosting Classifier. Each tuple is of the form `(j, thresh)`
        where `j` is the index of the feature and `thresh` is the threshold found for that feature.

    Examples
    --------
    An example with the well known Iris dataset, which uses set_output to return a dataframe.
    Note that a regular lossless binarization would require 119 features.

    >>> from sklearn.datasets import load_iris
    >>> enc = ThresholdGuessBinarizer()
    >>> X, y = load_iris(as_frame=True, return_X_y=True)
    >>> enc.fit_transform(X, y)
    [[0. 1. 1. ... 1. 1. 1.]
    [1. 1. 1. ... 1. 1. 1.]
    [1. 1. 1. ... 1. 1. 1.]
    ...
    [0. 0. 0. ... 0. 0. 0.]
    [0. 0. 1. ... 0. 0. 0.]
    [0. 1. 1. ... 0. 1. 1.]]
    >>> enc.set_output(transform='pandas').fit_transform(X, y)
         sepal length (cm) <= 4.950000047683716  sepal length (cm) <= 5.950000047683716  sepal length (cm) <= 6.3500001430511475  ...  petal width (cm) <= 1.550000011920929  petal width (cm) <= 1.8499999642372131  sepal length (cm) <= 6.0
    0                                       0.0                                     1.0                                      1.0  ...                                    1.0                                     1.0                       1.0
    1                                       1.0                                     1.0                                      1.0  ...                                    1.0                                     1.0                       1.0
    2                                       1.0                                     1.0                                      1.0  ...                                    1.0                                     1.0                       1.0
    3                                       1.0                                     1.0                                      1.0  ...                                    1.0                                     1.0                       1.0
    4                                       0.0                                     1.0                                      1.0  ...                                    1.0                                     1.0                       1.0
    ..                                      ...                                     ...                                      ...  ...                                    ...                                     ...                       ...
    145                                     0.0                                     0.0                                      0.0  ...                                    0.0                                     0.0                       0.0
    146                                     0.0                                     0.0                                      1.0  ...                                    0.0                                     0.0                       0.0
    147                                     0.0                                     0.0                                      0.0  ...                                    0.0                                     0.0                       0.0
    148                                     0.0                                     0.0                                      1.0  ...                                    0.0                                     0.0                       0.0
    149                                     0.0                                     1.0                                      1.0  ...                                    0.0                                     1.0                       1.0

    [150 rows x 12 columns]
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        max_depth: int = 3,
        random_state: int = 0,
        column_elimination: bool = True

    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.column_elimination = column_elimination
        
    def fit(self, X, y, columns=None):
        """
        Fit ThresholdGuessBinarizer to X, y.

        If the user would like to produce a transformed dataset with legible 
        feature names, then `X` should be a pandas DataFrame with named 
        columns, or provde the `columns` parameter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to fit.
        
        y : array-like of shape (n_samples,)
            The target variable used to fit the internal Gradient Boosting Classifier.

        columns : array-like of shape [n_features] default=None
            The names of the features in X. If None, then the feature names are
            `x0`, `x1`, ..., `x(n_features_in_-1)`.

        Returns
        -------
        self : object 
            Fitted encoder.
        """
        # Store the column names. columns check has to be deone before check_array because it converts X to a numpy array
        self.feature_names_in_ = columns
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns

        # Input validation
        X, y = check_X_y(X, y, accept_sparse=False)
        _, m = X.shape

        # In the case that the user did not provide column names, we create them
        if self.feature_names_in_ is None:
            self.feature_names_in_ = [f'x{i}' for i in range(m)]

        # Store the number of features
        self.n_features_in_ = m

        # Extract the thresholds from the model
        clf = GradientBoostingClassifier(loss='log_loss',
                                                learning_rate=self.learning_rate,
                                                n_estimators=self.n_estimators,
                                                max_depth=self.max_depth,
                                                random_state=self.random_state)
        clf.fit(X, y)
        thresholds = []
        estimators = clf.estimators_
        for j in range(self.n_features_in_):
            thresholds_j = []
            for k in range(len(estimators)):
                thresholds_j.append(self.__threshold(estimators[k, 0], j))
            thresholds_j = np.unique(np.concatenate(thresholds_j))
            for thresh in thresholds_j:
                thresholds.append((j, thresh))

        # Perform Dataset thresholding based on found thresholds
        X_thr = np.zeros((X.shape[0], len(thresholds)))
        for i, (j, thresh) in enumerate(thresholds):
            X_thr[:, i] = X[:, j] <= thresh
        X_thr = X_thr.astype(float)

        # Refit the model on the thresholded data
        if self.column_elimination:
            self.thresholds_ = self.__column_elimination(X_thr, y, thresholds, clf)
        else:
            self.thresholds_ = thresholds

        self.n_features_out_ = len(self.thresholds_)

        # Return the transformer
        return self

    def get_feature_names_out(self, *args, **params):
        """
        Generates the names of the transformed features. This is necessary to 
        provide the `set_output` api that enables SciKit-Learn Transformers to
        return DataFrames (with named columns) instead of ndarrays.
        """
        check_is_fitted(self, ['n_features_in_', 'feature_names_in_', 'thresholds_', 'n_features_out_'])
        return [f'{self.feature_names_in_[j]} <= {thresh}' for j, thresh in self.thresholds_]

    

    def transform(self, X):
        """
        Transform X using the thresholds found by the Gradient Boosting Classifier.

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        Xt: ndarray of shape [n_samples, n_features_out_]
            Transformed input.
        """
        # Check is fit had ben called
        check_is_fitted(self, ['n_features_in_', 'feature_names_in_', 'thresholds_', 'n_features_out_'])
        
        # Input validation
        X = check_array(X, accept_sparse=False)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but ThresholdBinarizer is expecting {self.n_features_in_} features as input")

        return np.concatenate([np.atleast_2d(X[:, j] <= thresh).T for j, thresh in self.thresholds_], axis=1).astype(float)
    
    def __threshold(self, estimator, feature):
        """
        Returns the threshold of the estimator for the given feature if it exists.
        """
        f = estimator.tree_.feature
        t = estimator.tree_.threshold
        return t[f==feature]
    
    def __column_elimination(self, X, y, thresholds, clf):
        """
        Iteratively removes the least important feature until the score of the model
        decreases. Returns the remaining thresholds.
        """
        # Check is fit had ben called
        check_is_fitted(self, ['n_features_in_', 'feature_names_in_'])
        
        # Fit the model on the thresholded data
        clf.fit(X, y)
        base_score = clf.score(X, y)

        # Column elimination procedure
        curr_score = np.inf
        i = 0
        last_removed = None
        max_iter = len(thresholds) - 1
        while curr_score >= base_score and i < max_iter and clf.feature_importances_.size > 0:
            # Find the least important feature
            least_important = np.argmin(clf.feature_importances_)
            last_removed = (X[:, least_important], thresholds[least_important])
            # Remove the least important feature
            X = np.delete(X, least_important, axis=1)
            thresholds.remove(thresholds[least_important])
            # Re-fit the model
            clf.fit(X, y)
            # Update the current score
            curr_score = clf.score(X, y)
            i += 1
        
        # Add the last removed feature back
        if last_removed is not None:
            thresholds.append(last_removed[1])
        
        return thresholds
    
    def feature_map(self):
        """
        Extracts the mapping of original features to the transformed features. This can be passed to 
        the GOSDTClassifier to generate N-ary trees.
        
        Returns
        -------
        ret : dict
            A dictionary where the keys are the original feature indices and the values are the indices
            of the transformed features that correspond to the original feature.
        """
        
        # Check if fit had been called
        check_is_fitted(self, ['feature_names_in_', 'n_features_in_', 'thresholds_', 'n_features_out_'])
        
        # Create the feature map
        ret = {}
        for i, (j, _) in enumerate(self.thresholds_):
            if j not in ret:
                ret[j] = [i]
            else:
                ret[j].append(i)        
                
        return ret
        

