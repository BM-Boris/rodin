import numpy as np
import pandas as pd
from scipy import sparse
from typing import Union, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.multitest as smm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
import dash_bio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
from .mummichog.functional_analysis import *
from io import StringIO
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.request import urlopen 
sns.set_theme()

class Rodin_Class:
    """
    Rodin metabolomics analysis container.
    
    This class stores the main feature matrix together with sample metadata,
    feature metadata, dimensionality-reduction results, and other unstructured
    analysis outputs generated during a Rodin workflow.
    
    Attributes
    ----------
    X : pandas.DataFrame or None
        Main feature matrix with ``features`` in rows and ``samples`` in columns.
    samples : pandas.DataFrame or None
        Sample metadata table. The first column is treated as the sample identifier.
    features : pandas.DataFrame or None
        Feature metadata table aligned to the rows of ``X``.
    uns : dict
        Unstructured analysis outputs and metadata.
    dr : dict
        Stored dimensionality-reduction results such as PCA, UMAP, or t-SNE.
    """

    def __init__(
        self,
        X: Union[np.ndarray, sparse.spmatrix, pd.DataFrame, None] = None,
        samples: Union[pd.DataFrame, None] = None,
        features: Union[pd.DataFrame, None] = None,
        uns: Optional[dict] = None,
        dr: Optional[dict] = None
    ):

        # Initialize private backing fields
        """
        Initialize a ``Rodin_Class`` object.
        
        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.spmatrix or pandas.DataFrame or None, optional
            Main feature matrix with ``features`` in rows and ``samples`` in columns.
        samples : pandas.DataFrame or None, optional
            Sample metadata aligned to the columns of ``X``.
        features : pandas.DataFrame or None, optional
            Feature metadata aligned to the rows of ``X``.
        uns : dict or None, optional
            Unstructured annotations to attach to the object.
        dr : dict or None, optional
            Precomputed dimensionality-reduction results to attach to the object.
        
        Raises
        ------
        TypeError
            If ``X``, ``samples``, or ``features`` are provided in an unsupported container type.
        ValueError
            If sample identifiers or feature dimensions are not aligned across inputs.
        
        Notes
        -----
        Validation is performed immediately during initialization, including sample-ID
        matching and feature-dimension consistency.
        """
        self._X = None
        self._samples = None
        self._features = None
        #
        self.X = self._validate_dataframe(X, "X")
        self.samples = self._validate_dataframe(samples, "samples")
        self.features = self._validate_dataframe(features, "features")
        self.uns = uns if uns is not None else {}
        self.dr = dr if dr is not None else {}

        self._validate_sample_ids()
        self._validate_feature_dimensions()

    @property
    def X(self):
        """
        Return the main feature matrix.
        
        Returns
        -------
        pandas.DataFrame or None
            Matrix with ``features`` in rows and ``samples`` in columns.
        
        Notes
        -----
        ``X`` is expected to stay row-aligned with ``features`` and column-aligned with
        ``samples``.
        """
        return self._X

    @X.setter
    def X(self, value):
        """
        Set the main feature matrix and validate alignment.
        
        Parameters
        ----------
        value : pandas.DataFrame or None
            New feature matrix with ``features`` in rows and ``samples`` in columns.
        
        Raises
        ------
        TypeError
            If ``value`` is not a pandas DataFrame or ``None``.
        ValueError
            If the updated matrix is inconsistent with ``samples`` or ``features``.
        """
        self._X = self._validate_dataframe(value, "X")
        self._validate_sample_ids()
        self._validate_feature_dimensions()

    @property
    def samples(self):
        """
        Return the sample metadata table.
        
        Returns
        -------
        pandas.DataFrame or None
            Metadata table whose first column contains sample identifiers.
        
        Notes
        -----
        The sample order must match the column order of ``X``.
        """
        return self._samples

    @samples.setter
    def samples(self, value):
        """
        Set the sample metadata table and validate sample identifiers.
        
        Parameters
        ----------
        value : pandas.DataFrame or None
            New sample metadata table.
        
        Raises
        ------
        TypeError
            If ``value`` is not a pandas DataFrame or ``None``.
        ValueError
            If sample identifiers do not match the columns of ``X``.
        """
        self._samples = self._validate_dataframe(value, "samples")
        self._validate_sample_ids()

    @property
    def features(self):
        """
        Return the feature metadata table.
        
        Returns
        -------
        pandas.DataFrame or None
            Metadata table aligned to the rows of ``X``.
        
        Notes
        -----
        This table stores feature annotations together with downstream statistical
        results generated by Rodin methods.
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        Set the feature metadata table and validate dimensions.
        
        Parameters
        ----------
        value : pandas.DataFrame or None
            New feature metadata table.
        
        Raises
        ------
        TypeError
            If ``value`` is not a pandas DataFrame or ``None``.
        ValueError
            If the number of rows does not match the number of rows in ``X``.
        """
        self._features = self._validate_dataframe(value, "features")
        self._validate_feature_dimensions()

    def _validate_matrix_or_dataframe(self, obj, name):
        """
        Validate that an object is a supported matrix-like container.
        
        Parameters
        ----------
        obj : object
            Candidate object to validate.
        name : str
            Human-readable argument name used in error messages.
        
        Returns
        -------
        object
            The original object when validation succeeds.
        
        Raises
        ------
        TypeError
            If ``obj`` is not ``None``, a NumPy array, a SciPy sparse matrix, or a pandas DataFrame.
        """
        if obj is not None and not isinstance(obj, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
            raise TypeError(f"{name} should be a numpy array, a scipy sparse matrix, or a pandas DataFrame.")
        return obj

    def _validate_dataframe(self, dataframe, name):
        """
        Validate that an object is a pandas DataFrame.
        
        Parameters
        ----------
        dataframe : object
            Candidate object to validate.
        name : str
            Human-readable argument name used in error messages.
        
        Returns
        -------
        pandas.DataFrame or None
            The original object when validation succeeds.
        
        Raises
        ------
        TypeError
            If ``dataframe`` is not ``None`` and not a pandas DataFrame.
        """
        if dataframe is not None and not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"{name} should be a pandas DataFrame.")
        return dataframe

    def _validate_sample_ids(self):
        """
        Validate that sample identifiers match between ``X`` and ``samples``.
        
        Raises
        ------
        ValueError
            If the first column of ``samples`` does not contain exactly the same sample IDs,
            in the same order, as the columns of ``X``.
        """
        if self.X is not None and self.samples is not None:
            x_columns_set = set(self.X.columns)
            sample_ids_set = set(self.samples.iloc[:, 0])
            if x_columns_set != sample_ids_set:
                mismatched_x = x_columns_set - sample_ids_set
                mismatched_sample_ids = sample_ids_set - x_columns_set
                total_mismatches = len(mismatched_x) + len(mismatched_sample_ids)
            
                if total_mismatches <= 6:
                    error_message = "Column names in X do not match sample IDs in samples."
                    if mismatched_x:
                        error_message += f" X columns not found in samples: {mismatched_x}."
                    if mismatched_sample_ids:
                        error_message += f" Sample IDs not found in X: {mismatched_sample_ids}."
                else:
                    error_message = (
                        f"Column names in X do not match sample IDs in samples. "
                        f"Total mismatches: {total_mismatches}."
                    )
                raise ValueError(error_message)
            if list(self.X.columns) != list(self.samples.iloc[:, 0]):
                raise ValueError("The order of sample IDs must be the same in X and in samples")

                
                
    def _validate_feature_dimensions(self):
        """
        Validate that ``X`` and ``features`` have matching feature counts.
        
        Raises
        ------
        ValueError
            If the number of rows in ``features`` does not match the number of rows in ``X``.
        """
        if self._X is not None and self._features is not None:
            if self._X.shape[0] != self._features.shape[0]:
                raise ValueError(f"Number of columns in X ({self._X.shape[0]}) does not match number of rows ({self._features.shape[0]}) in features")



    def __repr__(self):
        """
        Return a compact text representation of the Rodin object.
        
        Returns
        -------
        str
            Summary string containing the object dimensions and stored analysis keys.
        """
        x_shape = self.X.shape if self.X is not None else (0, 0)
        num_features = x_shape[0]
        num_samples = x_shape[1]

        available_reductions = ", ".join(self.dr.keys()) if self.dr else ""
        if available_reductions!="":
            available_reductions = f"\ndr: {available_reductions}"

        available_uns = ", ".join(self.uns.keys()) if self.uns else ""
        if available_uns!="":
            available_uns = f"\nuns: {available_uns}"

    
        return f"< Rodin object > \ndim: {num_features} X {num_samples}" +available_reductions +available_uns


    def __getitem__(self, idx):
        """
        Slice a Rodin object using an aligned pandas selection.
        
        Parameters
        ----------
        idx : pandas.DataFrame
            DataFrame-based selection aligned to ``X``, ``features``, or ``samples``.
        
        Returns
        -------
        Rodin_Class
            New Rodin object containing the sliced matrices and metadata.
        
        Raises
        ------
        ValueError
            If the provided selection does not align with one of the supported Rodin tables.
        
        Notes
        -----
        This method is commonly used for feature-level filtering, for example by passing
        subsets derived from ``obj.features``.
        """
        
        if isinstance(idx, pd.DataFrame): 
            # Slicing by rows - assuming idx is a boolean mask from obj.X
                row_idx = idx.index[idx.any(axis=1)]
                col_idx = idx.columns

        if set(col_idx).issubset(self.X.columns):
        # Slice the X DataFrame
            X_sliced = self.X.loc[row_idx, col_idx]

            features_sliced = self.features.loc[row_idx]

            sample_ids = self.samples.iloc[:, 0]
            samples_sliced = self.samples[sample_ids.isin(col_idx)]

        # Creating a new Rodin object with sliced data
            sliced_obj = Rodin_Class(X=X_sliced, samples=samples_sliced, features=features_sliced,
                           uns=self.uns, dr=self.dr)

        elif set(col_idx).issubset(self.features.columns):

        # Slice the features DataFrame
            features_sliced = self.features.loc[row_idx, col_idx]
            X_sliced = self.X.loc[row_idx]

            sliced_obj = Rodin_Class(X=X_sliced, samples=self.samples, features=features_sliced,
                           uns=self.uns, dr=self.dr)

        elif set(col_idx).issubset(self.samples.columns):

        # Slice the samples DataFrame
            samples_sliced = self.samples.loc[row_idx, col_idx]
            sample_ids = samples_sliced.iloc[:, 0]
            X_sliced = self.X[sample_ids]
            sliced_obj = Rodin_Class(X=X_sliced, samples=samples_sliced, features=self.features,
                           uns=self.uns, dr=self.dr)


        else:
            raise ValueError("Invalid slice input")


        return sliced_obj

    def run_pca(self, n_components=2, custom_name='pca'):
        """
        Run principal component analysis on ``X`` and store the embedding in ``dr``.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of principal components to compute.
        custom_name : str, default='pca'
            Key used to store the result in ``self.dr``.
        
        Returns
        -------
        None
            The PCA scores are stored in ``self.dr[custom_name]``.
        
        Raises
        ------
        ValueError
            If ``X`` is missing or not a pandas DataFrame.
        """
        
        # Validate that X is not None and is a DataFrame
        if self.X is None or not isinstance(self.X, pd.DataFrame):
            raise ValueError("X matrix is not set or is not a DataFrame")

        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(self.X.T)

        # Store the results in dr attribute
        self.dr[custom_name] = principal_components


    def run_umap(self, n_components=2, use_pca=False, pca_name='pca', custom_name='umap', **umap_params):
        """
        Run UMAP on ``X`` or on a stored PCA embedding.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of UMAP dimensions to compute.
        use_pca : bool, default=False
            If ``True``, use a stored PCA embedding instead of ``X.T``.
        pca_name : str, default='pca'
            Key of the PCA embedding used when ``use_pca=True``.
        custom_name : str, default='umap'
            Key used to store the UMAP result in ``self.dr``.
        **umap_params : dict
            Additional keyword arguments forwarded to ``umap.UMAP``.
        
        Returns
        -------
        None
            The embedding is stored in ``self.dr[custom_name]``.
        
        Raises
        ------
        ValueError
            If ``use_pca=True`` and the requested PCA result is missing or has too few components.
        """
        
        if use_pca:
            if pca_name not in self.dr or self.dr[pca_name].shape[1] <= n_components:
                raise ValueError("Specified PCA result not found or insufficient components.")

            data = self.dr[pca_name]
        else:
            data = self.X.T

        # Perform UMAP
        umap_result = umap.UMAP(n_components=n_components, **umap_params).fit_transform(data)

        # Store the UMAP results
        self.dr[custom_name] = umap_result


    def run_tsne(self, n_components=2, use_pca=False, pca_name='pca', custom_name='t-sne', **tsne_params):
        """
        Run t-SNE on ``X`` or on a stored PCA embedding.
        
        Parameters
        ----------
        n_components : int, default=2
            Number of t-SNE dimensions to compute.
        use_pca : bool, default=False
            If ``True``, use a stored PCA embedding instead of ``X.T``.
        pca_name : str, default='pca'
            Key of the PCA embedding used when ``use_pca=True``.
        custom_name : str, default='t-sne'
            Key used to store the t-SNE result in ``self.dr``.
        **tsne_params : dict
            Additional keyword arguments forwarded to ``sklearn.manifold.TSNE``.
        
        Returns
        -------
        None
            The embedding is stored in ``self.dr[custom_name]``.
        
        Raises
        ------
        ValueError
            If ``use_pca=True`` and the requested PCA result is missing or has too few components.
        """
        
        if use_pca:
            if pca_name not in self.dr or self.dr[pca_name].shape[1] < n_components:
                raise ValueError("Specified PCA result not found or insufficient components.")

            data = self.dr[pca_name]
        else:
            data = self.X.T

        # Perform t-SNE
        tsne_result = TSNE(n_components=n_components, **tsne_params).fit_transform(data)

        # Store the t-SNE results
        self.dr[custom_name] = tsne_result


    def plot(self, hue=None, dr_name='umap', size=None, markers=None,title = "",interactive=True, **scatterplot_params):
        """
        Plot a stored dimensionality-reduction embedding.
        
        Parameters
        ----------
        hue : str or None, optional
            Column in ``self.samples`` used for point color.
        dr_name : str, default='umap'
            Key of the dimensionality-reduction result stored in ``self.dr``.
        size : str or None, optional
            Column in ``self.samples`` used for point size.
        markers : str or None, optional
            Column in ``self.samples`` used for point symbol or marker style.
        title : str, default=''
            Plot title.
        interactive : bool, default=True
            If ``True``, return a Plotly scatter plot. Otherwise return a seaborn/matplotlib plot.
        **scatterplot_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        object
            Plotly figure when ``interactive=True``, otherwise the matplotlib/seaborn axes object.
        
        Raises
        ------
        ValueError
            If ``dr_name`` is missing or if requested metadata columns are not present in ``samples``.
        """
        
        if dr_name not in self.dr:
            raise ValueError(f"Reduction '{dr_name}' not found in 'dr'.")

        if hue and hue not in self.samples:
            raise ValueError(f"Column '{hue}' not found in 'samples'.")
        if size and size not in self.samples:
            raise ValueError(f"Column '{size}' not found in 'samples'.")
        if markers and markers not in self.samples:
            raise ValueError(f"Column '{markers}' not found in 'samples'.")

        # Retrieve the reduction data
        dr_data = self.dr[dr_name]

        # Check if reduction data is 2D
        if dr_data.shape[1] != 2:
            raise ValueError("Reduction data must be 2-dimensional for plotting.")

        if interactive:
            # Prepare the plot
            scatter_args = {'x': dr_data[:, 0], 'y': dr_data[:, 1], 'width':900,'height':550,
                            'hover_name': self.samples.iloc[:,0], **scatterplot_params}
            
            if hue: scatter_args['color'] = self.samples[hue]
            if size: scatter_args['size'] = self.samples[size]
            if markers: scatter_args['symbol'] = self.samples[markers]
            
        
            # Prepare the plot
            ax=px.scatter(**scatter_args,labels={
                     "x": f'{dr_name}_1',
                     "y": f'{dr_name}_2'})
            ax.update_layout(
                title={
                    'text': f"{title}",
                    'x': 0.45,
                    'xanchor': 'center'
            })

        else:
            # Prepare the plot
            scatter_args = {'x': dr_data[:, 0], 'y': dr_data[:, 1], **scatterplot_params}
            if hue: scatter_args['hue'] = self.samples[hue]
            if size: scatter_args['size'] = self.samples[size]
            if markers: scatter_args['style'] = self.samples[markers]
        
            # Prepare the plot
            ax=sns.scatterplot(**scatter_args)
        
            plt.title(title)
            plt.xlabel(f'{dr_name}_1')
            plt.ylabel(f'{dr_name}_2')
        
            try:
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            except:
                pass
        
        return ax

    def transform(
            self,
            thresh: float = 0.5,
            norm: str | None = "q",
            scale: str | bool | None = "uv",
            log: bool = True
        ):
        """
        Preprocess the feature matrix stored in ``self.X``.
        
        Parameters
        ----------
        thresh : float, default=0.5
            Maximum allowed fraction of zeros per feature before the feature is removed.
        norm : {'q', 't', 'median', 'mean', None}, default='q'
            Normalization method applied across ``samples``.
        scale : {'uv', 'pareto', 'minmax', None} or bool, default='uv'
            Scaling method applied after normalization. Boolean values keep backward-compatible
            behavior where ``True`` maps to ``'uv'`` and ``False`` maps to ``None``.
        log : bool, default=True
            If ``True``, apply ``log2(x + 1)`` after normalization and scaling.
        
        Returns
        -------
        Rodin_Class
            The same object with updated ``X`` and aligned ``features``.
        
        Raises
        ------
        ValueError
            If ``self.X`` is missing or if an unsupported normalization or scaling option is requested.
        
        Notes
        -----
        The preprocessing workflow fills missing values with zero, filters ``features`` by missingness,
        applies optional normalization and scaling, performs optional log transformation, and then
        removes constant or duplicated ``features``.
        """

    
        # ------------------------------------------------------------------ #
        # Sanity checks                                                      #
        # ------------------------------------------------------------------ #
        if self.X is None:
            raise ValueError("self.X is empty – assign a DataFrame before calling transform().")
    
        # ------------------------------------------------------------------ #
        # 1. Fill NAs and filter by missingness                              #
        # ------------------------------------------------------------------ #
        df = self.X.fillna(0.0)
        row_mask = (df == 0).mean(axis=1) <= thresh
        col_mask = (df == 0).mean(axis=0) <= 1.0          
        df = df.loc[row_mask, col_mask]
    
        # ------------------------------------------------------------------ #
        # 2. Normalisation                                                   #
        # ------------------------------------------------------------------ #
        norm = None if norm is None else norm.lower()
        if norm == 'q':                                     # Quantile
            sorted_vals = np.sort(df, axis=0)
            mean_rank   = sorted_vals.mean(axis=1)
            ranks       = df.rank(method="min").astype(int) - 1
            df = pd.DataFrame(mean_rank[ranks.values], index=df.index, columns=df.columns)
    
        elif norm == 't':                                   # Total intensity
            df = df.div(df.sum(axis=0), axis=1) * 1e5
    
        elif norm == 'median':                              # Median scaling
            med_per_sample = df.median(axis=0)
            grand_median   = med_per_sample.median()
            df = df.div(med_per_sample, axis=1) * grand_median
    
        elif norm == 'mean':                                # Mean scaling
            mean_per_sample = df.mean(axis=0)
            grand_mean      = mean_per_sample.mean()
            df = df.div(mean_per_sample, axis=1) * grand_mean
    
        elif norm is not None:
            raise ValueError(f'Unknown norm="{norm}". Choose q, t, median, mean or None.')
    
        # ------------------------------------------------------------------ #
        # 3. Scaling                                                         #
        # ------------------------------------------------------------------ #
        if isinstance(scale, bool):
            scale = 'uv' if scale else None
        scale = None if scale is None else str(scale).lower()
    
        if scale == 'uv':                                    # Unit variance
            stds = df.std(axis=1).replace(0, 1)
            df = df.div(stds, axis=0)
    
        elif scale == 'pareto':                              # Pareto scaling
            stds = np.sqrt(df.std(axis=1)).replace(0, 1)
            df = df.div(stds, axis=0)
    
        elif scale == 'minmax':                              # 0–1 scaling
            mins   = df.min(axis=1)
            ranges = (df.max(axis=1) - mins).replace(0, 1)
            df = df.sub(mins, axis=0).div(ranges, axis=0)
    
        elif scale is not None:
            raise ValueError(f'Unknown scale="{scale}". Choose uv, pareto, minmax or None.')
    
        # ------------------------------------------------------------------ #
        # 4. Log transformation                                              #
        # ------------------------------------------------------------------ #
        if log:
            df = np.log2(df + 1)
    
        # ------------------------------------------------------------------ #
        # 5. Remove constant and duplicate features                          #
        # ------------------------------------------------------------------ #
        df = df.loc[df.nunique(axis=1) > 1]
        df = df.drop_duplicates(ignore_index=False)
    
        filtered = self.X.shape[0] - df.shape[0]
        print(f"Number of features filtered: {filtered}")
    
        # ------------------------------------------------------------------ #
        # 6. Update object state                                             #
        # ------------------------------------------------------------------ #
        self._X        = df
        self._features = self.features.loc[df.index]        # keep metadata aligned
        self.X         = self._X                            # triggers validators
        self.features  = self._features
    
        return self


    def oneway_anova(self, column_name):
        """
        Run one-way ANOVA for each feature.
        
        Parameters
        ----------
        column_name : str
            Column in ``self.samples`` defining the groups to compare.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with ANOVA p-values and adjusted p-values.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, or if ``column_name`` is not present in ``samples``.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling oneway_anova.")

        if column_name not in self.samples.columns:
            raise ValueError(f"Column '{column_name}' not found in samples.")
        

        # Extract unique classes and prepare the data
        unique_classes = self.samples[column_name].unique()
        data_groups = [self.X[self.samples[self.samples[column_name] == i].iloc[:,0]].values for i in unique_classes]
        # Perform ANOVA
        F, p = stats.f_oneway(*data_groups,axis=1)
        # Adjust the p-values for multiple testing
        p_adj = stats.false_discovery_control(ps=p, method='bh')
        
        # Add results to the features dataframe
        self.features[f'p_value(owa) {column_name}'] = p
        self.features[f'p_adj(owa) {column_name}'] = p_adj

        return self.features


    def ttest(self, column_name):
        """
        Run an unpaired two-group t-test for each feature.
        
        Parameters
        ----------
        column_name : str
            Column in ``self.samples`` defining the two groups.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with t-test p-values and adjusted p-values.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, if ``column_name`` is absent, or if the column does not contain exactly two groups.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling ttest.")

        if column_name not in self.samples.columns:
            raise ValueError(f"Column '{column_name}' not found in samples.")
            
        unique_classes = self.samples[column_name].unique()
        
        if len(unique_classes) != 2:
            raise ValueError("The t-test function expects exactly two unique classes.")
        
        # Extract data for each class
        data_class1 = self.X[self.samples[self.samples[column_name] == unique_classes[0]].iloc[:, 0]].values
        data_class2 = self.X[self.samples[self.samples[column_name] == unique_classes[1]].iloc[:, 0]].values
        # Perform ANOVA
        t_stat, p = stats.ttest_ind(data_class1, data_class2, axis=1, equal_var=False)

        # Adjust the p-values for multiple testing
        p_adj = stats.false_discovery_control(ps=p, method='bh')
        
        # Add results to the features dataframe
        self.features[f'p_value(tt) {column_name}'] = p
        self.features[f'p_adj(tt) {column_name}'] = p_adj

        return self.features

    def ttest_paired(self, column_names):
        """
        Run a paired t-test for each feature.
        
        Parameters
        ----------
        column_names : list[str]
            Two-element list where the first value is the condition column and the second value is
            the subject or pairing column in ``self.samples``.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with paired t-test p-values and adjusted p-values.
        
        Raises
        ------
        ValueError
            If required inputs are missing, if the condition column does not contain exactly two levels,
            if duplicate subject-condition rows exist, or if no complete subject pairs are available.
        
        Notes
        -----
        Only subjects with observations in both levels of the condition column are used in the test.
        """

        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling ttest_paired.")

        if len(column_names) != 2:
            raise ValueError("Please provide exactly two column names: [condition_col, subject_col].")

        for col in column_names:
            if col not in self.samples.columns:
                raise ValueError(f"Column '{col}' not found in samples.")

        cond_col = column_names[0]
        subject_col = column_names[1]
        sid_col = self.samples.columns[0]

        meta = self.samples[[sid_col, cond_col, subject_col]].copy()
        meta = meta.dropna(subset=[sid_col, cond_col, subject_col])

        uniq_levels = pd.unique(meta[cond_col])
        if len(uniq_levels) != 2:
            raise ValueError(
                f"ttest_paired expects exactly 2 unique levels in '{cond_col}', got {len(uniq_levels)}."
            )

        if meta.duplicated(subset=[subject_col, cond_col]).any():
            bad = meta[meta.duplicated(subset=[subject_col, cond_col], keep=False)].sort_values([subject_col, cond_col])
            raise ValueError(
                "Found duplicate (subject, condition) rows. Each subject must have at most 1 sample per level.\n"
                f"Example rows:\n{bad.head(10)}"
            )

        try:
            levels = sorted(list(uniq_levels))
        except Exception:
            levels = list(uniq_levels)

        lv0, lv1 = levels[0], levels[1]

        wide = meta.pivot(index=subject_col, columns=cond_col, values=sid_col)
        pairs = wide[[lv0, lv1]].dropna()

        if pairs.shape[0] == 0:
            raise ValueError(
                "No complete pairs found across the two levels. "
                "Every subject must have one sample in each of the two levels."
            )

        s0 = pairs[lv0].astype(str).tolist()
        s1 = pairs[lv1].astype(str).tolist()

        x_cols = set(map(str, self.X.columns))
        missing = set(s0 + s1) - x_cols
        if missing:
            raise ValueError(f"Some paired sample IDs are missing in X columns: {list(sorted(missing))[:10]}")

        y0 = self.X.loc[:, s0].to_numpy()
        y1 = self.X.loc[:, s1].to_numpy()

        _, pvals = stats.ttest_rel(y1, y0, axis=1, nan_policy="omit")
        p_adj = stats.false_discovery_control(ps=np.nan_to_num(pvals, nan=1.0), method="bh")

        label = f"{cond_col} [{subject_col}]"
        self.features[f"p_value(ptt) {label}"] = pvals
        self.features[f"p_adj(ptt) {label}"] = p_adj

        return self.features


    def pls_da(self, column_name):
        """
        Run PLS-DA and store variable importance scores.
        
        Parameters
        ----------
        column_name : str
            Column in ``self.samples`` used as the response variable.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with a ``vips`` column.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, or if ``column_name`` is not present in ``samples``.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling pls_da.")

        if column_name not in self.samples.columns:
            raise ValueError(f"Column '{column_name}' not found in samples.")

        # Extract data matrix and class labels
        X = self.X.T
        y = self.samples[column_name].values
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Perform PLS-DA
        plsr = PLSRegression(n_components=2)
        plsr.fit(X, y_encoded)

        # Function to calculate VIP scores
        def vip(x, y, model):
            """
            Compute variable importance in projection scores for a fitted PLS model.
            
            Parameters
            ----------
            x : numpy.ndarray
                Predictor matrix used to fit the model.
            y : numpy.ndarray
                Response vector or matrix used to fit the model.
            model : object
                Fitted PLS model exposing ``x_scores_``, ``x_rotations_``, and ``y_loadings_``.
            
            Returns
            -------
            numpy.ndarray
                VIP score for each predictor.
            """
            t = model.x_scores_
            w = model.x_rotations_
            q = model.y_loadings_

            m, p = x.shape
            _, h = t.shape

            vips = np.zeros((p,))

            s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
            total_s = np.sum(s)

            for i in range(p):
                weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
                vips[i] = np.sqrt(p * (s.T @ weight) / total_s)

            return vips

        # Calculate and add VIP scores to the features dataframe
        self.features['vips'] = vip(X, y_encoded, plsr)

        return self.features

    def twoway_anova(self, column_names):
        """
        Run two-way ANOVA for each feature.
        
        Parameters
        ----------
        column_names : list[str]
            Two sample-metadata columns used as factors in the model.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with two-way ANOVA results.
        
        Raises
        ------
        ValueError
            If inputs are missing, if any requested column is absent, or if exactly two factor names are not provided.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling twoway_anova.")
    
        if len(column_names) != 2:
            raise ValueError("Please provide exactly two column names for two-way ANOVA.")
        
        for col in column_names:
            if col not in self.samples.columns:
                raise ValueError(f"Column '{col}' not found in samples.")
    
        class_labels_1 = self.samples[column_names[0]].values
        class_labels_2 = self.samples[column_names[1]].values
    
        # Placeholder list for storing p-values
        p_vals_list = []
        for j in tqdm(range(self.X.shape[0]), desc="Processing ANOVA"):
            values = self.X.iloc[j, :].values
            df_new = pd.DataFrame({
                'Intensity': values,
                column_names[0]: class_labels_1,
                column_names[1]: class_labels_2
            })
    
            # Define the formula for two-way ANOVA with interaction
            formula = f'Intensity ~ C({column_names[0]}) * C({column_names[1]})'
            model = smf.ols(formula, data=df_new).fit()
            anova_table = sm.stats.anova_lm(model, typ=1)
    
            current_p_vals = anova_table['PR(>F)'].values[:-1]  # Exclude residual row
            p_vals_list.append(current_p_vals)
    
        p_vals = np.array(p_vals_list)
        anova_classes = anova_table.index.values[:-1]  # Exclude residual row
    
        _anova_classes = []
        for name in anova_classes:
            _anova_classes.append(name.replace('C(', '').replace(')', ''))
    
        for i in range(len(_anova_classes)):
            p_value_col = f"p_value(twa) {_anova_classes[i]}"
            p_adj_col = f"p_adj(twa) {_anova_classes[i]}"
            self.features[p_value_col] = p_vals[:, i]
            self.features[p_adj_col] = stats.false_discovery_control(ps=p_vals[:, i], method='bh')
    
        return self.features

    def sf_lr(self, target_column, moderator=None, interaction=False,degree=1, **kwargs):
        """
        Run feature-wise linear regression.
        
        Parameters
        ----------
        target_column : str
            Continuous response column in ``self.samples``.
        moderator : str or None, optional
            Optional moderator column included in the model.
        interaction : bool, default=False
            If ``True``, include an interaction term between the feature and moderator.
        degree : int, default=1
            Polynomial degree applied to the feature term.
        **kwargs : dict
            Additional keyword arguments forwarded to the regression backend.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with regression coefficients, p-values, or related statistics.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, if ``target_column`` is absent, or if the moderator column is invalid.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling single_feature_lr.")
    
        if target_column not in self.samples.columns:
            raise ValueError(f"Column '{target_column}' not found in samples.")
    
        if moderator and moderator not in self.samples.columns:
            raise ValueError(f"Moderator column '{moderator}' not found in samples.")

        if moderator and self.samples[moderator].dtype == 'O':
            raise ValueError(f"Moderator column '{moderator}' must be numerical.")

        
        df = self.X.T.copy()
        n_cols = df.shape[1]
        features_list = []  # Renamed from 'list' to 'features_list' to avoid shadowing built-in names
        dependent_var = self.samples[target_column].copy()
        dependent_var.index = self.samples.iloc[:,0]
        if moderator:
            df['moderator_var'] = self.samples[moderator].values
            features_list = ['moderator_var']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if interaction:
                # For each feature, calculate interaction term with moderator
                    for column in df.columns[:-1]:  # Exclude the last column which is 'moderator_var'
                        df[f'{column}_interaction'] = df[column] * df['moderator_var']
                        features_list.append(f'{column}_interaction')
                        p_int = []
    
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        for column in tqdm(df.columns[:n_cols]):
    
            cols_to_use = [column] + features_list if not interaction else [column, 'moderator_var', f'{column}_interaction']
    
            if degree>1:
                poly_features = poly.fit_transform(df[[column]])
                poly_feature_names = [f'{column}'] + [f'{column}^{i}' for i in range(2, degree + 1)]
                poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index).iloc[:,1:]
                #print(poly_df)
    
                independent_vars = sm.add_constant(df[cols_to_use].join(poly_df))
            else:
                independent_vars = sm.add_constant(df[cols_to_use])
            
            model = sm.OLS(dependent_var, independent_vars).fit(**kwargs)
            # Store the p-value of the feature
    
            if column==0:
                pvals=model.pvalues.values
            else:
                pvals = np.vstack((pvals,model.pvalues.values))
        pvals=np.transpose(pvals)
                
                
                
        
        # Update self.features with p-values and adjusted p-value
        self.features[f'p_value(lr) {target_column}'] = pvals[1]
        self.features[f'p_adj(lr) {target_column}'] = stats.false_discovery_control(ps=pvals[1], method='bh')
        if interaction:
            self.features[f'p_value(lr) {target_column}*{moderator}'] = pvals[3]
            self.features[f'p_adj(lr) {target_column}*{moderator}'] = stats.false_discovery_control(ps=pvals[3], method='bh')
        if degree>1:
            for i in range(2, degree + 1):
                self.features[f'p_value(lr) {target_column}^{i}'] = pvals[-degree+i-1]
                self.features[f'p_adj(lr) {target_column}^{i}'] = stats.false_discovery_control(ps=pvals[-degree+i-1], method='bh')        
    
        return self.features


    def sf_lg(self, target_column, moderator=None, interaction=False, regu=False,degree=1, **kwargs):
        """
        Run feature-wise logistic regression.
        
        Parameters
        ----------
        target_column : str
            Categorical response column in ``self.samples``.
        moderator : str or None, optional
            Optional moderator column included in the model.
        interaction : bool, default=False
            If ``True``, include an interaction term between the feature and moderator.
        regu : bool, default=False
            If ``True``, enable regularized logistic regression.
        degree : int, default=1
            Polynomial degree applied to the feature term.
        **kwargs : dict
            Additional keyword arguments forwarded to the regression backend.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with logistic-regression statistics.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, if ``target_column`` is absent, or if the moderator column is invalid.
        """
        
        if self.X is None or self.samples is None:
                raise ValueError("Both X and samples must be assigned before calling single_feature_lr.")
        
        if target_column not in self.samples.columns:
            raise ValueError(f"Column '{target_column}' not found in samples.")
    
        if moderator and moderator not in self.samples.columns:
            raise ValueError(f"Moderator column '{moderator}' not found in samples.")
            
        if moderator and self.samples[moderator].dtype == 'O':
            raise ValueError(f"Moderator column '{moderator}' must be numerical.")

        
    
        
        df = self.X.T.copy()
        n_cols = df.shape[1]
        p_values = []
        features_list = []  # Renamed from 'list' to 'features_list' to avoid shadowing built-in names
        dependent_var = self.samples[target_column].copy()
        if dependent_var.dtype == 'O':
            dependent_var = pd.Series(LabelEncoder().fit_transform(dependent_var))
        dependent_var.index = self.samples.iloc[:,0]
    
        if moderator:
            df['moderator_var'] = self.samples[moderator].values
            features_list = ['moderator_var']
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if interaction:
                # For each feature, calculate interaction term with moderator
                    for column in df.columns[:-1]:  # Exclude the last column which is 'moderator_var'
                        df[f'{column}_interaction'] = df[column] * df['moderator_var']
                        features_list.append(f'{column}_interaction')
                        p_int = []
                        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        for column in tqdm(df.columns[:n_cols]):
            cols_to_use = [column] + features_list if not interaction else [column, 'moderator_var', f'{column}_interaction']
            if degree>1:
                poly_features = poly.fit_transform(df[[column]])
                poly_feature_names = [f'{column}'] + [f'{column}^{i}' for i in range(2, degree + 1)]
                poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index).iloc[:,1:]

                independent_vars = sm.add_constant(df[cols_to_use].join(poly_df))
            else:
                independent_vars = sm.add_constant(df[cols_to_use])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if regu:
                        model = sm.Logit(dependent_var, independent_vars).fit_regularized(disp=0,**kwargs)
                    else:    
                        model = sm.Logit(dependent_var, independent_vars).fit(disp=0,**kwargs)
                    # Store the p-value of the feature
                    if column==0:
                        pvals=model.pvalues.values
                        n=len(pvals)
                    else:
                        pvals = np.vstack((pvals,model.pvalues.values))
            except Exception as e:
                print(f"Error processing column {column}: {e}")
                pvals = np.vstack((pvals,np.full(n, np.nan)))
                
        pvals=np.transpose(pvals)
        # Update self.features with p-values and adjusted p-value
        self.features[f'p_value(lg) {target_column}'] = pvals[1]
        self.features[f'p_adj(lg) {target_column}'] = stats.false_discovery_control(ps=np.nan_to_num(pvals[1],nan=1.0), method='bh')
        if interaction:
            self.features[f'p_value(lg) {target_column}*{moderator}'] = pvals[3]
            self.features[f'p_adj(lg) {target_column}*{moderator}'] = stats.false_discovery_control(ps=np.nan_to_num(pvals[3],nan=1.0), method='bh')
        if degree>1:
            for i in range(2, degree + 1):
                self.features[f'p_value(lg) {target_column}^{i}'] = pvals[-degree+i-1]
                self.features[f'p_adj(lg) {target_column}^{i}'] = stats.false_discovery_control(ps=np.nan_to_num(pvals[-degree+i-1],nan=1.0), method='bh')        
    
        return self.features

                    

    def rf_class(self, target_column, n_estimators=100, random_state=16,cv=0, **kwargs):
        """
        Train a random-forest classifier on sample metadata labels.
        
        Parameters
        ----------
        target_column : str
            Column in ``self.samples`` used as the classification target.
        n_estimators : int, default=100
            Number of trees in the forest.
        random_state : int, default=16
            Random seed for reproducibility.
        cv : int, default=0
            Number of cross-validation folds. Use ``0`` to skip cross-validation.
        **kwargs : dict
            Additional keyword arguments forwarded to ``RandomForestClassifier``.
        
        Returns
        -------
        dict
            Dictionary containing feature-importance values and classification metrics.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, or if ``target_column`` is not present in ``samples``.
        """
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling rf_class.")
    
        if target_column not in self.samples.columns:
            raise ValueError(f"Column '{target_column}' not found in samples.")
    
        X = self.X.T
        y = self.samples[target_column]
    
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)

        if cv>0:
            # Perform cross-validated predictions
            y_pred = cross_val_predict(clf, X, y, cv=cv)
        
            # Generate classification report
            report = classification_report(y, y_pred, output_dict=True)
            print(pd.DataFrame(report).transpose())
    
        # Train a final model to get feature importances
        clf.fit(X, y)
        feature_importances = clf.feature_importances_
        self.features[f'imp(rf) {target_column}'] = feature_importances  # Update features DataFrame with importances
    
        return self.features
        

    def rf_regress(self, target_column, n_estimators=100, random_state=16, cv=0, **kwargs):
        """
        Train a random-forest regressor on sample metadata values.
        
        Parameters
        ----------
        target_column : str
            Column in ``self.samples`` used as the regression target.
        n_estimators : int, default=100
            Number of trees in the forest.
        random_state : int, default=16
            Random seed for reproducibility.
        cv : int, default=0
            Number of cross-validation folds. Use ``0`` to skip cross-validation.
        **kwargs : dict
            Additional keyword arguments forwarded to ``RandomForestRegressor``.
        
        Returns
        -------
        dict
            Dictionary containing feature-importance values and regression metrics.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, or if ``target_column`` is not present in ``samples``.
        """
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling rf_regress.")
    
        if target_column not in self.samples.columns:
            raise ValueError(f"Column '{target_column}' not found in samples.")
    
        X = self.X.T
        y = self.samples[target_column]
    
        clf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs)

        if cv>0:
            y_pred = cross_val_predict(clf, X, y, cv=cv)
        
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
        
            print("MSE: ",mse,"\nMAE: ",mae,"\nR2: ",r2)
    
        # Train a final model to get feature importances
        clf.fit(X, y)
        feature_importances = clf.feature_importances_
        self.features[f'imp(rf) {target_column}'] = feature_importances  # Update features DataFrame with importances
    
        return self.features
        
    
    def fold_change(self, column_name,reference=None):
        """
        Calculate log fold change for each feature between sample groups.
        
        Parameters
        ----------
        column_name : str
            Column in ``self.samples`` defining the groups to compare.
        reference : str or None, optional
            Reference level used as the denominator group.
        
        Returns
        -------
        pandas.DataFrame
            Updated ``self.features`` with fold-change columns.
        
        Raises
        ------
        ValueError
            If ``X`` or ``samples`` is missing, or if ``column_name`` is not present in ``samples``.
        
        Notes
        -----
        This method is typically used after log transformation so that differences correspond
        to log fold change.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling oneway_anova.")

        if column_name not in self.samples.columns:
            raise ValueError(f"Column '{column_name}' not found in samples.")
        

        # Extract unique classes and prepare the data
        unique_classes = self.samples[column_name].unique()
        if reference!=None:
            unique_classes=np.concatenate(([reference], unique_classes[unique_classes != reference])) if reference in unique_classes else unique_classes
          
        data_groups = [self.X[self.samples[self.samples[column_name] == i].iloc[:,0]].values for i in unique_classes]
        
        mean_group_0 = np.mean(data_groups[0], axis=1)
        avg_fc = []

        for idx, group in enumerate(data_groups[1:]):
            mean_current_group = np.mean(group, axis=1)
            log_fold_change = mean_current_group - mean_group_0
    
            self.features[f'lfc ({unique_classes[idx+1]} vs {unique_classes[0]})'] = log_fold_change
    
            if(len(unique_classes) > 2):
                avg_fc.append(log_fold_change)
                
        if(len(unique_classes) > 2):
            self.features[f'lfc (others vs {unique_classes[0]})'] = np.mean(avg_fc, axis=0)
            
        return self.features

    def clustergram(self, title="", interactive=True,
                    height=820,width=900,center_values=False,standardize='row',
                    link_method='ward',hidden_labels='row',color_map='RdBu',line_width=1.4,hue=None, **clustergram_params):
        """
        Generate a clustered heatmap from ``self.X``.
        
        Parameters
        ----------
        title : str, default=''
            Plot title.
        interactive : bool, default=True
            If ``True``, return an interactive Dash Bio clustergram. Otherwise return a seaborn clustermap.
        height : int or None, default=820
            Figure height in pixels for interactive output.
        width : int or None, default=900
            Figure width in pixels for interactive output.
        center_values : bool, default=False
            If ``True``, center values before plotting when supported by the backend.
        standardize : {'row', 'column', None}, default='row'
            Standardization axis used before clustering.
        link_method : str, default='ward'
            Linkage method used for hierarchical clustering.
        hidden_labels : str or None, default='row'
            Axis labels to hide in the clustergram.
        color_map : str, default='RdBu'
            Colormap used for the heatmap.
        line_width : float, default=1.4
            Width of grid lines between heatmap cells.
        hue : str or None, optional
            Column in ``self.samples`` used to annotate ``samples``.
        **clustergram_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        object
            Interactive clustergram figure or seaborn cluster object.
        
        Raises
        ------
        ValueError
            If ``X`` is missing.
        """
        
        if self.X is None:
            raise ValueError("X attribute must be assigned before plotting a clustergram.")

        data = self.X.astype(float)
        title_part = title
        if interactive:
            # Use Dash Bio for interactive visualization
            fig = dash_bio.Clustergram(
                data=data,
                column_labels=list(data.columns.values),
                row_labels=list(data.index),
                height=height,
                width=width,
                center_values=center_values,
                standardize=standardize,
                link_method=link_method,
                hidden_labels=hidden_labels,
                color_map=color_map,
                line_width=line_width,
                **clustergram_params
            )

            
            title_part = title
            fig.update_layout(
                title={
                    'text': f"{title_part}",
                    'x': 0.45,
                    'xanchor': 'center'
                })
            if hue:
                ticks = fig.layout.xaxis11['ticktext']
                cats = self.samples[hue].unique()
                colors=px.colors.qualitative.Plotly
                keys = dict(zip(cats, colors))
                ticks = [f"<span style='color:{str(keys[self.samples[self.samples.iloc[:,0]==i][hue].values[0]])}'> {str(i)} </span>" for i in ticks]

                fig.layout.xaxis11['ticktext']=ticks

                s=""
                for i in keys:
                    s+=f"<span style='color:{keys[i]}'><b>• {i}<br></b></span>"

                fig.update_layout(annotations=[dict(
                        text=s,
                        showarrow=False,
                        xref='x11',
                        xanchor="right",
                        x=0,
                        opacity=0.8,)])
        
                
        else:
            standard_scale = 0 if standardize=='row' else 1
            figsize=(height/100,width/100)
            # Use Seaborn for static visualization
            fig = sns.clustermap(data, figsize=figsize,method=link_method, cmap=color_map,
                                 standard_scale=standard_scale,**clustergram_params)
            fig.fig.suptitle(title_part) 

        return fig

    def boxplot(
    self,
    hue,
    pathways=None,
    eids=None,
    rows=None,
    significant=0.05,
    grid_dim=None,
    figsize=None,
    title="",
    zeros=True,
    cutoff_path=0.05,
    interactive=True,
    category_order=None,
    hue2=None,
    hue2_order=None,
    hue2_palette=None,
    **boxplot_params):
        """
        Generate box plots for selected ``features`` or pathway-derived feature sets.
        
        Parameters
        ----------
        hue : str
            Column in ``self.samples`` used for x-axis grouping.
        pathways : str or list[str] or None, optional
            Pathway name or names used to resolve ``features`` through ``show_compounds``.
        eids : sequence or None, optional
            Empirical compound identifiers used to resolve feature rows.
        rows : sequence of int or None, optional
            Explicit feature row indices to plot.
        significant : float, default=0.05
            P-value threshold applied when rows are resolved from ``pathways`` or ``eids``.
        grid_dim : tuple[int, int] or None, optional
            Explicit subplot grid dimensions.
        figsize : tuple[float, float] or None, optional
            Figure size for static output.
        title : str, default=''
            Overall plot title.
        zeros : bool, default=True
            If ``False``, replace zero values with missing values before plotting.
        cutoff_path : float, default=0.05
            Pathway p-value threshold used when ``pathways`` is provided.
        interactive : bool, default=True
            If ``True``, generate Plotly output. Otherwise generate matplotlib/seaborn output.
        category_order : list or None, optional
            Explicit order for hue categories.
        hue2 : str or None, optional
            Optional second grouping column for split box plots.
        hue2_order : list or None, optional
            Explicit order for ``hue2`` categories.
        hue2_palette : dict or list or None, optional
            Palette used for ``hue2`` groups.
        **boxplot_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        list or object
            Generated plot objects for the requested rows or pathways.
        
        Notes
        -----
        Common ``**boxplot_params`` options include Plotly-style arguments such as
        ``boxpoints='outliers'``, ``jitter=0.3``, ``pointpos=-1.4``,
        ``marker={'size': 5, 'opacity': 0.65}``, and ``line={'width': 1}``.
        
        In static seaborn mode, useful keyword arguments include ``width=0.55``,
        ``showfliers=False``, ``linewidth=1``, ``dodge=True``, and ``palette='deep'``.
        """

        figs = []
        if figsize is not None:
            tmp = 0

        # Handle the case where pathways, eids, and rows are all None
        if pathways is None and eids is None and rows is None:
            compounds_df = self.show_compounds(cutoff_path=cutoff_path, cutoff_eids=significant)
            pathways = compounds_df.index.get_level_values("pathway").unique().tolist()

        # If pathways is not None, handle it
        if pathways is not None:
            if isinstance(pathways, str):
                pathways = [pathways]

            compounds_df = self.show_compounds(paths=pathways, cutoff_path=cutoff_path, cutoff_eids=significant)
            max_input_rows = compounds_df.groupby(level="pathway")["input_row"].nunique().max()

            for pathway in pathways:
                pathway_compounds = compounds_df.xs(pathway, level="pathway")
                eids = pathway_compounds.index.get_level_values("compound_id").unique().tolist()

                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns["compounds"][
                    (self.uns["compounds"]["EID"].isin(eids)) &
                    (self.uns["compounds"]["p_value"] <= significant)
                ].input_row.values

                # If no rows are found, skip this pathway
                if len(rows) == 0:
                    print(f"No significant compounds found for pathway: {pathway}")
                    continue

                # Ensure rows is a list
                if isinstance(rows, int):
                    rows = [rows]

                rows = np.unique(rows)

                # Determine the number of subplots and set default grid dimensions if not provided
                n_plots = len(rows)
                if grid_dim is None:
                    grid_dim = (1, max_input_rows)

                nrows, ncols = grid_dim
                if figsize is None:
                    figsize = (5 * ncols, 5 * nrows)
                else:
                    height = figsize[1] * 100
                    width = figsize[0] * 100

                if interactive:
                    fig = make_subplots(
                        rows=int(nrows),
                        cols=int(ncols),
                        subplot_titles=[
                            f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}"
                            for row in rows
                        ],
                    )

                    for i, row in enumerate(rows):
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)

                        if hue and hue in self.samples.columns:
                            # NEW: split/color by hue2 only if user provided hue2
                            if hue2 and (hue2 in self.samples.columns):
                                cvals = self.samples[hue2].astype(str).values
                                levels = hue2_order if hue2_order is not None else pd.unique(cvals)

                                for lvl in levels:
                                    m = (cvals == str(lvl))
                                    fig.add_trace(
                                        go.Box(
                                            y=df.values[m],
                                            x=self.samples[hue].values[m],
                                            name=str(lvl),
                                            legendgroup=str(lvl),
                                            showlegend=(i == 0),
                                            marker_color=(hue2_palette.get(str(lvl)) if isinstance(hue2_palette, dict) else None),
                                            **boxplot_params
                                        ),
                                        row=i // ncols + 1,
                                        col=i % ncols + 1
                                    )
                            else:
                                fig.add_trace(
                                    go.Box(y=df, x=self.samples[hue].values, **boxplot_params),
                                    row=i // ncols + 1,
                                    col=i % ncols + 1
                                )
                        else:
                            fig.add_trace(
                                go.Box(y=df, **boxplot_params),
                                row=i // ncols + 1,
                                col=i % ncols + 1
                            )

                        fig.update_yaxes(title_text=f"{row}", row=i // ncols + 1, col=i % ncols + 1, title_standoff=0)

                    fig.update_annotations(font_size=13)
                    fig.update_layout(title_text=f"{pathway}", showlegend=False, margin=dict(r=50))

                    # NEW: only when hue2 is used
                    if hue2 and (hue2 in self.samples.columns) and (hue and hue in self.samples.columns):
                        fig.update_layout(showlegend=True, boxmode="group")

                    if "tmp" in locals():
                        fig.update_layout(height=height, width=width)

                    if category_order:
                        fig.update_xaxes(dict(categoryarray=category_order))

                    fig.show()

                else:
                    # Create a figure and a grid of subplots
                    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                    if nrows * ncols > 1:
                        axes = np.array(axes).reshape(-1)
                    else:
                        axes = [axes]

                    # Iterate over the specified rows and plot
                    for i, row in enumerate(rows):
                        if i < len(axes):
                            ax = axes[i]
                            df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)

                            if hue and hue in self.samples.columns:
                                # NEW: seaborn hue only when hue2 is provided
                                if hue2 and (hue2 in self.samples.columns):
                                    plot_df = pd.DataFrame(
                                        {
                                            "value": df.values,
                                            "x": self.samples[hue].values,
                                            "c": self.samples[hue2].values,
                                        }
                                    )
                                    sns.boxplot(
                                        data=plot_df,
                                        x="x",
                                        y="value",
                                        hue="c",
                                        order=(category_order if category_order else None),
                                        hue_order=(hue2_order if hue2_order else None),
                                        palette=(hue2_palette if hue2_palette else None),
                                        ax=ax,
                                        **boxplot_params
                                    )
                                else:
                                    sns.boxplot(y=df, x=self.samples[hue].values, ax=ax, **boxplot_params)
                            else:
                                sns.boxplot(y=df, ax=ax, **boxplot_params)

                            # Retrieve and set the corresponding EID as the subplot title
                            eid = self.uns["compounds"][self.uns["compounds"]["input_row"] == row]["EID"].values
                            ax.set_title(f"EID: {eid}")

                    # Hide any unused subplots
                    for j in range(i + 1, len(axes)):
                        axes[j].axis("off")

                    # Set the overall title and show plot
                    fig.tight_layout()
                    fig.suptitle(f"{pathway}")
                    fig.subplots_adjust(top=0.88)
                    plt.show()
                    plt.close(fig)

                    figs.append(fig)

        else:
            # Handle 'eids' parameter and extract corresponding rows
            if eids is not None:
                if isinstance(eids, int):
                    eids = [eids]
                rows = self.uns["compounds"][
                    (self.uns["compounds"]["EID"].isin(eids)) &
                    (self.uns["compounds"]["p_value"] <= significant)
                ].input_row.values

            # If rows is None or empty after handling eids or pathway, return without plotting
            if rows is None or len(rows) == 0:
                print("No rows to plot.")
                return []

            # Ensure 'rows' is a list
            if isinstance(rows, int):
                rows = [rows]

            rows = np.unique(rows)

            # Determine the number of subplots and set default grid dimensions if not provided
            n_plots = len(rows)
            if grid_dim is None:
                grid_dim = (1, n_plots)

            nrows, ncols = grid_dim
            if figsize is None:
                figsize = (5 * ncols, 5 * nrows)
            else:
                height = figsize[1] * 100
                width = figsize[0] * 100

            if interactive:
                fig = make_subplots(rows=int(nrows), cols=int(ncols))

                for i, row in enumerate(rows):
                    df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)

                    if hue and hue in self.samples.columns:
                        # NEW: split/color by hue2 
                        if hue2 and (hue2 in self.samples.columns):
                            cvals = self.samples[hue2].astype(str).values
                            levels = hue2_order if hue2_order is not None else pd.unique(cvals)

                            for lvl in levels:
                                m = (cvals == str(lvl))
                                fig.add_trace(
                                    go.Box(
                                        y=df.values[m],
                                        x=self.samples[hue].values[m],
                                        name=str(lvl),
                                        legendgroup=str(lvl),
                                        showlegend=(i == 0),
                                        marker_color=(hue2_palette.get(str(lvl)) if isinstance(hue2_palette, dict) else None),
                                        **boxplot_params
                                    ),
                                    row=i // ncols + 1,
                                    col=i % ncols + 1
                                )
                        else:
                            fig.add_trace(
                                go.Box(y=df, x=self.samples[hue].values, **boxplot_params),
                                row=i // ncols + 1,
                                col=i % ncols + 1
                            )
                    else:
                        fig.add_trace(
                            go.Box(y=df, **boxplot_params),
                            row=i // ncols + 1,
                            col=i % ncols + 1
                        )

                    fig.update_yaxes(title_text=f"{row}", row=i // ncols + 1, col=i % ncols + 1, title_standoff=0)

                fig.update_annotations(font_size=13)
                fig.update_layout(title=title, showlegend=False, margin=dict(r=50))

                # NEW: only when hue2 is used
                if hue2 and (hue2 in self.samples.columns) and (hue and hue in self.samples.columns):
                    fig.update_layout(showlegend=True, boxmode="group")

                if "tmp" in locals():
                    fig.update_layout(height=height, width=width)

                if category_order:
                    fig.update_xaxes(dict(categoryarray=category_order))

                fig.show()

            else:
                # Create a figure and a grid of subplots
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                if nrows * ncols > 1:
                    axes = np.array(axes).reshape(-1)
                else:
                    axes = [axes]

                # Iterate over the specified rows and plot
                for i, row in enumerate(rows):
                    if i < len(axes):
                        ax = axes[i]
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)

                        if hue and hue in self.samples.columns:
                            # NEW: seaborn hue only when hue2 is provided
                            if hue2 and (hue2 in self.samples.columns):
                                plot_df = pd.DataFrame(
                                    {
                                        "value": df.values,
                                        "x": self.samples[hue].values,
                                        "c": self.samples[hue2].values,
                                    }
                                )
                                sns.boxplot(
                                    data=plot_df,
                                    x="x",
                                    y="value",
                                    hue="c",
                                    order=(category_order if category_order else None),
                                    hue_order=(hue2_order if hue2_order else None),
                                    palette=(hue2_palette if hue2_palette else None),
                                    ax=ax,
                                    **boxplot_params
                                )
                            else:
                                sns.boxplot(y=df, x=self.samples[hue].values, ax=ax, **boxplot_params)
                        else:
                            sns.boxplot(y=df, ax=ax, **boxplot_params)

                        eid = self.uns["compounds"][self.uns["compounds"]["input_row"] == row]["EID"].values
                        ax.set_title(f"EID: {eid}")

                # Set the overall title and show plot
                plt.tight_layout()
                plt.suptitle(title)
                plt.subplots_adjust(top=0.88)
                plt.close(fig)

                figs = fig

        return figs or None






    def violinplot(self, hue, pathways=None, eids=None, rows=None, significant=0.05, grid_dim=None, figsize=None, title="", zeros=True, cutoff_path=0.05,interactive=True,category_order=None, **violinplot_params):
        """
        Generate violin plots for selected ``features`` or pathway-derived feature sets.
        
        Parameters
        ----------
        hue : str
            Column in ``self.samples`` used for grouping on the x-axis.
        pathways : str or list[str] or None, optional
            Pathway name or names used to resolve ``features`` through ``show_compounds``.
        eids : sequence or None, optional
            Empirical compound identifiers used to resolve feature rows.
        rows : sequence of int or None, optional
            Explicit feature row indices to plot.
        significant : float, default=0.05
            P-value threshold applied when rows are resolved from ``pathways`` or ``eids``.
        grid_dim : tuple[int, int] or None, optional
            Explicit subplot grid dimensions.
        figsize : tuple[float, float] or None, optional
            Figure size for static output.
        title : str, default=''
            Overall plot title.
        zeros : bool, default=True
            If ``False``, replace zero values with missing values before plotting.
        cutoff_path : float, default=0.05
            Pathway p-value threshold used when ``pathways`` is provided.
        interactive : bool, default=True
            If ``True``, generate Plotly output. Otherwise generate matplotlib/seaborn output.
        category_order : list or None, optional
            Explicit order for hue categories.
        **violinplot_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        list or object
            Generated plot objects for the requested rows or pathways.
        """
        
        figs = []
        if figsize is not None:
            tmp=0
        
        # Handle the case where pathways, eids, and rows are all None
        if pathways is None and eids is None and rows is None:
            compounds_df = self.show_compounds(cutoff_path=cutoff_path, cutoff_eids=significant)
            pathways = compounds_df.index.get_level_values('pathway').unique().tolist()
    
        # If pathways is not None, handle it
        if pathways is not None:
            if isinstance(pathways, str):
                pathways = [pathways]
    
            compounds_df = self.show_compounds(paths=pathways, cutoff_path=cutoff_path, cutoff_eids=significant)
            max_input_rows = compounds_df.groupby(level='pathway')['input_row'].nunique().max()
            
            for pathway in pathways:
                pathway_compounds = compounds_df.xs(pathway, level='pathway')
                eids = pathway_compounds.index.get_level_values('compound_id').unique().tolist()
                
                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns['compounds'][(self.uns['compounds']['EID'].isin(eids)) & (self.uns['compounds']['p_value'] <= significant)].input_row.values
                
                # If no rows are found, skip this pathway
                if len(rows) == 0:
                    print(f"No significant compounds found for pathway: {pathway}")
                    continue
    
                # Ensure rows is a list
                if isinstance(rows, int):
                    rows = [rows]
    
                rows = np.unique(rows)
    
                # Determine the number of subplots and set default grid dimensions if not provided
                n_plots = len(rows)
                if grid_dim is None:
                    grid_dim = (1, max_input_rows)  # Default to one column with a row for each plot
    
                nrows, ncols = grid_dim
                if figsize is None:
                    figsize = (5 * ncols, 5 * nrows)
                else:
                    height=figsize[1]*100
                    width=figsize[0]*100
    
                # Create a figure and a grid of subplots
                #####    
                if interactive:
                    fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
                    for i, row in enumerate(rows):
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
                        if hue and hue in self.samples.columns:
                            fig.add_trace(go.Violin(y=df,x=self.samples[hue].values,**violinplot_params), row=i//ncols+1, col=i%ncols+1)
                        else:
                            fig.add_trace(go.Violin(y=df,**violinplot_params), row=i//ncols+1, col=i%ncols+1)
                            
                        fig.update_yaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=0)
                    fig.update_annotations(font_size=13)    
                    fig.update_layout(title_text=f"{pathway}",showlegend=False,margin=dict(r=50))
                    if 'tmp' in locals():
                            fig.update_layout(height=height,width=width)
                    if category_order:
                        fig.update_xaxes(dict(categoryarray=category_order))
                    fig.show()
                else:
                    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                    if nrows * ncols > 1:
                        axes = np.array(axes).reshape(-1)  # Flatten axes array for easy iteration
                    else:
                        axes = [axes]  # Ensure axes is iterable for a single subplot
        
                    # Iterate over the specified rows and plot
                    for i, row in enumerate(rows):
                        if i < len(axes):  # Check to avoid IndexError
                            ax = axes[i]
                            df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
        
                            if hue and hue in self.samples.columns:
                                sns.violinplot(y=df, x=self.samples[hue].values, ax=ax, **violinplot_params)
                            else:
                                sns.violinplot(y=df, ax=ax, **violinplot_params)
        
                            # Retrieve and set the corresponding EID as the subplot title
                            eid = self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values
                            ax.set_title(f"EID: {eid}")
        
                    # Hide any unused subplots
                    for j in range(i+1, len(axes)):
                        axes[j].axis('off')
        
                    # Set the overall title and show plot
                    fig.tight_layout()
                    fig.suptitle(f"{pathway}")
                    fig.subplots_adjust(top=0.88)  # Adjust subplots to fit the main title
                    plt.show()
                    plt.close(fig)
        
                    # Append the figure to the list
                    figs.append(fig)
        else:
            # Handle 'eids' parameter and extract corresponding rows
            if eids is not None:
                if isinstance(eids, int):
                    eids = [eids]
                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns['compounds'][(self.uns['compounds']['EID'].isin(eids)) & (self.uns['compounds']['p_value'] <= significant)].input_row.values
            
            # If rows is None or empty after handling eids or pathway, return without plotting
            if rows is None or len(rows) == 0:
                print("No rows to plot.")
                return []
    
            # Ensure 'rows' is a list
            if isinstance(rows, int):
                rows = [rows]
    
            rows = np.unique(rows)
    
            # Determine the number of subplots and set default grid dimensions if not provided
            n_plots = len(rows)
            if grid_dim is None:
                grid_dim = (1, n_plots)  # Default to one column with a row for each plot
    
            nrows, ncols = grid_dim
            if figsize is None:
                figsize = (5 * ncols, 5 * nrows)
            else:
                height=figsize[1]*100
                width=figsize[0]*100
                
            if interactive:
                fig = make_subplots(rows=int(nrows), cols=int(ncols))
                for i, row in enumerate(rows):
                    df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
                    if hue and hue in self.samples.columns:
                        fig.add_trace(go.Violin(y=df,x=self.samples[hue].values,**violinplot_params), row=i//ncols+1, col=i%ncols+1)
                    else:
                        fig.add_trace(go.Violin(y=df,**violinplot_params), row=i//ncols+1, col=i%ncols+1)
                        
                    fig.update_yaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=0)
                fig.update_annotations(font_size=13)    
                fig.update_layout(title=title,showlegend=False,margin=dict(r=50))
                if 'tmp' in locals():
                        fig.update_layout(height=height,width=width)
                if category_order:
                        fig.update_xaxes(dict(categoryarray=category_order))
                fig.show()
            else:
                # Create a figure and a grid of subplots
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                if nrows * ncols > 1:
                    axes = np.array(axes).reshape(-1)  # Flatten axes array for easy iteration
                else:
                    axes = [axes]  # Ensure axes is iterable for a single subplot
        
                # Iterate over the specified rows and plot
                for i, row in enumerate(rows):
                    if i < len(axes):  # Check to avoid IndexError
                        ax = axes[i]
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
        
                        if hue and hue in self.samples.columns:
                            sns.violinplot(y=df, x=self.samples[hue].values, ax=ax, **violinplot_params)
                        else:
                            sns.violinplot(y=df, ax=ax, **violinplot_params)
        
                        # Retrieve and set the corresponding EID as the subplot title
                        eid = self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values
                        ax.set_title(f"EID: {eid}")
        
                # Set the overall title and show plot
                plt.tight_layout()
                plt.suptitle(title)
                plt.subplots_adjust(top=0.88) # Adjust subplots to fit the main title
                plt.close(fig)
        
                figs=fig
    
        return figs or None



    def regplot(self, column, pathways=None, eids=None, rows=None, significant=0.05, grid_dim=None, figsize=None, title="", zeros=True, x_limit=None, cutoff_path=0.05,interactive=True, trend='ols', **regplot_params):
        """
        Generate regression plots for selected ``features`` or pathway-derived feature sets.
        
        Parameters
        ----------
        column : str
            Continuous column in ``self.samples`` used for the x-axis.
        pathways : str or list[str] or None, optional
            Pathway name or names used to resolve ``features`` through ``show_compounds``.
        eids : sequence or None, optional
            Empirical compound identifiers used to resolve feature rows.
        rows : sequence of int or None, optional
            Explicit feature row indices to plot.
        significant : float, default=0.05
            P-value threshold applied when rows are resolved from ``pathways`` or ``eids``.
        grid_dim : tuple[int, int] or None, optional
            Explicit subplot grid dimensions.
        figsize : tuple[float, float] or None, optional
            Figure size for static output.
        title : str, default=''
            Overall plot title.
        zeros : bool, default=True
            If ``False``, replace zero values with missing values before plotting.
        x_limit : tuple or None, optional
            Optional x-axis limits for static output.
        cutoff_path : float, default=0.05
            Pathway p-value threshold used when ``pathways`` is provided.
        interactive : bool, default=True
            If ``True``, generate Plotly output. Otherwise generate matplotlib/seaborn output.
        trend : str or None, default='ols'
            Trend-line mode used for interactive output.
        **regplot_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        list or object
            Generated plot objects for the requested rows or pathways.
        """
        figs=[]
        if figsize is not None:
            tmp=0
    
        # Handle the case where pathways, eids, and rows are all None
        if pathways is None and eids is None and rows is None:
            compounds_df = self.show_compounds(cutoff_path=cutoff_path, cutoff_eids=significant)
            pathways = compounds_df.index.get_level_values('pathway').unique().tolist()
    
        # If pathways is not None, handle it
        if pathways is not None:
            if isinstance(pathways, str):
                pathways = [pathways]
    
            compounds_df = self.show_compounds(paths=pathways, cutoff_path=cutoff_path, cutoff_eids=significant)
            max_input_rows = compounds_df.groupby(level='pathway')['input_row'].nunique().max()
            
            for pathway in pathways:
                pathway_compounds = compounds_df.xs(pathway, level='pathway')
                eids = pathway_compounds.index.get_level_values('compound_id').unique().tolist()
                
                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns['compounds'][(self.uns['compounds']['EID'].isin(eids)) & (self.uns['compounds']['p_value'] <= significant)].input_row.values
                
                # If no rows are found, skip this pathway
                if len(rows) == 0:
                    print(f"No significant compounds found for pathway: {pathway}")
                    continue
    
                # Ensure rows is a list
                if isinstance(rows, int):
                    rows = [rows]
    
                rows = np.unique(rows)
    
                # Determine the number of subplots and set default grid dimensions if not provided
                n_plots = len(rows)
                if grid_dim is None:
                    grid_dim = (1, max_input_rows)  # Default to one column with a row for each plot
    
                nrows, ncols = grid_dim
                if figsize is None:
                    figsize = (5 * ncols, 5 * nrows)
                else:
                    height=figsize[1]*100
                    width=figsize[0]*100
                #####    
                if interactive:
                    fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
                    for i, row in enumerate(rows):
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
    
                        scatter_trace=go.Scatter(y=df,x=self.samples[column].values,mode="markers",hovertext=self.samples.iloc[:,0],**regplot_params)
                        fig.add_trace(scatter_trace, row=i//ncols+1, col=i%ncols+1)
                        if trend:
                            trendline = px.scatter(y=df,x=self.samples[column].values, trendline=trend,trendline_color_override='#BE9B7B').data[1]
                            fig.add_trace(trendline,row=i//ncols+1, col=i%ncols+1)
                            
                        fig.update_yaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=0)
                    fig.update_annotations(font_size=13)    
                    fig.update_layout(title_text=f"{pathway}",showlegend=False,margin=dict(r=50))
                    if 'tmp' in locals():
                            fig.update_layout(height=height,width=width)
                    fig.show()

                else:
                    # Create a figure and a grid of subplots
                    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                    if nrows * ncols > 1:
                        axes = np.array(axes).reshape(-1)  # Flatten axes array for easy iteration
                    else:
                        axes = [axes]  # Ensure axes is iterable for a single subplot
        
                    # Iterate over the specified rows and plot
                    for i, row in enumerate(rows):
                        if i < len(axes):  # Check to avoid IndexError
                            ax = axes[i]
                            df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
        
                            sns.regplot(y=df, x=self.samples[column], ax=ax, **regplot_params)
        
                            # Retrieve and set the corresponding EID as the subplot title
                            eid = self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values
                            ax.set_title(f"EID: {eid}")
                            ax.set_xlim(x_limit)
        
                    # Hide any unused subplots
                    for j in range(i+1, len(axes)):
                        axes[j].axis('off')
        
                    # Set the overall title and show plot
                    fig.tight_layout()
                    fig.suptitle(f"{pathway}")
                    fig.subplots_adjust(top=0.88)  # Adjust subplots to fit the main title
                    plt.show()
                    plt.close(fig)
        
                    # Append the figure to the list
                    figs.append(fig)
        else:
            # Handle 'eids' parameter and extract corresponding rows
            if eids is not None:
                if isinstance(eids, int):
                    eids = [eids]
                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns['compounds'][(self.uns['compounds']['EID'].isin(eids)) & (self.uns['compounds']['p_value'] <= significant)].input_row.values
            
            # If rows is None or empty after handling eids or pathway, return without plotting
            if rows is None or len(rows) == 0:
                print("No rows to plot.")
                return []
    
            # Ensure 'rows' is a list
            if isinstance(rows, int):
                rows = [rows]
    
            rows = np.unique(rows)
    
            # Determine the number of subplots and set default grid dimensions if not provided
            n_plots = len(rows)
            if grid_dim is None:
                grid_dim = (1, n_plots)  # Default to one column with a row for each plot
    
            nrows, ncols = grid_dim
            if figsize is None:
                figsize = (5 * ncols, 5 * nrows)
            else:
                height=figsize[1]*100
                width=figsize[0]*100
            if interactive:
                fig = make_subplots(rows=int(nrows), cols=int(ncols))
                for i, row in enumerate(rows):
                    df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)

                    scatter_trace=go.Scatter(y=df,x=self.samples[column].values,mode="markers",hovertext=self.samples.iloc[:,0],**regplot_params)
                    fig.add_trace(scatter_trace, row=i//ncols+1, col=i%ncols+1)
                    
                    trendline = px.scatter(y=df,x=self.samples[column].values, trendline=trend,trendline_color_override='#BE9B7B').data[1]
                    fig.add_trace(trendline,row=i//ncols+1, col=i%ncols+1)
                        
                    fig.update_yaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=0)
                fig.update_annotations(font_size=13)    
                fig.update_layout(title=title,showlegend=False,margin=dict(r=50))
                if 'tmp' in locals():
                        fig.update_layout(height=height,width=width)
                fig.show()

            else:
                # Create a figure and a grid of subplots
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                if nrows * ncols > 1:
                    axes = np.array(axes).reshape(-1)  # Flatten axes array for easy iteration
                else:
                    axes = [axes]  # Ensure axes is iterable for a single subplot
        
                # Iterate over the specified rows and plot
                for i, row in enumerate(rows):
                    if i < len(axes):  # Check to avoid IndexError
                        ax = axes[i]
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
        
                        sns.regplot(y=df, x=self.samples[column], ax=ax, **regplot_params)
        
                        # Retrieve and set the corresponding EID as the subplot title
                        eid = self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values
                        ax.set_title(f"EID: {eid}")
                        ax.set_xlim(x_limit)
        
                # Set the overall title and show plot
                plt.tight_layout()
                plt.suptitle(title)
                plt.subplots_adjust(top=0.88) # Adjust subplots to fit the main title
                plt.close(fig)
        
                figs=fig
    
        return figs or None

    def volcano(self, p, effect_size, sign_line=0.05, annotation='index',
            effect_size_line=None, logp=True, title="", legend=None, genomewideline_color='#EF553B', effect_size_line_color='#EF553B', highlight_color='#119DFF', col='#2A3F5F',
                effect_size_line_width=1, **volcano_params):
        """
        Generate a volcano plot from feature statistics.
        
        Parameters
        ----------
        p : str
            Column in ``self.features`` containing p-values.
        effect_size : str
            Column in ``self.features`` containing effect sizes.
        sign_line : float, default=0.05
            Threshold used for the horizontal significance line.
        annotation : str, default='index'
            Column name or index label used for point annotations.
        effect_size_line : list or None, optional
            Optional pair of x positions for vertical effect-size threshold lines.
        logp : bool, default=True
            If ``True``, transform p-values to ``-log10(p)``.
        title : str, default=''
            Plot title.
        legend : dict or None, optional
            Optional custom legend specification.
        genomewideline_color : str, default='#EF553B'
            Color of the significance line.
        effect_size_line_color : str, default='#EF553B'
            Color of the effect-size threshold lines.
        highlight_color : str, default='#119DFF'
            Color used for highlighted points.
        col : str, default='#2A3F5F'
            Base point color.
        effect_size_line_width : int, default=1
            Width of the effect-size threshold lines.
        **volcano_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Interactive volcano plot.
        
        Raises
        ------
        ValueError
            If ``self.features`` is missing or required columns are not present.
        """
    
        if self.features is None:
            raise ValueError("The 'features' attribute must be assigned before plotting a volcano plot.")
        
        # Ensure effect_size_line is initialized
        if effect_size_line is None:
            effect_size_line = [False, False]
        
        # Check if required columns exist in self.features
        if p not in self.features.columns or effect_size not in self.features.columns:
            raise ValueError(f"Columns '{p}' and '{effect_size}' must exist in 'self.features'.")
        
        # Log-transform p-values if necessary
        ylabel = f'-log10 [{p}]' if logp else p
        if logp:
            sign_line = -np.log10(sign_line)
        
        # Generate the plot
        fig = dash_bio.VolcanoPlot(
            self.features.reset_index(),
            p=p,
            effect_size=effect_size,
            snp=None,
            gene=None,
            annotation=annotation,
            logp=logp,
            genomewideline_value=sign_line,
            effect_size_line=effect_size_line,
            effect_size_line_color=effect_size_line_color,
            effect_size_line_width=effect_size_line_width,
            genomewideline_color=genomewideline_color,
            highlight_color=highlight_color,
            col=col,
            xlabel=effect_size,
            ylabel=ylabel,
            legend=legend,
            **volcano_params
        )
    
        # Adjust x-axis range
        fig.update_xaxes(range=[
            self.features[effect_size].min() - self.features[effect_size].std() / 5,
            self.features[effect_size].max() + self.features[effect_size].std() / 5
        ])
        fig.update_layout(
        title={
            'text': f"{title}",
            'x': 0.45,
            'xanchor': 'center'
        })
    
        return fig




    def save(self,path):
        """
        Serialize the current Rodin object to a pickle file.
        
        Parameters
        ----------
        path : str or path-like
            Destination path for the pickle file.
        
        Returns
        -------
        None
            The object is written to disk in pickle format.
        """
        
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    def analyze_pathways(self,pvals,stats,network='human_mfn',mode='f_positive',instrument='unspecified',permutation=100,
                         force_primary_ion=True,cutoff=0,modeling=None,pws_name='pathways',cmp_name='compounds'):
        """
        Run pathway analysis from feature-level statistics.
        
        Parameters
        ----------
        pvals : str
            Column in ``self.features`` containing p-values.
        stats : str
            Column in ``self.features`` containing feature-level statistics or effect scores.
        network : str, default='human_mfn'
            Metabolic network model to use.
        mode : str, default='f_positive'
            Ionization or acquisition mode used for metabolite matching.
        instrument : str, default='unspecified'
            Instrument or mass-accuracy specification.
        permutation : int, default=100
            Number of permutations used for enrichment estimation.
        force_primary_ion : bool, default=True
            Whether to enforce primary-ion matching.
        cutoff : float, default=0
            Feature-level cutoff used during pathway analysis.
        modeling : str or None, optional
            Optional permutation-modeling strategy.
        pws_name : str, default='pathways'
            Key used to store pathway results in ``self.uns``.
        cmp_name : str, default='compounds'
            Key used to store compound-level matches in ``self.uns``.
        
        Returns
        -------
        pandas.DataFrame
            Pathway-enrichment table also stored in ``self.uns[pws_name]``.
        
        Notes
        -----
        Compound-level matches generated during the analysis are also stored in ``self.uns[cmp_name]``.
        """

        arg_dict = {
        '-n': network,
        '-m': mode,
        '-u': instrument,
        '-p': permutation,
        '-z': force_primary_ion,
        '-c': cutoff,
        '-d': modeling,
        }

        args = [(f"-{k.strip('-')}", v) for k, v in arg_dict.items()]

        optdict = dispatcher(args)
    
        print_and_loginfo("Started @ %s\n" %time.asctime())
        df = self.features.loc[:,[self.features.columns[0],self.features.columns[1],pvals,stats]]

        userData = InputUserData(optdict,df)
        
        #specify which metabolic model 
        if userData.paradict['network'] in ['human', 'hsa', 'Human', 'human_mfn', 'hsa_mfn', '']:
            theoreticalModel = metabolicNetwork(metabolicModels[ 'human_model_mfn' ])
    
        elif userData.paradict['network'] in ['worm', 'C. elegans', 'icel1273', 'Caenorhabditis elegans']:
            theoreticalModel = metabolicNetwork(metabolicModels[ 'worm_model_icel1273' ])
        #
        # get user specified model
        #
        else:
            try:
                theoreticalModel = metabolicNetwork(
                    # get user input JSON model and convert to mummichog format
                    read_user_json_model(userData.paradict['network'])
                )      
            except FileNotFoundError:
                raise FileNotFoundError( "Not being able to find ", userData.paradict['network'] )
            finally:
                print("Support of custom metabolic models is in progress. Pls contact author.")
        
        # calculating isotopes/adducts, to test, print(list(theoreticalModel.Compounds.items())[64: 70])
        theoreticalModel.update_Compounds_adducts(mode=userData.paradict['mode'])
    
        # matching model data with user data
        mixedNetwork = DataMeetModel(theoreticalModel, userData)
    
        # getting a list of Pathway instances, with p-values, in PA.resultListOfPathways
        PA = PathwayAnalysis(mixedNetwork.model.metabolic_pathways, mixedNetwork)
        PA.cpd_enrich_test()

        s = "input_row\tEID\tstr_row_ion\tcompounds\tcompound_names\tinput_row\tm/z\tretention_time\tp_value\tstatistic\tCompoundID_from_user\n"
        for row in mixedNetwork.mzrows:
            # not all input rows match to an empCpd
            try:
                for E in mixedNetwork.rowindex_to_EmpiricalCompounds[row]:
                    names = [mixedNetwork.model.dict_cpds_def.get(x, '') for x in E.compounds]
                    s += '\t'.join([row, E.EID, E.str_row_ion, ';'.join(E.compounds), '$'.join(names)]
                        ) + '\t' + mixedNetwork.rowDict[row].make_str_output() + '\n'
            except KeyError:
                pass
        data_io = StringIO(s)
        # Read the string into a DataFrame
        df = pd.read_csv(data_io, sep='\t')
        for index, row in df.iterrows():
            input_row_value = row['input_row']
            str_row_ion_values = row['str_row_ion'].split(';')
    
            # Regex pattern to match 'row{number}_'
            pattern = re.compile(re.escape(input_row_value) + r'_[^;]+')
    
            # Find all matches
            matches = pattern.findall(row['str_row_ion'])
    
            # Clean the values by removing the "row{number}_" part
            cleaned_values = [re.sub(r'^' + re.escape(input_row_value) + '_', '', m) for m in matches]
    
            # Update the DataFrame
            df.at[index, 'str_row_ion'] = ';'.join(cleaned_values)
        df.drop(['CompoundID_from_user','input_row.1'],axis=1,inplace=True)
        df['input_row'] = df['input_row'].str.extract('(\d+)').astype(int)
        df['compound_names'] = df['compound_names'].apply(lambda x: None if isinstance(x, str) and x.replace('$', '') == '' else x)

        resultstr = [['pathway', 'overlap_size', 'pathway_size', 'p-value', 
                      'overlap_EmpiricalCompounds (id)', 'overlap_features (id)', 'overlap_features (name)',] ]
        
        for P in PA.resultListOfPathways:
            empCpds = [E.EID for E in P.overlap_EmpiricalCompounds]
            cpds = [E.chosen_compounds for E in P.overlap_EmpiricalCompounds]
            names = [ [mixedNetwork.model.dict_cpds_def.get(x, '') for x in y] for y in cpds ]
            resultstr.append([str(x) for x in [P.name, P.overlap_size, P.EmpSize, P.adjusted_p]]
                             + [','.join(empCpds), ','.join(['/'.join(x) for x in cpds]), '$'.join(['/'.join(x) for x in names]) ])

        df2 = pd.DataFrame(resultstr[1:], columns=resultstr[0]).drop(['overlap_features (id)','overlap_features (name)'],axis=1)

        df['input_row'] = df['input_row']-1
        self.uns[pws_name] = df2
        self.uns[cmp_name] = df
        
        return df2
                             

    def show_compounds(self, cutoff_path=0.05, cutoff_eids=0.05, paths=None):
        """
        Return compound-level matches for selected pathways.
        
        Parameters
        ----------
        cutoff_path : float, default=0.05
            Maximum pathway p-value used when ``paths`` is not provided.
        cutoff_eids : float, default=0.05
            Maximum compound p-value used to retain matched compounds.
        paths : str or list[str] or None, optional
            Explicit pathway name or names to include.
        
        Returns
        -------
        pandas.DataFrame
            Compound table indexed by pathway and empirical compound identifier.
        """
        
        # Handle paths parameter and filter pathways
        if paths is not None:
            if isinstance(paths, str):
                paths = [paths]
            pathways_df_filtered = self.uns['pathways'][self.uns['pathways']['pathway'].isin(paths)]
        else:
            pathways_df_filtered = self.uns['pathways'][self.uns['pathways']['p-value'].astype(float) <= cutoff_path]
    
        # Prepare a list to store the results
        results = []
    
        # Iterate through filtered pathways and collect compounds
        for index, pathway_row in pathways_df_filtered.iterrows():
            pathway = pathway_row['pathway']
            compound_ids = pathway_row['overlap_EmpiricalCompounds (id)'].split(',')
            
            # Filter compounds based on cutoff_eids
            compounds_df = self.uns['compounds']
            compounds_filtered = compounds_df[
                (compounds_df['EID'].isin(compound_ids)) &
                (compounds_df['p_value'].astype(float) <= cutoff_eids)
            ]
            
            # Append the results for this pathway
            if not compounds_filtered.empty:
                for _, compound_row in compounds_filtered.iterrows():
                    results.append({
                        'pathway': pathway,
                        'compound_id': compound_row['EID'],
                        'str_row_ion': compound_row['str_row_ion'],
                        'compounds': compound_row['compounds'],
                        'compound_names': compound_row['compound_names'],
                        'input_row': compound_row['input_row'],
                        'm/z': compound_row['m/z'],
                        'retention_time': compound_row['retention_time'],
                        'p_value': compound_row['p_value'],
                        'statistic': compound_row['statistic']
                    })
    
        results_df = pd.DataFrame(results)
        results_df.set_index(['pathway', 'compound_id'], inplace=True)
        
        return results_df


    #####################    
    def web_boxplot(self, hue, dist='box', pathways=None, eids=None, rows=None, significant=0.05, grid_dim=None, figsize=None, title="", zeros=True, cutoff_path=0.05,interactive=True,category_order=None, trend='ols',points='all',**boxplot_params):

        """
        Generate web-oriented plot objects for box, violin, or scatter distributions.
        
        Parameters
        ----------
        hue : str
            Column in ``self.samples`` used for grouping or the x-axis.
        dist : {'box', 'violin', 'scatter'}, default='box'
            Plot type to generate.
        pathways : str or list[str] or None, optional
            Pathway name or names used to resolve ``features`` through ``show_compounds``.
        eids : sequence or None, optional
            Empirical compound identifiers used to resolve feature rows.
        rows : sequence of int or None, optional
            Explicit feature row indices to plot.
        significant : float, default=0.05
            P-value threshold applied when rows are resolved from ``pathways`` or ``eids``.
        grid_dim : tuple[int, int] or None, optional
            Explicit subplot grid dimensions.
        figsize : tuple[float, float] or None, optional
            Figure size for static output when applicable.
        title : str, default=''
            Overall plot title.
        zeros : bool, default=True
            If ``False``, replace zero values with missing values before plotting.
        cutoff_path : float, default=0.05
            Pathway p-value threshold used when ``pathways`` is provided.
        interactive : bool, default=True
            If ``True``, generate Plotly output.
        category_order : list or None, optional
            Explicit order for x-axis categories.
        trend : str or None, default='ols'
            Trend-line mode used for scatter output.
        points : str, default='all'
            Point-display mode for violin plots.
        **boxplot_params : dict
            Additional keyword arguments forwarded to the plotting backend.
        
        Returns
        -------
        list
            Plot objects generated for the requested rows or pathways.
        """
        figs = []
        
        # Handle the case where pathways, eids, and rows are all None
        if pathways is None and eids is None and rows is None:
            compounds_df = self.show_compounds(cutoff_path=cutoff_path, cutoff_eids=significant)
            pathways = compounds_df.index.get_level_values('pathway').unique().tolist()
        
        
        if pathways is not None and rows is None:
            if isinstance(pathways, str):
                pathways = [pathways]
    
            compounds_df = self.show_compounds(paths=pathways, cutoff_path=cutoff_path, cutoff_eids=significant)
            max_input_rows = compounds_df.groupby(level='pathway')['input_row'].nunique().max()
            
            for pathway in pathways:
                pathway_compounds = compounds_df.xs(pathway, level='pathway')
                eids = pathway_compounds.index.get_level_values('compound_id').unique().tolist()
                
                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns['compounds'][(self.uns['compounds']['EID'].isin(eids)) & (self.uns['compounds']['p_value'] <= significant)].input_row.values
                
                # If no rows are found, skip this pathway
                if len(rows) == 0:
                    print(f"No significant compounds found for pathway: {pathway}")
                    continue
    
                # Ensure rows is a list
                if isinstance(rows, int):
                    rows = [rows]
    
                rows = np.unique(rows)
    
                # Determine the number of subplots and set default grid dimensions if not provided
                n_plots = len(rows)
                if n_plots>4:
                    nnrows=n_plots//4 + (n_plots%4>0)
                    ncols=4
                else:
                    nnrows=1
                    ncols=max_input_rows if max_input_rows<=4 else 4
                    
                grid_dim = (nnrows, ncols)  # Default to one column with a row for each plot
    
                nrows, ncols = grid_dim
                if nnrows>1:
                    height=nnrows*325
                    tmp=0
                else:
                    tmp=1
                    
              
                #####    
                if interactive:
                    fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
                    for i, row in enumerate(rows):
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
                        if dist=='box':
                            if hue and hue in self.samples.columns:
                                fig.add_trace(go.Box(y=df,x=self.samples[hue].values,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                            else:
                                fig.add_trace(go.Box(y=df,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                        elif dist=='violin':
                            if hue and hue in self.samples.columns:
                                fig.add_trace(go.Violin(y=df,x=self.samples[hue].values,points=points,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                            else:
                                fig.add_trace(go.Violin(y=df,points=points,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                        else:
                            scatter_trace=go.Scatter(y=df,x=self.samples[hue].values,mode="markers",hovertext=self.samples.iloc[:,0],**boxplot_params)
                            fig.add_trace(scatter_trace, row=i//ncols+1, col=i%ncols+1)
                            if trend:
                                trendline = px.scatter(y=df,x=self.samples[hue].values, trendline=trend,trendline_color_override='#BE9B7B').data[1]
                                fig.add_trace(trendline,row=i//ncols+1, col=i%ncols+1)
    
                            
                        fig.update_xaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=10)
                    fig.update_annotations(font_size=13)    
                    fig.update_layout(title_text=f"{pathway}",showlegend=False,margin=dict(b=10,r=50))
                    if tmp ==0:
                        fig.update_layout(height=height)
                    if category_order:
                        fig.update_xaxes(dict(categoryarray=category_order))
                    figs.append(fig)
        else:
            # Handle 'eids' parameter and extract corresponding rows
            if eids is not None:
                if isinstance(eids, int):
                    eids = [eids]
                # Filter for rows with p_value less than or equal to the significant threshold
                rows = self.uns['compounds'][(self.uns['compounds']['EID'].isin(eids)) & (self.uns['compounds']['p_value'] <= significant)].input_row.values
            
            # If rows is None or empty after handling eids or pathway, return without plotting
            if rows is None or len(rows) == 0:
                print("No rows to plot.")
                return []
    
            # Ensure 'rows' is a list
            if isinstance(rows, int):
                rows = [rows]
    
            rows = np.unique(rows)
    
            # Determine the number of subplots and set default grid dimensions if not provided
            n_plots = len(rows)
            if n_plots>4:
                nnrows=n_plots//4 + (n_plots%4>0)
                ncols=4
            else:
                nnrows=1
                ncols=n_plots   
                
            grid_dim = (nnrows, ncols)  # Default to one column with a row for each plot
    
            nrows, ncols = grid_dim
            if nnrows>1:
                height=nnrows*325
                tmp=0
            else:
                tmp=1
            #####    
            if interactive:
                fig = make_subplots(rows=int(nrows), cols=int(ncols))
                for i, row in enumerate(rows):
                    df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
                    if dist=='box':
                        if hue and hue in self.samples.columns:
                            fig.add_trace(go.Box(y=df,x=self.samples[hue].values,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                        else:
                            fig.add_trace(go.Box(y=df,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                    elif dist=='violin':
                        if hue and hue in self.samples.columns:
                            fig.add_trace(go.Violin(y=df,x=self.samples[hue].values,points=points,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                        else:
                            fig.add_trace(go.Violin(y=df,points=points,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                    else:
                        scatter_trace=go.Scatter(y=df,x=self.samples[hue].values,mode="markers",hovertext=self.samples.iloc[:,0],**boxplot_params)
                        fig.add_trace(scatter_trace, row=i//ncols+1, col=i%ncols+1)
                        if trend:
                            trendline = px.scatter(y=df,x=self.samples[hue].values, trendline=trend,trendline_color_override='#BE9B7B').data[1]
                            fig.add_trace(trendline,row=i//ncols+1, col=i%ncols+1)
    
                    fig.update_xaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=10)
                fig.update_annotations(font_size=13)    
                fig.update_layout(title=title,showlegend=False,margin=dict(r=50))
                if tmp == 0:
                        fig.update_layout(height=height)
                if category_order:
                        fig.update_xaxes(dict(categoryarray=category_order))
                        
                figs.append(fig)
                        
        return figs
        
        
    def web_plot(self, hue=None, dr_name='umap', size=None, markers=None,title = "",interactive=True, **scatterplot_params):
            
        """
        Generate a Plotly embedding plot for web contexts.
        
        Parameters
        ----------
        hue : str or None, optional
            Column in ``self.samples`` used for point color.
        dr_name : str, default='umap'
            Key of the dimensionality-reduction result stored in ``self.dr``.
        size : str or None, optional
            Column in ``self.samples`` used for point size.
        markers : str or None, optional
            Column in ``self.samples`` used for point symbol.
        title : str, default=''
            Plot title.
        interactive : bool, default=True
            Reserved for API compatibility.
        **scatterplot_params : dict
            Additional keyword arguments forwarded to Plotly.
        
        Returns
        -------
        plotly.graph_objects.Figure
            Plotly scatter figure for the requested embedding.
        
        Raises
        ------
        ValueError
            If the requested embedding or metadata columns are missing.
        """
        if dr_name not in self.dr:
            raise ValueError(f"Reduction '{dr_name}' not found in 'dr'.")
    
        if hue and hue not in self.samples:
            raise ValueError(f"Column '{hue}' not found in 'samples'.")
        if size and size not in self.samples:
            raise ValueError(f"Column '{size}' not found in 'samples'.")
        if markers and markers not in self.samples:
            raise ValueError(f"Column '{markers}' not found in 'samples'.")
    
        # Retrieve the reduction data
        dr_data = self.dr[dr_name]
    
        # Check if reduction data is 2D
        if dr_data.shape[1] != 2:
            raise ValueError("Reduction data must be 2-dimensional for plotting.")
    
        if interactive:
            # Prepare the plot
            scatter_args = {'x': dr_data[:, 0], 'y': dr_data[:, 1],
                            'hover_name': self.samples.iloc[:,0], **scatterplot_params}
            
            if hue: scatter_args['color'] = self.samples[hue]
            if size: scatter_args['size'] = self.samples[size]
            if markers: scatter_args['symbol'] = self.samples[markers]
            
        
            # Prepare the plot
            ax=px.scatter(**scatter_args,labels={
                     "x": f'{dr_name}_1',
                     "y": f'{dr_name}_2'})
            ax.update_layout(
                title={
                    'text': f"{title}",
                    'x': 0.45,
                    'xanchor': 'center'
            },margin=dict(t=100))
        
        return ax

                         
        



def create_object_csv(file_path_features, file_path_classes,feat_sep='\t',class_sep='\t', feat_stat='mzrt'):
    """
    Create a ``Rodin_Class`` object from feature and metadata CSV files.
    
    Parameters
    ----------
    file_path_features : str or path-like
        Path to the feature table.
    file_path_classes : str or path-like
        Path to the sample metadata table.
    feat_sep : str, default='\\t'
        Field separator used in the feature table.
    class_sep : str, default='\\t'
        Field separator used in the sample metadata table.
    feat_stat : {'mzrt', 'ann'}, default='mzrt'
        Feature-table layout. Use ``'mzrt'`` when the first two columns are ``m/z`` and retention time,
        or ``'ann'`` when the first column stores feature annotations.
    
    Returns
    -------
    Rodin_Class
        Rodin object populated from the provided files.
    """
    
    # Load the data
    data = pd.read_csv(file_path_features,sep = feat_sep)
    # Extract features DataFrame (first two columns)
    if(feat_stat=='ann'):
        features = data.iloc[:, :1]
        X = data.iloc[:, 1:]
    else:
        features = data.iloc[:, :2]
        features = features.astype('float')
        # Extract X matrix (all columns except the first two and the last annotation columns)
        # Converting to a sparse matrix if needed
        X = data.iloc[:, 2:]

    # Create a placeholder samples DataFrame
    # This can be replaced with actual sample annotations if available
    samples = pd.read_csv(file_path_classes,sep = class_sep)
    samples.iloc[:,0] = samples.iloc[:,0].astype(str)

    return Rodin_Class(X=X, features=features,samples=samples)

def _sep(src, hint=None) -> str:
    """
    Infer a table delimiter from a file path, URL, or file-like object.
    
    Parameters
    ----------
    src : str or path-like or file-like
        Input source used to read the first line.
    hint : str or None, optional
        Explicit delimiter to return without autodetection.
    
    Returns
    -------
    str
        Detected delimiter, currently tab or comma.
    
    Raises
    ------
    ValueError
        If no supported delimiter can be detected.
    """
    if hint is not None:
        return hint

    if isinstance(src, (str, os.PathLike)):
        if src.startswith(("http://", "https://")):
            with urlopen(src) as fh:
                line = fh.readline()
        else:
            with open(src, "rb") as fh:
                line = fh.readline()
    else:
        pos = src.tell() if hasattr(src, "tell") else None
        if hasattr(src, "seek"):
            src.seek(0)
        line = src.readline()
        if isinstance(line, str):
            pass
        else:
            try:
                line = line.decode("utf-8-sig", "replace")
            except Exception:
                line = line.decode("utf-8", "replace")
        if pos is not None and hasattr(src, "seek"):
            src.seek(pos)
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8-sig", "replace")
        except Exception:
            line = line.decode("utf-8", "replace")

    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    raise ValueError("Cannot detect separator (no ',' or '\t' found).")

def _auto_mode(df, hint=None) -> str:
    """
    Infer the feature-table mode from column headers and values.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Feature table after any leading ``#`` index column has been removed.
    hint : {'mzrt', 'ann'} or None, optional
        Explicit mode to use without autodetection.
    
    Returns
    -------
    str
        Either ``'mzrt'`` or ``'ann'``.
    
    Notes
    -----
    The function prefers ``'mzrt'`` when header names look like ``m/z`` and retention time,
    or when the first feature column is mostly numeric.
    """
    if hint in ("mzrt", "ann"):
        return hint
    # header hints
    c0 = str(df.columns[0]).lower() if df.shape[1] >= 1 else ""
    c1 = str(df.columns[1]).lower() if df.shape[1] >= 2 else ""
    if any(k in c0 for k in ("mz", "m/z", "mass")) and any(k in c1 for k in ("rt", "retention","time","r/t")):
        return "mzrt"

    col0_num = pd.to_numeric(df.iloc[:, 0], errors="coerce").notna().mean() if df.shape[1] >= 1 else 0.0
    return "mzrt" if col0_num >= 0.8 else "ann"

def create(features_file, meta_file=None, feat_sep=None, meta_sep=None, mode=None):
    """
    Create a ``Rodin_Class`` object from a feature table and optional metadata.
    
    This function reads a feature-intensity table together with optional sample
    metadata, aligns overlapping sample identifiers, and returns a Rodin object
    with synchronized ``X``, ``features``, and ``samples`` tables.
    
    Parameters
    ----------
    features_file : str or path-like or file-like
        Feature table in CSV or TSV format.
    meta_file : str or path-like or file-like or None, optional
        Sample metadata table. If omitted, a minimal ``samples`` table is created from the columns of ``X``.
    feat_sep : str or None, optional
        Delimiter for the feature table. If ``None``, the delimiter is inferred automatically.
    meta_sep : str or None, optional
        Delimiter for the metadata table. If ``None``, the delimiter is inferred automatically.
    mode : {'mzrt', 'ann'} or None, optional
        Feature-table layout. If ``None``, the layout is inferred from header names and the first feature column.
    
    Returns
    -------
    Rodin_Class
        Rodin object with aligned ``X``, ``features``, and ``samples``. The returned object also stores
        ``obj.uns['mode']`` and ``obj.uns['file_type']``.
    
    Raises
    ------
    ValueError
        If delimiters cannot be inferred, if sample IDs do not overlap between the feature and metadata tables,
        or if a leading ``#`` index column has inconsistent length.
    
    Notes
    -----
    When metadata is provided, only overlapping sample IDs are retained and the columns of ``X`` are reordered
    to match the metadata order.
    
    If the first header in the feature table or metadata table is ``#``, that column is used as the row index
    for the corresponding table and then removed from the data matrix.
    
    Columns in the provisional ``X`` table whose names start with prefixes such as ``p_val``, ``p_adj``, ``umap``,
    ``pca``, ``t-sne``, ``imp``, ``lfc``, or ``vip`` are moved into ``features``.
    
    Prints
    ------
    status : str
        Informational messages about detected separators, inferred mode, removed ``samples``, reordering, and final size.
    """
    used_feat_sep = _sep(features_file, feat_sep)
    data = pd.read_csv(features_file, sep=used_feat_sep)

    # 
    feature_index = None
    if len(data.columns) > 0 and str(data.columns[0]).strip() == '#':
        feature_index = data.iloc[:, 0].astype(str)
        data = data.iloc[:, 1:].copy()
        dups = feature_index[feature_index.duplicated()].unique().tolist()
        if len(dups) > 0:
            print(f"[Rodin] Warning: leading '#' index has {len(dups)} duplicate value(s).")
            if len(dups) <= 10:
                print(f"[Rodin] Duplicate index values: {', '.join(map(str, dups))}")
        print("[Rodin] Feature index set from leading '#' column.")

    # --- Mode detection on cleaned columns (so '#' не ломает авто-режим)
    mode_ = _auto_mode(data, hint=mode)
    print(f"[Rodin] Mode: {repr(mode_)}")

    # --- Split features vs X
    if mode_ == 'ann':
        features = data.iloc[:, :1].copy()
        X = data.iloc[:, 1:].copy()
    else:
        features = data.iloc[:, :2].astype(float).copy()
        X = data.iloc[:, 2:].copy()

    # --- If we captured '#' index, apply it to both features and X (row index)
    if feature_index is not None:
        if len(feature_index) != len(features):
            raise ValueError("[Rodin] Length mismatch: '#' index length does not match number of rows.")
        features.index = feature_index.values
        X.index = feature_index.values

    # --- Move per-feature extra columns from X to features by prefixes (case-insensitive)
    _prefixes = ("p_val", "p_adj", "umap", "pca", "t-sne", "imp", "lfc", "vip")
    x_cols_lower = {c: str(c).lower() for c in X.columns}
    extra_cols = [c for c, lc in x_cols_lower.items() if any(lc.startswith(p) for p in _prefixes)]
    if extra_cols:
        features = pd.concat([features, X.loc[:, extra_cols]], axis=1)
        X = X.drop(columns=extra_cols)

    # --- Metadata
    if meta_file is None:
        samples = pd.DataFrame({'sample_id': X.columns.astype(str)})
        used_meta_sep = None
    else:
        used_meta_sep = _sep(meta_file, meta_sep)
        samples = pd.read_csv(meta_file, sep=used_meta_sep)

        # --- Handle '#' as index in metadata (same as features)
        if len(samples.columns) > 0 and str(samples.columns[0]).strip() == '#':
            meta_index = samples.iloc[:, 0].astype(str)
            samples = samples.iloc[:, 1:].copy()
            samples.index = meta_index
            print("[Rodin] Metadata index set from leading '#' column.")


    print(f"[Rodin] Features sep: {repr(used_feat_sep)}; Metadata sep: {repr(used_meta_sep)}")

    # --- Ensure string IDs
    X.columns = X.columns.astype(str)
    id_col = samples.columns[0]
    samples[id_col] = samples[id_col].astype(str)

    # --- Handle duplicate IDs in metadata (keep first)
    dup_meta = samples[id_col][samples[id_col].duplicated()].unique().tolist()
    if len(dup_meta) > 0:
        print(f"[Rodin] Warning: {len(dup_meta)} duplicate sample IDs in metadata; keeping first occurrence.")
        if len(dup_meta) <= 10:
            print(f"[Rodin] Duplicates: {', '.join(map(str, dup_meta))}")
        samples = samples.drop_duplicates(subset=id_col, keep='first')

    x_ids = list(X.columns)
    meta_ids = list(samples[id_col])

    # --- What will be removed
    removed_from_X = [sid for sid in x_ids if sid not in meta_ids]
    removed_from_meta = [sid for sid in meta_ids if sid not in x_ids]

    if removed_from_X:
        print(f"[Rodin] Samples in X but missing in metadata: {len(removed_from_X)} removed.")
        if len(removed_from_X) <= 10:
            print(f"[Rodin] Removed from X: {', '.join(removed_from_X)}")

    if removed_from_meta:
        print(f"[Rodin] Samples in metadata but missing in X: {len(removed_from_meta)} removed.")
        if len(removed_from_meta) <= 10:
            print(f"[Rodin] Removed from metadata: {', '.join(removed_from_meta)}")

    # --- Keep intersection, ordered by metadata
    keep_ids = [sid for sid in meta_ids if sid in x_ids]
    if len(keep_ids) == 0:
        raise ValueError("[Rodin] No overlapping sample IDs between features (X) and metadata.")

    # Track prior aligned order to report if changed
    prev_aligned_order = [sid for sid in x_ids if sid in meta_ids]

    # --- Apply filtering & reordering
    X = X.loc[:, keep_ids]
    samples = (samples[samples[id_col].isin(keep_ids)]
                     .set_index(id_col)
                     .loc[keep_ids]
                     .reset_index())

    if prev_aligned_order != keep_ids:
        print("[Rodin] Column order changed to the order of metadata.")

    print(f"[Rodin] Final: {X.shape[1]} samples, {X.shape[0]} features")
    obj = Rodin_Class(X=X, features=features, samples=samples)
    obj.uns['mode'] = mode_
    obj.uns['file_type'] = 'metabolomics' if mode_ == 'mzrt' else 'other'
    return obj




def import_object(path):
    """
    Load a pickled ``Rodin_Class`` object from disk.
    
    Parameters
    ----------
    path : str or path-like
        Path to the pickle file.
    
    Returns
    -------
    Rodin_Class
        Deserialized Rodin object loaded from ``path``.
    """
    
    with open(path, 'rb') as file:
        loaded_obj = pickle.load(file)

    return loaded_obj


