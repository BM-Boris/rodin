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
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
import pingouin as pg
from tqdm.auto import tqdm
import dash_bio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pickle
from .mummichog.functional_analysis import *
from io import StringIO
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
sns.set_theme()

class Rodin_Class:
    """
    A metabolomics data analysis class, supporting data manipulation, statistical analysis, 
    dimensionality reduction, and visualization. Primarily designed for metabolomics research.

    Attributes:
    ----------
    X : pd.DataFrame
        Primary data matrix with metabolite intensities across samples (#features × #samples).
    samples : pd.DataFrame
        Annotations for each sample, including sample ID, experimental conditions, and other metadata.
    features : pd.DataFrame
        Annotations and statistical test results for each metabolite feature, including metabolite ID, 
        mass-to-charge ratio (m/z), retention time, and results from statistical analyses.
    uns : dict
        Dictionary for additional unstructured annotations or metadata related to the experiment.
    dr : dict
        Stored results from various dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP).

    This class provides a comprehensive toolkit for processing and analyzing metabolomics data, 
    from initial data handling to in-depth statistical evaluation.
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
        return self._X

    @X.setter
    def X(self, value):
        self._X = self._validate_dataframe(value, "X")
        self._validate_sample_ids()
        self._validate_feature_dimensions()

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = self._validate_dataframe(value, "samples")
        self._validate_sample_ids()

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = self._validate_dataframe(value, "features")
        self._validate_feature_dimensions()

    def _validate_matrix_or_dataframe(self, obj, name):
        if obj is not None and not isinstance(obj, (np.ndarray, sparse.spmatrix, pd.DataFrame)):
            raise TypeError(f"{name} should be a numpy array, a scipy sparse matrix, or a pandas DataFrame.")
        return obj

    def _validate_dataframe(self, dataframe, name):
        if dataframe is not None and not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"{name} should be a pandas DataFrame.")
        return dataframe

    def _validate_sample_ids(self):
        if self.X is not None and self.samples is not None:
            x_columns_set = set(self.X.columns)
            sample_ids_set = set(self.samples.iloc[:, 0])
            if x_columns_set != sample_ids_set:
                raise ValueError("Column names in X do not match sample IDs in samples")
            if list(self.X.columns) != list(self.samples.iloc[:, 0]):
                raise ValueError("The order of sample IDs must be the same in X and in samples")

                
                
    def _validate_feature_dimensions(self):
        if self._X is not None and self._features is not None:
            if self._X.shape[0] != self._features.shape[0]:
                raise ValueError(f"Number of columns in X ({self._X.shape[0]}) does not match number of rows ({self._features.shape[0]}) in features")



    def __repr__(self):
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
        Enables advanced slicing of the Rodin object using DataFrame slices or masks.
    
        Parameters:
        - idx (Union[pd.DataFrame, pd.Series, slice]): The slicing index. Can be a DataFrame for boolean indexing, 
          a Series for label-based indexing, or a slice object.
    
        Raises:
        - ValueError: If the provided index is invalid or if the slicing results in mismatched dimensions.
    
        Returns:
        - Rodin_Class: A new Rodin_Class object with the sliced data.
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
        Perform PCA on the X matrix and store the results in the dr attribute.
    
        Parameters:
        - n_components (int, optional): Number of principal components to compute. Defaults to 2.
        - custom_name (str, optional): Key under which the PCA result is stored in the dr dictionary. Defaults to 'pca'.
    
        Raises:
        - ValueError: If X is None or not a DataFrame.
    
        Returns:
        - None: The method updates the dr attribute of the object.
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
        Perform UMAP on the X matrix or on PCA components and store the results.
    
        Parameters:
        - n_components (int): Number of dimensions for UMAP projection.
        - use_pca (bool): Whether to use PCA components as input for UMAP. Defaults to False.
        - pca_name (str): The key under which PCA results are stored, used if use_pca is True.
        - custom_name (str): Key under which the UMAP result is stored in the dr dictionary.
        - umap_params (dict): Additional parameters for UMAP.
    
        Raises:
        - ValueError: If PCA results are required but not found, or if X is None.
    
        Returns:
        - None: The method updates the dr attribute of the object with UMAP results.
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
        Perform t-SNE on the X matrix or on PCA components and store the results.
    
        Parameters:
        - n_components (int): Number of dimensions for t-SNE projection.
        - use_pca (bool): Whether to use PCA components as input for t-SNE. Defaults to False.
        - pca_name (str): The key under which PCA results are stored, used if use_pca is True.
        - custom_name (str): Key under which the t-SNE result is stored in the dr dictionary.
        - tsne_params (dict): Additional parameters for t-SNE.
    
        Raises:
        - ValueError: If PCA results are required but not found, or if X is None.
    
        Returns:
        - None: The method updates the dr attribute of the object with t-SNE results.
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
        Plot the results of a dimensionality reduction technique.
    
        Parameters:
        - dr_name (str, optional): Name of the dimensionality reduction result to use for the plot. Defaults to 'umap'.
        - hue (str, optional): Column name in the 'samples' DataFrame for point coloring. Defaults to None.
        - size (str, optional): Column name in the 'samples' DataFrame to adjust point sizes. Defaults to None.
        - markers (str, optional): Column name in the 'samples' DataFrame for point styles. Defaults to None.
        - title (str, optional): Title for the plot. Defaults to an empty string.
        - interactive (bool, optional): Whether to create an interactive Plotly scatterplot or a static Seaborn scatterplot. Defaults to True.
        - scatterplot_params (dict, optional): Additional keyword arguments for seaborn.scatterplot.
    
        Raises:
        - ValueError: If specified dr_name is not found in 'dr' or if hue, size, or markers are not columns in 'samples'.
    
        Returns:
        ax: The matplotlib figure object containing the generated plots.
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

    def transform(self, thresh=0.5, norm='q',scale=True, log=True):
        """
        Transforms the X matrix by filling missing values, applying threshold filtering, normalization, scaling, and log transformation and removes features with constant values across all samples and duplicates.
    
        Parameters:
        - thresh (float, optional): Threshold for missing values. Defaults to 0.5.
        - norm (str, optional): Normalization method, either 'q' for Quantile or 't' for Total Intensity, or None to skip normalization. Defaults to 'q'.
        - scale (bool, optional): Whether to scale rows to unit variance. Defaults to True.
        - log (bool, optional): Whether to apply log transformation. Defaults to True.

        Raises:
        - ValueError: If X is None or norm is not a valid option.
    
        Returns:
        - Rodin_Class: The current object with transformed X matrix.
        """
        
        if self.X is None:
            raise ValueError("The X attribute is empty. Please assign a DataFrame to X before calling transform.")

        # Fill missing values with zeros
        df_features_processed = self.X.fillna(0.0)
        thresh_cols=1
        # Filter based on threshold for missing values (rows and columns)
        row_mask = (df_features_processed == 0).mean(axis=1) <= thresh
        col_mask = (df_features_processed == 0).mean(axis=0) <= thresh_cols
        df_features_processed = df_features_processed.loc[row_mask, col_mask]

        # Normalization
        if norm is not None:
            if norm == 'q':
                # Quantile normalization
                df_sorted = np.sort(df_features_processed, axis=0)
                df_mean = df_sorted.mean(axis=1)
                df_ranked = df_features_processed.rank(method="min").astype(int) - 1
                df_norm = pd.DataFrame(df_mean[df_ranked.values], index=df_features_processed.index,
                                       columns=df_features_processed.columns)
            elif norm == 't':
                # Total intensity normalization
                df_norm = df_features_processed.div(df_features_processed.sum(axis=0), axis=1) * 1e5
            else:
                raise ValueError('Provide a valid normalization method: "q" for Quantile Normalization or "t" for Total Intensity Normalization')
        else:
            df_norm = df_features_processed
            
        if scale:
            row_stds = df_norm.std(axis=1)
            row_stds[row_stds == 0] = 1
            df_norm = df_norm.div(row_stds, axis=0)
        
        # Log transformation
        if log:
            df_norm = np.log2(df_norm + 1)

        # Remove rows with constant values across all samples and duplicates
        df_features_processed = df_norm.loc[df_norm.nunique(axis=1) > 1]
        df_features_processed = df_features_processed.drop_duplicates(ignore_index=False)


        filtered_feature_count = self.X.shape[0] - df_features_processed.shape[0]
        print(f"Number of features filtered: {filtered_feature_count}")
        
        # Use the internal attribute to bypass validation temporarily
        self._X = df_features_processed  # Use the internal attribute to bypass validation temporarily
        self._features = self.features.loc[self._X.index]

        # Reapply validation if needed
        self.X = self._X
        self.features = self._features


        return self

    def oneway_anova(self, column_name):
        """
        Performs a one-way ANOVA test on the data, grouped by the specified column in the samples DataFrame.
    
        Parameters:
        - column_name (str): Column name in the samples DataFrame to group the data for ANOVA.
    
        Raises:
        - ValueError: If X or samples are None or if the column_name is not in samples.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with ANOVA test results.
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
        Performs a t-test between two groups in the data, defined by the specified column in the samples DataFrame.
    
        Parameters:
        - column_name (str): Column name in the samples DataFrame to define the two groups.
    
        Raises:
        - ValueError: If X or samples are None, if the column_name is not in samples, or if there are not exactly two unique classes.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with t-test results.
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


    def pls_da(self, column_name):
        """
        Performs Partial Least Squares Discriminant Analysis (PLS-DA) on the data.
    
        Parameters:
        - column_name (str): Column name in the samples DataFrame to use as the response variable.
    
        Raises:
        - ValueError: If X or samples are None or if the column_name is not in samples.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with VIP scores from PLS-DA.
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
        Performs a two-way ANOVA test on the data, grouped by two specified columns in the samples DataFrame.
    
        Parameters:
        - column_names (List[str]): List of two column names in the samples DataFrame to group the data for ANOVA.
    
        Raises:
        - ValueError: If X or samples are None, if either column name is not in samples, or if there are not exactly two column names provided.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with two-way ANOVA test results.
        """
        
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling twoway_anova.")
    
        if len(column_names) != 2:
            raise ValueError("Please provide exactly two column names for two-way ANOVA.")
        
        for col in column_names:
            if col not in self.samples.columns:
                raise ValueError(f"Column '{col}' not found in samples.")
    
        # Extract class labels for the two factors
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
    
            current_p_vals = pg.anova(data=df_new, dv='Intensity', between=column_names, ss_type=1)['p-unc'][:-1].values
            p_vals_list.append(current_p_vals)
    
        p_vals = np.array(p_vals_list)
        anova_classes = pg.anova(data=df_new, dv='Intensity', between=column_names, ss_type=1)['Source'][:-1].values
    
        # Update the features DataFrame with ANOVA results
        for i in range(len(anova_classes)):
            p_value_col = f"p_value(twa) {anova_classes[i]}"
            p_adj_col = f"p_adj(twa) {anova_classes[i]}"
            self.features[p_value_col] = p_vals.T[i]
            self.features[p_adj_col] = stats.false_discovery_control(ps=p_vals.T[i], method='bh')
    
        return self.features

    def sf_lr(self, target_column, moderator=None, interaction=False,degree=1, **kwargs):
        """
        Performs linear regression for each feature in the dataset against the target column, optionally includinga moderator
        and interaction term. Updates the features DataFrame with regression p-values and adjusted p-values.
    
        Parameters:
        - target_column (str): The name of the column in the 'samples' DataFrame to use as the dependent variable.
        - moderator (str, optional): The name of the moderator variable column in the 'samples' DataFrame. If provided, includes this variable in the regression.
        - interaction (bool, optional): If True and a moderator is provided, includes the interaction term between the feature and moderator in the model.
        - degree (int, optional): The degree of polynomial terms to include in the regression. Defaults to 1.
        - **kwargs: Additional keyword arguments can be passed to the regression method.


        Raises:
        - ValueError: If 'X' or 'samples' are None, if 'target_column' is not in 'samples', or if 'moderator' is specified but not found in 'samples'.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with new columns for regression p-values and adjusted p-values.
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
        Performs logistic regression for each feature in the dataset against the target column, optionally including a moderator
        and interaction term. Updates the features DataFrame with regression p-values and adjusted p-values.
    
        Parameters:
        - target_column (str): The name of the column in the 'samples' DataFrame to use as the dependent variable.
        - moderator (str, optional): The name of the moderator variable column in the 'samples' DataFrame. If provided, includes this variable in the regression.
        - interaction (bool, optional): If True and a moderator is provided, includes the interaction term between the feature and moderator in the model.
        - regu (bool, optional): Enables regularization for the regression model. If True, L1 (Lasso) regularization is enabled and the `alpha` parameter should be specified to control the regularization strength. Default is False.
        - degree (int, optional): The degree of polynomial terms to include in the regression. Defaults to 1.
        - **kwargs: Additional keyword arguments can be passed to the regression method, such as `alpha` to specify the regularization strength.

        Raises:
        - ValueError: If 'X' or 'samples' are None, if 'target_column' is not in 'samples', or if 'moderator' is specified but not found in 'samples'.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with new columns for regression p-values and adjusted p-values.
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

                    

    def rf_class(self, target_column, n_estimators=100, random_state=16,cv=4, **kwargs):
        """
        Trains a Random Forest Classifier using 4-fold cross-validation on the data, returns feature importances,
        and print a classification report.
    
        Parameters:
        - target_column (str): Column name in the 'samples' DataFrame to use as the target variable.
        - n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
        - random_state (int, optional): Random state for reproducibility. Defaults to 16.
        - cv (int, optional): Number of folds for validation. Defaults to 4.
    
        Raises:
        - ValueError: If X or samples are None or if the target_column is not in samples.
    
        Returns:
        - dict: Dictionary containing feature importances and classification report DataFrame.
        """
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling rf_class.")
    
        if target_column not in self.samples.columns:
            raise ValueError(f"Column '{target_column}' not found in samples.")
    
        X = self.X.T
        y = self.samples[target_column]
    
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, **kwargs)
    
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
        

    def rf_regress(self, target_column, n_estimators=100, random_state=16, cv=4, **kwargs):
        """
        Trains a Random Forest Regressor using cross-validation on the data, returns feature importances,
        and print regression metrics.
    
        Parameters:
        - target_column (str): Column name in the 'samples' DataFrame to use as the target variable.
        - n_estimators (int, optional): The number of trees in the forest. Defaults to 100.
        - random_state (int, optional): Random state for reproducibility. Defaults to 16.
        - cv (int, optional): Number of folds for cross-validation. Defaults to 4.
    
        Raises:
        - ValueError: If X or samples are None or if the target_column is not in samples.
    
        Returns:
        - dict: Dictionary containing feature importances and regression metrics.
        """
        if self.X is None or self.samples is None:
            raise ValueError("Both X and samples must be assigned before calling rf_regress.")
    
        if target_column not in self.samples.columns:
            raise ValueError(f"Column '{target_column}' not found in samples.")
    
        X = self.X.T
        y = self.samples[target_column]
    
        clf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, **kwargs)
    
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
        Calculates log fold change for the data grouped by the specified column in the samples DataFrame.
    
        Parameters:
        - column_name (str): Column name in the samples DataFrame to group the data for fold change calculation.
        - reference (str): Value to use as a reference for fold change. Defaults to None.
    
        Raises:
        - ValueError: If X or samples are None, or if the column_name is not in samples.
    
        Returns:
        - pd.DataFrame: The updated features DataFrame with log fold change results.
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
            log_fold_change = np.log2(np.expm1(mean_group_0+1e-9) / np.expm1(mean_current_group+1e-9))
    
            self.features[f'lfc ({unique_classes[0]} vs {unique_classes[idx+1]})'] = log_fold_change
    
            if(len(unique_classes) > 2):
                avg_fc.append(log_fold_change)
                
        if(len(unique_classes) > 2):
            self.features[f'lfc ({unique_classes[0]} vs others)'] = np.mean(avg_fc, axis=0)
            
        return self.features

    def clustergram(self, title="", interactive=True,
                    height=820,width=900,center_values=False,standardize='row',
                    link_method='ward',hidden_labels='row',color_map='RdBu',line_width=1.4,hue=None, **clustergram_params):
        """
        Creates a clustergram (clustered heatmap) of the X matrix.
    
        Parameters:
        - title (str, optional): Title for the plot. Defaults to an empty string.
        - interactive (bool, optional): Whether to create an interactive Dash Bio Clustergram or a static Seaborn clustermap. Defaults to True.
        - height, width (int, optional): Height and width of the plot. Defaults to 820 and 900 respectively.
        - center_values (bool, optional): Whether to center values in the clustergram. Defaults to False.
        - standardize (str, optional): Standardization method, either 'row' or 'column'. Defaults to 'row'.
        - link_method (str, optional): Linkage method for hierarchical clustering. Defaults to 'ward'.
        - hidden_labels (str, optional): Labels to hide in the plot. Defaults to 'row'.
        - color_map (str, optional): Color map for the heatmap. Defaults to 'RdBu'.
        - line_width (float, optional): Line width for the clustergram. Defaults to 1.4.
        - hue (str, optional): Column name in the samples DataFrame used to color the labels. Defaults to None.
        - clustergram_params (dict, optional): Additional keyword arguments for Dash Bio Clustergram or Seaborn clustermap.
    
        Raises:
        - ValueError: If X is None.
    
        Returns:
        - object: Dash Bio Clustergram object if interactive, else Seaborn ClusterGrid object.
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

    def boxplot(self, hue, pathways=None, eids=None, rows=None, significant=0.05, grid_dim=None, figsize=None, title="", zeros=True, cutoff_path=0.05,interactive=True,category_order=None, **boxplot_params):
        """
        Generates box plots for specified pathways, rows, or EIDs with an option to filter by significance.
    
        This function creates separate figures for each pathway, plotting all EIDs associated with that pathway.
        The function can also plot specified rows or rows corresponding to a list of EIDs.
    
        Parameters:
        hue (str): Column name in 'self.samples' to be used for hue in the plots.
        pathways (list of str or str, optional): Specific pathway(s) to include. If specified, plots all EIDs associated with these pathways.
        eids (list, optional): A list of EIDs for which corresponding rows are to be plotted. If specified, only rows with these EIDs and p-values <= 'significant' will be considered.
        rows (list, optional): A list of rows to be plotted. If 'None', rows will be determined based on 'eids' or 'pathway'.
        significant (float, optional): A significance level (p-value threshold) to filter rows based on 'eids'. Default is 0.05.
        grid_dim (tuple, optional): Dimensions for the grid of subplots (rows, columns). By default, it creates a single column of plots.
        figsize (tuple, optional): Size of the figure (width, height). Auto-adjusted based on the number of plots if not provided.
        title (str, optional): Overall title for the plot.
        zeros (bool, optional): Whether to include zeros in the plot. Defaults to True.
        cutoff_path (float, optional): Threshold for filtering pathways based on p-value. Defaults to 0.05.
        interactive (bool, optional): Whether to generate interactive plots using Plotly. Defaults to True.
        category_order (list, optional): A list of classes to change the order of plots.
        **boxplot_params: Additional keyword arguments to be passed to seaborn's violinplot function.
    
        Returns:
        figs (list): A list of matplotlib figure objects containing the generated box plots.
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
                #####    
                if interactive:
                    fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
                    for i, row in enumerate(rows):
                        df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
                        if hue and hue in self.samples.columns:
                            fig.add_trace(go.Box(y=df,x=self.samples[hue].values,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                        else:
                            fig.add_trace(go.Box(y=df,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                            
                        fig.update_yaxes(title_text=f"{row}",row=i//ncols+1, col=i%ncols+1,title_standoff=0)
                    fig.update_annotations(font_size=13)    
                    fig.update_layout(title_text=f"{pathway}",showlegend=False,margin=dict(r=50))
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
                                sns.boxplot(y=df, x=self.samples[hue].values, ax=ax, **boxplot_params)
                            else:
                                sns.boxplot(y=df, ax=ax, **boxplot_params)
        
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
            #####    
            if interactive:
                fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
                for i, row in enumerate(rows):
                    df = self.X.loc[row] if zeros else self.X.loc[row].replace(0, np.nan)
                    if hue and hue in self.samples.columns:
                        fig.add_trace(go.Box(y=df,x=self.samples[hue].values,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                    else:
                        fig.add_trace(go.Box(y=df,**boxplot_params), row=i//ncols+1, col=i%ncols+1)
                        
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
                            sns.boxplot(y=df, x=self.samples[hue].values, ax=ax, **boxplot_params)
                        else:
                            sns.boxplot(y=df, ax=ax, **boxplot_params)
        
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





    def violinplot(self, hue, pathways=None, eids=None, rows=None, significant=0.05, grid_dim=None, figsize=None, title="", zeros=True, cutoff_path=0.05,interactive=True,category_order=None, **violinplot_params):
        """
        Generates violin plots for specified pathways, rows, or EIDs with an option to filter by significance.
    
        This function creates separate figures for each pathway, plotting all EIDs associated with that pathway.
        The function can also plot specified rows or rows corresponding to a list of EIDs.
    
        Parameters:
        hue (str): Column name in 'self.samples' to be used for hue in the plots.
        pathways (list of str or str, optional): Specific pathway(s) to include. If specified, plots all EIDs associated with these pathways.
        eids (list, optional): A list of EIDs for which corresponding rows are to be plotted. If specified, only rows with these EIDs and p-values <= 'significant' will be considered.
        rows (list, optional): A list of rows to be plotted. If 'None', rows will be determined based on 'eids' or 'pathway'.
        significant (float, optional): A significance level (p-value threshold) to filter rows based on 'eids'. Default is 0.05.
        grid_dim (tuple, optional): Dimensions for the grid of subplots (rows, columns). By default, it creates a single column of plots.
        figsize (tuple, optional): Size of the figure (width, height). Auto-adjusted based on the number of plots if not provided.
        title (str, optional): Overall title for the plot.
        zeros (bool, optional): Whether to include zeros in the plot. Defaults to True.
        cutoff_path (float, optional): Threshold for filtering pathways based on p-value. Defaults to 0.05.
        interactive (bool, optional): Whether to generate interactive plots using Plotly. Defaults to True.
        category_order (list, optional): A list of classes to change the order of plots.
        **violinplot_params: Additional keyword arguments to be passed to seaborn's violinplot function.
    
        Returns:
        figs (list): A list of matplotlib figure objects containing the generated violin plots.
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
                fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
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
        Generates regression plots for specified pathways, rows, or EIDs with an option to filter by significance.
    
        This function creates separate figures for each pathway, plotting all EIDs associated with that pathway.
        The function can also plot specified rows or rows corresponding to a list of EIDs.
    
        Parameters:
        column (str): Column name in 'self.samples' to be used for x in the plots.
        pathways (list of str or str, optional): Specific pathway(s) to include. If specified, plots all EIDs associated with these pathways.
        eids (list, optional): A list of EIDs for which corresponding rows are to be plotted. If specified, only rows with these EIDs and p-values <= 'significant' will be considered.
        rows (list, optional): A list of rows to be plotted. If 'None', rows will be determined based on 'eids' or 'pathway'.
        significant (float, optional): A significance level (p-value threshold) to filter rows based on 'eids'. Default is 0.05.
        grid_dim (tuple, optional): Dimensions for the grid of subplots (rows, columns). By default, it creates a single column of plots.
        figsize (tuple, optional): Size of the figure (width, height). Auto-adjusted based on the number of plots if not provided.
        title (str, optional): Overall title for the plot.
        zeros (bool, optional): Whether to include zeros in the plot. Defaults to True.
        x_limit (tuple, optional): Limits for the x-axis. Defaults to None.
        cutoff_path (float, optional): Threshold for filtering pathways based on p-value. Defaults to 0.05.
        interactive (bool, optional): Whether to generate interactive plots using Plotly. Defaults to True.
        trendline (str,optional): Use in interactive mode. One of 'ols', 'lowess', 'rolling', 'expanding' or 'ewm'. If ols, an Ordinary Least Squares regression line will be drawn for each discrete-color/symbol group. If 'lowess', a Locally Weighted Scatterplot Smoothing line will be drawn for each discrete-color/symbol group. If 'rolling', a Rolling (e.g. rolling average, rolling median) line will be drawn for each discrete-color/symbol group. If 'expanding', an Expanding (e.g. expanding average, expanding sum)line will be drawn for each discrete-color/symbol group. If 'ewm', an Exponentially Weighted Moment (e.g. exponentially-weighted movingaverage) line will be drawn for each discrete-color/symbol group.
        
        **regplot_params: Additional keyword arguments to be passed to seaborn's regplot function.
    
        Returns:
        figs (list): A list of matplotlib figure objects containing the generated regression plots.
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
                fig = make_subplots(rows=int(nrows), cols=int(ncols), subplot_titles=[f"EID: {self.uns['compounds'][self.uns['compounds']['input_row'] == row]['EID'].values}" for row in rows])
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


    def save(self,path):
        """
        Saves the current Rodin_Class object to a file using pickle.
    
        Parameters:
        - path (str): The file path to save the object to.
    
        Returns:
        - None: The object is saved to a file and the method returns nothing.
        """
        
        with open(path, 'wb') as file:
            pickle.dump(self, file)


    def analyze_pathways(self,pvals,stats,network='human_mfn',mode='f_positive',instrument='unspecified',permutation=100,
                         force_primary_ion=True,cutoff=0,modeling=None,pws_name='pathways',cmp_name='compounds'):
        """
        Analyzes metabolic pathways by integrating user data with a metabolic network model. 
        The method performs pathway analysis based on provided statistics and p-values, 
        using various options for mass spectrometry data interpretation and pathway modeling.
    
        Parameters:
        - pvals (str): Column name in 'features' DataFrame representing p-values.
        - stats (str): Column name in 'features' DataFrame representing statistic scores.
        - network (str, optional): The metabolic network model to use (e.g., 'human_mfn', 'worm'). Defaults to 'human_mfn'.
        - mode (str, optional): Mass spectrometry analysis mode (e.g., 'positive', 'negative'). Defaults to 'f_positive'.
        - instrument (str, optional): Instrument accuracy specification (ppm). Defaults to 'unspecified'.
        - permutation (int, optional): Number of permutations for null distribution estimation. Defaults to 100.
        - force_primary_ion (bool, optional): Whether to enforce primary ion presence in metabolite prediction. Defaults to True.
        - cutoff (float, optional): Cutoff p-value for selecting significant features.
        - modeling (str, optional): Method for modeling permutation data (e.g., 'gamma'). Defaults to None.
        - pws_name (str, optional): Key to store pathway analysis results in 'uns' dictionary. Defaults to 'pathways'.
        - cmp_name (str, optional): Key to store compound analysis results in 'uns' dictionary. Defaults to 'compounds'.
    
        Returns:
        - pd.DataFrame: A DataFrame with pathway analysis results, also stored in the 'uns' dictionary under 'pws_name'.
    
        The method constructs a metabolic network from the specified model, matches model data with user data, 
        and performs pathway enrichment analysis. It supports different analytical modes and custom metabolic models.
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
        Returns a DataFrame with pathways and their associated compounds based on the given thresholds.
        
        Parameters:
        - cutoff_path (float, optional): Threshold for filtering pathways based on p-value. Defaults to 0.05.
        - cutoff_eids (float, optional): Threshold for filtering compounds based on p-value. Defaults to 0.05.
        - paths (list of str or str, optional): List of specific pathways to include. Defaults to None, which includes all pathways that pass the cutoff.
        
        Returns:
        - DataFrame: A DataFrame with pathways and their associated compounds.
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

                         
        



def create_object_csv(file_path_features, file_path_classes,feat_sep='\t',class_sep='\t', feat_stat='mzrt'):
    """
    Creates a Rodin_Class object from CSV files for features and classes.

    Parameters:
    - file_path_features (str): File path to the CSV file containing features.
    - file_path_classes (str): File path to the CSV file containing class information.
    - feat_sep (str, optional): Separator used in the features CSV file. Defaults to '\t'.
    - class_sep (str, optional): Separator used in the classes CSV file. Defaults to '\t'.
    - feat_stat (str, optional): Feature status mode indicating the layout of the feature table. Use 'mzrt' if the first two columns are mass-to-charge ratio (mz) and retention time (rt), or 'ann' if one column is dedicated to annotations.
   
    Returns:
    - Rodin_Class: A new instance of Rodin_Class populated with data from the provided CSV files.
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


def import_object(path):
    """
    Loads a Rodin_Class object from a pickle file.

    Parameters:
    - path (str): The file path from which to load the object.

    Returns:
    - Rodin_Class: The loaded Rodin_Class object.
    """
    
    with open(path, 'rb') as file:
        loaded_obj = pickle.load(file)

    return loaded_obj





    

    


    


   
