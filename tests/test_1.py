import pytest
from rodin import Rodin_Class 
import pandas as pd
import numpy as np

@pytest.fixture
def rodin_instance():
    np.random.seed(0)  # For consistent test results
    data = pd.DataFrame(np.random.rand(1000, 15))  # 1000 features, 15 samples
    samples = pd.DataFrame({
        'ID': range(15),
        'Class': np.random.choice(['A', 'B', 'C'], size=15)
    })
    features = pd.DataFrame({
        'mz': np.random.rand(1000),
        'rt': np.random.rand(1000)
    })
    return Rodin_Class(X=data, samples=samples, features=features)

def test_init(rodin_instance):
    assert isinstance(rodin_instance, Rodin_Class)
    assert rodin_instance.X.shape == (1000, 15)
    assert 'Class' in rodin_instance.samples.columns
    assert all(col in rodin_instance.features.columns for col in ['mz', 'rt'])


def test_validate_dataframe(rodin_instance):
    with pytest.raises(TypeError):
        rodin_instance._validate_dataframe([1, 2, 3], 'test')

def test_run_pca(rodin_instance):
    rodin_instance.run_pca(n_components=2)
    assert 'pca' in rodin_instance.dr
    assert rodin_instance.dr['pca'].shape == (15, 2)  # Check dimensions based on your PCA implementation

def test_run_umap(rodin_instance):
    rodin_instance.run_umap(n_components=2)
    assert 'umap' in rodin_instance.dr
    assert rodin_instance.dr['umap'].shape == (15, 2)  # Adjust based on UMAP implementation

def test_run_umap(rodin_instance):
    rodin_instance.run_umap(n_components=2)
    assert 'umap' in rodin_instance.dr
    assert rodin_instance.dr['umap'].shape == (15, 2)  # Adjust based on UMAP implementation

def test_run_tsne(rodin_instance):
    rodin_instance.run_tsne(n_components=2,perplexity=12)
    assert 't-sne' in rodin_instance.dr
    assert rodin_instance.dr['t-sne'].shape == (15, 2)  # Assuming t-SNE on transposed X matrix


def test_oneway_anova(rodin_instance):
    # Assuming 'Class' is a column in your samples DataFrame
    anova_results = rodin_instance.oneway_anova('Class')
    assert 'p_value(owa) Class' in anova_results.columns
    # Further assertions can be added based on expected ANOVA output
def test_getitem(rodin_instance):
    # Test slicing functionality
    sliced = rodin_instance[rodin_instance.X.iloc[0:10, 0:5]]  # Adjust slice as per your implementation
    assert isinstance(sliced, Rodin_Class)
    assert sliced.X.shape == (10, 5)
    assert sliced.samples.shape[0] == 5
    assert sliced.features.shape[0] == 10

def test_plot(rodin_instance, mocker):
    # Mock the plotting function to ensure it's called correctly
    mocker.patch('matplotlib.pyplot.show')

    # Assuming 'Class' is a valid column in samples for coloring
    rodin_instance.run_umap(n_components=2)  # Need to run a DR method first
    rodin_instance.plot(dr_name='umap', hue='Class')


def test_fold_change(rodin_instance):
    # Assuming 'Class' is a valid column in samples for fold change calculation
    results = rodin_instance.fold_change('Class')
    assert isinstance(results, pd.DataFrame)
    # Additional assertions based on expected behavior of fold_change

def test_clustergram(rodin_instance, mocker):
    # Mock the plotting function
    mocker.patch('matplotlib.pyplot.show')
    rodin_instance.clustergram()
    




