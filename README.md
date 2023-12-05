## **Rodin: Metabolomics Data Analysis Toolkit**

_Rodin_ is a Python library specifically designed for the comprehensive processing and analysis of metabolomics data. It is a class-methods based toolkit, facilitating a range of tasks from basic data manipulation to advanced statistical evaluations, visualization, and metabolic pathway analysis.

### **Features**

- **Efficient Data Handling**: Streamlined manipulation and transformation of metabolomics data.
- **Robust Statistical Analysis**: Includes ANOVA, t-tests, and more.
- **Advanced Dimensionality Reduction**: Techniques like PCA, t-SNE, UMAP.
- **Interactive Data Visualization**: Tools for effective data visualization.
- **Pathway Analysis**: Features for metabolic pathway analysis.

### **Installation**

We recommend installing Rodin in a separate Conda environment for effective dependency management.

#### Prerequisites

- Python (3.11 or higher)
- Conda (Anaconda or Miniconda)

#### Setting Up a Conda Environment

Create and activate a new Conda environment:

```bash
conda create -n rodin_env python=3.11
conda activate rodin_env
```
#### Install Rodin

Install Rodin directly from GitHub:
```bash
pip install git+https://github.com/BM-Boris/rodin.git
```

#### Basic Example

Here's a basic example demonstrating the usage of Rodin for data analysis. Comprehensive Jupyter notebook guides can be found in the 'guides' folder
```python
import rodin

# Assume 'data.csv' and 'classes.csv' are your datasets
features_path = 'path/to/data.csv'
classes_path = 'path/to/classes.csv'

# Creating an instance of Rodin_Class
rodin_instance = rodin.create_object_csv(features_path, classes_path)

# Transform the data (imputation, normalization, and log-transformation steps)
rodin_instance.transform()

# Run t-test comparing two groups based on 'age'
rodin_instance.ttest('age')

# Perform PCA with 2 principal components
rodin_instance.run_pca(n_components=2)

# Plotting the PCA results
# 'sample_type' column in the 'samples' DataFrame is used for coloring the points
rodin_instance.plot(dr_name='pca', hue='age', title='PCA Plot')

# Pathway analysis 
rodin_instance.analyze_pathways(pvals='p_value', stats='statistic')
# Replace 'p_value' and 'statistic' with the actual column names in your 'features' DataFrame(rodin_instance.features)
```

#### _Contact_
For questions, suggestions, or feedback, please contact boris.minasenko@emory.edu

