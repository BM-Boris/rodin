from setuptools import setup, find_packages

setup(
    name='rodin',
    version='0.1.0',
    packages=find_packages(),
    description='A comprehensive toolkit for processing and analyzing metabolomics data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Boris Minasenko',
    author_email='boris.minasenko@emory.edu',
    url='https://github.com/BM-Boris/rodin',
    install_requires=[
        'numpy>=1.21.4',  
        'pandas>=1.3.4',  
        'scipy>=1.7.3',  
        'scikit-learn>=1.0',  
        'umap-learn>=0.5.1',  
        'matplotlib>=3.5.0',  
        'seaborn>=0.11.2', 
        'statsmodels>=0.13.0',  
        'pingouin>=0.4.0', 
        'tqdm>=4.62.3',  
        'dash-bio>=0.8.0',  
        'dash==2.7.0',
        'pickle-mixin>=1.0.2',
        'networkx>=2.6',
        'fastcluster',
        
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
