# CountsOutlierDetector

CountsOutlierDetector is usable, but still in progress. It is designed to be a highly-interpretable outlier detector. 

## Overview
CountsOutlierDetector is an interpretable, intuitive, efficient outlier detector intended for tabular data. It is a categorical-based detector; all numeric data will be automatically binned. It is designed to provide clear explanations of the rows flagged as outliers and their specific scores. 

Despite the common business need to understand and assess outliers within tabular datasets, most outlier detectors are very much blackboxes, often even much more so than classifiers and regressors. Although the algorithms employed by many detectors are themselves interpretable, the individual predictions are generally not. For example, particularly with high-dimensional data, such standard detectors as Isolation Forest (IF), Local Outlier Factor (LOF), and kth Nearest Neighbors (kthNN), produce scores that may be difficult to assess. Futher, it's particularly difficult to assess the counterfactuals: what would need to change for these rows to not be flagged. Also very useful, but often impossible to assess with many outlier detectors, is the outlierness of the rows not flagged by the detectors. That is, when presented with the results of a detector on a dataset, it would be helpful to know how the flagged rows compare to the rows not flagged: how different were they, and what rows were almost, but not quite, flagged, which is key to understanding the effect of any thresholds or other decisions used. CountsOutlierDetector attempts to solve these issues by providing a highly transparent, non-stocastic outlier detection system. 

CountsOutlierDetector is based on the intuition that all outliers are essentially data instances with either unusual single values, or unusual combinations of values. It works by simply first examing each column indivually and identifying all values that are unusual with respect to their columns. These are then known as *1d outliers*, that is: outliers based on considering a single dimension at a time. It then examines each pair of columns, identifying the rows with pairs of unusual values. For example, having fur may be common, as well as laying eggs, but the combination is rare. Or, with numeric columns, an age of 1 yr may be common and a height of 6' as well, but the combination rare, most likely flagging an error. These are known as *2d outliers*. The detector then considers sets of 3 columns, and so on. 

At each stage, the algorithm looks for instances that are unusual specifically considering that dimensionality, excluding combinations already flagged in lower-dimensional spaces. This helps keep the explanations interpretable, but also avoids double counting. For example, in a table with columns: A,B,C,D,E, there may be a rare value in column A, say A<sub>15</sub>. Every combination of values that includes A<sub>15</sub>, for example A<sub>15</sub>,B<sub>1</sub>, as well as A<sub>15</sub>,B<sub>2</sub> and so on, would also be a rare combination, but are not counted, as the value A<sub>15</sub> has already been identified. Similarly any rare pairs of values in 2d space are excluded from consideration in higher dimensional spaces. However, a row may be flagged numerous times. For example, a row may have an unsual pair of values in columns A and B, as well as an unusual triple of values in columns C,D,E. 

CountsOutlierDetector takes its name from the fact it examines the exact count of each value with categorical columns, and each range of values with numeric columns. It considers the outlierness of a combination of values considering their marginal distributions as well as the expected joint distribution given the cardinalities of the relevant columns, so provides sensible outlier scores, but based on the actual count of each value or combination of values. This ensures the outlier scores provided are clear, have full context, and allow straight-forward comparisons between rows. The system is limited only by its tendency to consider only low dimensionalities. 

## Examining Greater Numbers of Dimensions
The detector may examine up to a specified number of dimensions, but our testing indicates it is rarely necessary to go beyond 3 or 4 dimensions. This ensures the outlier detection process is quite tractable, but also ensures that the outliers identified are interpretable. 

Most detectors, in some sense, attempt to find the same rare values and combinations of values as CountsOutlierDetector explicitely searches for, though use algorithms optimized for a variety of concerns, such as speed and scalability. Doing so, they more or less approximate the process here, though not necessarily examining all combinations; stocastic processes are often used, which lead to faster execution times and probably approximately correct solutions, but can miss some outliers. Many do, however, go in a sense beyond what CountsOutlierDector does, and check for higher-dimensional outliers, which this process may miss. For example, some detectors may detect instances that are effectively, for example, 8d-outliers (instances that are not unusual in any 1d, 2d,...7d sense, but are unusual if 8 dimensions are considered). By default, CountsOutlierDectors does not examine this many dimensions and would likely miss this instance. We argue that this is acceptable in most cases, as it's a favourable trade-off for more complete (and as such, less biased) examination of lower-dimensional spaces and more stable, as well as comprehensible explanations. 

Further, there is a strong argument that outliers in lower dimensions are most often more relevant than those in higher dimensions. For example, a row that is an unusual combination of, say, 10 values (but not of any 1, 2,...,9 values) is likely to be less relevant than a row with a single unusual value, or single unusual pair, or unusual triple of values. In most cases, when considering many columns, the number of rows in any given combination is very low in any case, so often no rows can be reliably flagged as outliers in any case. This is in distinction to considering, for example, two or three dimensions, where the number of rows expected in each combination, depending on the cardinality of the column in the case of categorical columns, can be reasonably high. 

As well, experiments using CountsOutlierDetector have found, usually an acceptable, useful number of rows are flagged in lower dimensions, obviating the need to search higher dimesions, though the code does allow examining higher dimensions where desired. But, examining these spaces is exponantially slower, and identifies, where it identifies any rows, rows that are (arguably) less relevant, statistically less significant, and less interpretable. Testing has shown that examing beyond 4 dimensions identifies usually few outliers, which makes intuitive sense as having a 5d outlier, while possible, requires having a row that is unusual in five dimensions, but not in any 1d, 2d, 3d or 4d subspace. 

We suspect the intractability of examining higher dimensional spaces in an exhaustive manner has dissuaded researchers from similar approaches, but we hope to demonstrate that examining up to, depending on the dataset and parameters used, approximately 3 to 5 dimensions is quite sufficient to identify the most outlierish instances. This approach provides a useful midpoint between detectors considering only single columns in isolation (ignoring all feature interactions) such as HBOS and Entropy-based methods on the one hand, and methods that consider all columns but tend to produce inscrutable results, such as IF, LOF, and kthNN. 

## Binning
A known issue with the use of categorical-based outlier detection systems, such as CountsOutlierDetector, with numeric data is they require binning the data, and the results can be sensitive to the binning system used, both the number of bins and the placement of the split points between bins. This is valid, but we believe acceptable, as the majority of realworld data is mixed, containing both categorical and numeric columns, and binning the numeric columns is a sensible, as well as interpretable, approach. We use, by default, seven bins (which corresponds to a common cardinality of categorical columns in the openml.org dataset) and equal-width binning. Equal-count binning is not advised, as it precluded identify 1d outliers in the binned columns. 

Binning numeric data does introduce a small number of decisions, but also eliminates the distance metrics required by numeric-based outlier detectors, and the need to numerically encode categorical columns. 

## Key Qualities of CountsOutlierDetector
This detector has the advantages of: 

1) It is able to provide as clear as possible explanations. Explanations are based on a single column if possible, two columns if necessary, three if necessary, and so on. That is, to explain why a row is scored as it is, an examplanation may be given using only as many columns as necessary to explain its score.  

2) It is able to provide full statistics about each space, which allows it to provide full context of the outlierness of each row. 

This project in an attempt to demonstrate, and provide a working example, that outlier detection can be completely transparent, and though optimized outlier detectors are also very useful, there is a clear utility in interpretable outlier detection. 

# Example

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
det = CountsOutlierDetector()
flagged_rows_df, row_explanations, output_msg, run_summary_df = det.predict(X)
```

The 4 returned pandas dataframes and strings provide information about what rows where flagged and why, as well as summary statistics about the dataset's outliers as a whole. 

# Example Files
[Example Counts Outlier Detector](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/Examples_Counts_Outlier_Detector.ipynb) is a simple notebook providing some examples of use of the detector.

[Tests Counts Outlier Detector](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/Test_CountsOutlierDetector.py) is a python file that tests the outlier detector over a large number of random datasets (100 by default). This uses the [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) tool to aid with collecting and filtering datasets from openml.org. Some results from the execution of this are included below.

# Statistics 

### Percents of Datasets Checked at Each Dimension
We ran CountsOutlierDetector on 100 random datasets collected from openml.org, allowing it to run on up to six dimensions, but limiting the estimated number of value combinations (given the number of columns considered and average cardinality of each column) to 100 million. This excluded most datasets from considering 5d and 6d outliers, and a small number of datasets from examining even 3d outliers.

<img src="https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/Results/percent_datasets_checked_by_dim_bar.png" alt="drawing" width="650"/>

13 out of the 100 did, given these settings, examine datasets for up to 6d outliers. 

### Counts by Dimension 
<img src="https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/Results/counts_by_dim_bar.png" alt="drawing" width="650"/>

This is based on testing 100 random datasets for outliers, up to 6d outliers. This indicates the % of rows flagged as each type of outlier. For most datasets, the 3d tests tend to flag the most rows, with a sharp drop off after that. In this case almost none of the 100 datasets flagged any 6d outliers, though the detector was set to avoid excessive calculations, so datasets skipped the tests for higher dimensions (see plot below for the number of datasets tested up to each dimensionality). Note: many rows were flagged as 1d/2d/3d/4d or 5d outliers multiple times, and so, to some extent, these bars count the same rows.

| Dimension | Avg % Rows Flagged |
| ------- | ---------- |
| 1 | 7.64 |
| 2 |  12.16 |
| 3 | 15.90 |
| 4 | 5.24 |
| 5 | 0.11 |
| 6 | 0.15 |

Even where 5d and 6d spaces are explored, they tend to find very few outliers. 

### Cumulative Counts by Dimension
<img src="https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/Results/counts_up_to_dim_bar.png" alt="drawing" width="650"/>

This is a cumulative count of the same information, indicating what percent of rows (averaged over 100 datasets) are flagged as outliers when examining up to each number of dimensions. This indicates that even at 1d, 2d, or 3d, it is often the case that more rows have been flagged than can be realistically investigated in any case, and while checking higher dimensions, is slower and less interpretable, it will likely discover few additional outliers on top of an already-sufficient result set. However, exceptions do occur.

