# CountsOutlierDetector

CountsOutlierDetector is a highly-interpretable outlier detector. 

## Overview
CountsOutlierDetector is an interpretable, intuitive, efficient outlier detector intended for tabular data. It is designed to provide clear explanations of the rows flagged as outliers and of their specific scores. 

Despite the common business need to understand and assess outliers within tabular datasets, most outlier detectors are very much blackboxes, often even more so than classifiers and regressors. Although the algorithms employed by many detectors are themselves interpretable, the individual predictions are generally not. For example, particularly with high-dimensional data, such standard detectors as Isolation Forest (IF), Local Outlier Factor (LOF), and kth Nearest Neighbors (kthNN), have algorithms that are straight-forwared to understand, but produce scores that may be difficult to assess. Futher, it's particularly difficult to assess the counterfactuals: what would need to change for these rows to not be flagged or to be scored lower. 

In cases where the outlierness of an instance is obvious on inspection, this may be a non-issue, but in other cases it can be a serious drawback. In many environments it's necessary to follow up on the outliers identified, and this is particularly difficult where it's not understood why the instances were scored as they were. Interpretable scores fascilitate further examination of the flagged outliers, and make it possible to examine the rows considered very normal for comparison, and those not quite flagged. This allows, for example, tuning of the detector to flag these as well if necessary, but also simple comparison to those that were flagged. CountsOutlierDetector attempts to solve these issues by providing a highly transparent, non-stocastic outlier detection system. 

CountsOutlierDetector is based on the intuition that all outliers are essentially data instances with either unusual single values, or unusual combinations of values. It works by first examing each column indivually and identifying all values that are unusual with respect to their columns. These are then known as *1d outliers*, that is: outliers based on considering a single dimension at a time. It then examines each pair of columns, identifying the rows with pairs of unusual values within each pair of columns. For example, having fur may be common, as well as laying eggs, but the combination is rare. Or, with numeric columns, an age of 1 yr may be common and a height of 6' as well, but the combination rare, most likely flagging an error. These are known as *2d outliers*. The detector then considers sets of 3 columns, sets of 4 columns, and so on. 

## Algorithm
At each stage, the algorithm looks for instances that are unusual specifically considering the current dimensionality (looping from 1, 2, 3, and up), excluding values or combinations already flagged in lower-dimensional spaces. This helps keep the explanations interpretable, but also avoids double counting. For example, in a table with columns: A,B,C,D,E, there may be a rare value in column A, say the 15th unique value in A, A<sub>15</sub>. Any rows containing value A<sub>15</sub> in Column A will then be flagged as a 1d outlier. 

Having done that, we do not wish to also flag combinations in higher dimensions that include A<sub>15</sub> -- these are already flagged. Every combination of values that includes A<sub>15</sub>, for example A<sub>15</sub>,B<sub>1</sub> (any rows with A<sub>15</sub> in column A and B<sub>1</sub> in column B), as well as A<sub>15</sub>,B<sub>2</sub>, A<sub>15</sub>,B<sub>3</sub> and so on, would also be a rare combination (by virtue of including A<sub>15</sub>), but are not counted, as the value A<sub>15</sub> has already been identified. That is, as A<sub>15</sub> is rare, A<sub>15</sub> combined with any other values will also be rare, but flagging these would be redundant. 

Similarly any rare pairs of values in 2d space are excluded from consideration in 3- and higher-dimensional spaces; any rare triples of values in 3d space will be excluded from 4- and higher-dimensional spaces; and so on. However, any row may be flagged numerous times. For example, a row may have an unsual value in Column F, an unsual pair of values in columns A,E, as well as an unusual triple of values in columns B,C,D. The row's total outlier score would be the sum of the scores derived from these.

## Interpretability
CountsOutlierDetector takes its name from the fact it examines the exact count of each value with categorical columns, and each range of values with numeric columns. It considers the outlierness of a combination of values considering their marginal distributions as well as the expected joint distribution given the cardinalities of the relevant columns, so provides sensible outlier scores, but based on the actual count of each value or combination of values. This ensures the outlier scores provided are clear, have full context, and allow straight-forward comparisons between rows. The name highlights the straight-forward and non-random nature of the detector, based on actual counts of values and combinations of values.  

The algorithm also provides clear outliers as it emphasizes outliers in low-dimensionalities, which are simpler to understand than those in higher dimensions. For example, indicating a row has an unusual combination of values in Columns A and B is inherently more comprehensible than indicating a row has an unusual combination of values in Colummns A, B, C, D, and E. In some sense, this may be considered a limitation: as the tool tends to flag outliers in low dimensionalities and not in high dimensionalities, it may be assumed it is missing those in higher dimensions. However, this is typically an artifact of the distribution of data, as testing to-date indicates the vast majority of outliers may be described as outliers using relatively few features, which is what this tool seeks to do. That is, the algorithm allows the tool to describe each outlier in as few dimensions as possible, which then presents each outlier as clearly as possible. 

## Visual Explanations
Using the explain_row() API, users can get a breakdown of the rational behind the score given for that row. For any one-dimension outliers found, bar plots or histograms are presented putting the value in context of the other values in the column. For further context, other values also flagged as anomalous are also shown. For any two-dimensional outliers found, a scatter plot (in the case of two numeric columns), strip plot (in the case on one numeric and one categorical columns), or heatmap (in the case of two categorical columns) will be presented. This shows quite clearly how the value compares to other values in this space. For higher-dimensional outliers, plots of this nature are not possible, and so bar plots are presented, giving the counts of each combination of values within the current space (combination of features), giving the count for the current combination of values / bins in context. 

For the highest level of interpretability, limit max_dimensions to 2, which will examine the dataset only for 1d and 2d outliers, presenting the results as one-dimensional bar plots or histograms, or two-dimensional plots, which allow the the most complete understanding of the space presented.

## Categorical vs Numeric Outlier Detectors
In general, most outlier detectors may be considered to be one of two types: categorical- or numeric-based detectors. In the former case, all columns are assumed to be categorical, and any numeric columns must be binned. In the latter case, all columns are assumed to be numeric, and any categorical columns must be numerically encoded, for example with one-hot encoding or count encoding. CountsOutlierDetector is an example of the former, treating all features as categorical. In the case of numeric columns, binning the values can loose some information, as the order of the bins is lost, but a categorical approach can add greatly to the interpretability of the model. This eliminates scaling data and deterimining the distance metric used between points. And, as all values are considered to be simply in one bin or not, it removes any processing time required to calculate the distances between points, between points and cluster centers, or other such distances.

To determine if a value is rare, we simply examine the count of its value. To determine if a pair of values is rare, we simply examine the count of this pair of values, compared to other pairs of values in the two columns. There may be some loss of fidelity in some cases, but overall, this allows for fast and interpretable results. 

## Examining Greater Numbers of Dimensions
The detector may examine up to a specified number of dimensions, but our testing indicates it is rarely necessary to go beyond three or four dimensions. This ensures the outlier detection process is quite tractable, but also ensures that the outliers identified are interpretable. 

Most detectors, in some sense, attempt to find the same rare values and combinations of values as CountsOutlierDetector explicitely searches for, though use algorithms optimized for a variety of concerns, such as speed and scalability. Doing so, they more-or-less approximate the process here, though not necessarily examining all combinations; stocastic processes are often used, which lead to faster execution times and probably approximately correct (PAC) solutions, but can miss some outliers. For example, if a row has a single, very unusual value in one column, any outlier detector would, everything else equal, wish to flag this (though may not if other rows are considered more outlierish with respect to that detector's algorithm, as can happen with CountsOutlierDetector as well, but assume for a moment a very unusual value in a column with few values, in a table with few columns). Though most detectors do not explicitely check for outliers in specific subspaces, they do hope to, following the algorithms they have, catch these. In many cases they will and many they will not, as different detectors emphasize different forms of outliers. CountsOutlierDetector will, though, be able to explain the score given. 

Many detectors do go, in a sense, beyond what CountsOutlierDector does, and check for higher-dimensional outliers, which this process may miss. For example, some detectors may detect instances that are effectively, for example, 8d-outliers (instances that are not unusual in any 1d, 2d,...7d sense, but are unusual if eight dimensions are considered). By default, CountsOutlierDectors does not examine this many dimensions and would likely miss this outlier. We argue that this is acceptable in most cases, as it's a favourable trade-off for more complete (and as such, less biased) examinations of lower-dimensional spaces and more stable, as well as comprehensible results. 

Further, there is a strong argument that outliers in lower dimensions are most often more relevant than those in higher dimensions. For example, a row that is an unusual combination of, say, 10 values (but not of any 1, 2,..., or 9 values) is likely to be less relevant than a row with a single unusual value, unusual pair, or unusual triple of values. This is in large part due to the curse of dimensionality, and the high likelihood of finding unusual combinations simply by chance in high dimensions. In most cases, when considering many columns, the number of combinations is very large, and consequently the number of rows in any given combination is very low, so often no rows can be reliably flagged as outliers in any case. This is in distinction to considering, for example, two or three dimensions, where the number of rows expected in each combination (depending on the cardinality and distribution of the column in the case of categorical columns, and depending on the distribution in the case of numeric columns), can be reasonably high. Where there are less combinations to consider, those combinations that are inliers will have substantially higher counts, and the outlier combinations may be identfified with significantly more reliability. 

As an example, consider a dataset with 10,000 rows. If the columns are, in this example, categorical, and each have cardinality of five, then considering first 1d outliers, for each column, we examine only each of the five values, and flag any that are very rare, if any. Here we expect the five values to have approximately 10,000/5 =2000 rows each, though the actual distribution will typically be quite uneven. When considering next 2d outliers, we look at each set of two columns, and so 5 * 5=25 combinations of values in each pair of columns. These will average 400 rows each, though again, some may be much more or much less frequent. With 3d outliers, there are 5 * 5 * 5=125 combinations of values, averaging 80 rows each. With 4d outliers, we have 5 * 5 * 5 * 5=625 combinations, averaging 16 rows each. At 5d we have 5 * 5 * 5 * 5 * 5=3125 combinations of values, averaging 3.2 rows each. So, with 10,000 rows, even by 4 or 5 dimensions, it may not be possible to identify outliers. However, this varies from one dataset to another. Often, even at 5 or 6 dimensions, though there are thousands of potential combinations, there are in fact only a small number of of actual combinations, and some of these are much more infrequent than the others and may reasonably be flagged as outliers. This does occur, though, exponantially less often as dimensions are added. 

As well, experiments using CountsOutlierDetector have found that usually an acceptable and useful number of rows are flagged in lower dimensions, obviating the need to search higher dimesions, though the tool does allow examining higher dimensions where desired. But, examining these spaces is exponantially slower, and identifies, where it identifies any rows, rows that are (arguably) less relevant, statistically less significant, and less interpretable. Testing has shown that examining beyond four dimensions identifies usually few outliers, which makes intuitive sense as having a 5d outlier, for example, while possible, requires having a row that is unusual in five dimensions, but not in any 1d, 2d, 3d or 4d subspace, and consider that there are a large number of 3d and 4d subspaces within a 5d space. 

We suspect the intractability of examining higher dimensional spaces in an exhaustive manner has dissuaded researchers from similar approaches, but we hope to demonstrate that examining up to, depending on the dataset and parameters used, approximately 3 to 5 dimensions is quite sufficient to identify the most outlierish instances. This approach provides a useful midpoint between detectors considering only single columns in isolation (ignoring all feature interactions) such as HBOS and Entropy-based methods on the one hand, and methods that consider all columns but tend to produce useful but inscrutable results, such as IF, LOF, and kth NN. 

## Binning
A known issue with the use of categorical-based outlier detection systems, such as CountsOutlierDetector, with numeric data is they require binning the data, and the results can be sensitive to the binning system used, both the number of bins and the placement of the split points between bins. This is valid, but we believe acceptable, as the majority of real-world data is mixed, containing both categorical and numeric columns, and binning the numeric columns is a sensible, as well as interpretable, approach. We use, by default, seven bins (which corresponds to a common cardinality of categorical columns in the openml.org dataset) and equal-width binning. Equal-count binning is not advised, as it precludes identifying 1d outliers in the binned columns. 

Binning numeric data does introduce a small number of decisions, but also eliminates the distance metrics required by numeric-based outlier detectors, and the need to numerically encode categorical columns. With numeric outlier detectors, it is necessary to encode the categorical features, which creates a similar set of decisions, which also affects the performance of the detectors. 

## Numbers of Rows Flagged
Ideally an outlier detector flags a sensible number of rows, for example 0.1 to 2% of the total rows, as this makes intuitive sense. Particularly with large datasets, we expect at least some outliers, but flagging too many is counterintuitive, as by definition only so many rows may be unusual. As with most outlier detectors, this tends to give a non-zero score to many rows, but allows ordering the the outliers based on score, where the score is a very straightforward value. 

One does usually want a diversity of outliers as well (some sets of outlier rows may be essentially the same thing, so a user needs only an example of one, and a count of how often outliers of this sort occur). CountsOutlierDetector fascilitates this, as it will classify the outliers based on the number of columns and the specific set of columns. For example, there may be several outliers based on the combination of values in columns B, C and E, which may be considered, though possibly somewhat different (if containing different specifical values), either identical or at least related outliers. And the rows containing the same specific values in these columns may also be inditified and grouped together. 

## Key Qualities of CountsOutlierDetector
This detector has the advantages of: 

1) It is able to provide as clear as possible explanations. To explain why a row is scored as it is, an explanation may be given using only as many columns as necessary to explain its score.  

2) It is able to provide full statistics about each space, which allows it to provide full context of the outlierness of each row. For a given set of say, two, columns, if the columns have cardinalities of, for example, 5 and 10, then there are 50 combinations. The counts of all 50 may be provided to provide context for any combinations that were flagged as particularly unusual. 
 
This project in an attempt to demonstrate, and provide a working example, that outlier detection can be completely transparent, and though optimized outlier detectors are also very useful, there is a clear utility in interpretable outlier detection. 

## Example

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
det = CountsOutlierDetector()
results = det.predict(X)
```

The 4 returned pandas dataframes and strings provide information about what rows where flagged and why, as well as summary statistics about the dataset's outliers as a whole. 

## Example Notebooks

**Simple Example Notebook**

The [simple example](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/Examples_Counts_Outlier_Detector.ipynb) notebook provides some simple examples, using synthetic data and some toy datasets from sklearn. 

**OpenML Demo Notebook**

The [OpenML demo](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/demo_OpenML.ipynb) notebook provides an example detecting outliers in a real dataset from OpenML. This provides an example of examining the outliers identified by the detector.

**Tuning Hyperparameters Notebook**

The [hyperparameter tuning notebook](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/tune_hyperparameters.ipynb) notebook uses a set of datasets from OpenML (a set not used elsewhere) to identify what are generally the optimum hyperparameters for CountOutlierDetector, considering the three most relevant to performance and execution time: number of bins, maximum number of dimensions, and threshould. This confirms the default hyperparameters are typically strong. This uses a doping method to randomly modify real datasets such that the modified rows are known to be likely more anomalous than the non-modified rows. Specifically, this uses a technique of calculating the outlier scores on both the original and modified versions of each dataset and examining the increases in outlier scores. Where CountsOutlierDetector is well-behaving, the increases will correlate well with the modifications made in the doping process. Doing this allows us to determine where the hyperparameters best allow the detector to identify the modified rows. 

This takes some time to execute, as it tests several combinations of hyperparameters per dataset.

**Evaluation Notebook**

The [evaluation](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/evaluate_using_doping.ipynb) notebook uses a large and random set of datasets from OpenML, distinct from those used in the Tuning notebook, to compare the accuracy of CountsOutlierDetector to IsolationForest, a standard and well-established outlier detector. Both used default parameters. Overall, CountsOutlierDetector is quite competitive, out-performing IsolationForest in two of the three metrics used. While somewhat slower than IsolationForest, CountsOutlierDetector also performed well with respect to execution time. The main difference is CountsOutlierDetector is fully interpretable.

**Compare Dimensionalities Notebook**

The [compare dimensionalities](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/compare_dimensions.ipynb) notebook examines how many outliers tend to be found in higher-dimensional spaces than lower-dimensional spaces. This establishes that, while examining higher dimensional spaces is often useful, on the whole, most outliers can be detected considering only up to two or three dimensions. In fact, the (hyperparameter tuning notebook)[]

Many datasets just don't have outliers. Of the 78? checked, only one has any at 6d, though it has many. 

As expected, the patterns are quite different from one dataset to another. 



## API

### CountsOutlierDetector
```python
CountsOutlierDetector(
             n_bins=7,
             bin_names=None,
             max_dimensions=3,
             threshold=0.05,
             check_marginal_probs=False,
             max_num_combinations=100_000,
             min_values_per_column=2,
             max_values_per_column=25,
             results_folder="",
             results_name="",
             run_parallel=False,
             verbose=False)
```

**n_bins**: int

The number of bins used to reduce numeric columns to a small set of ordinal values.
        
**bin_names**: list of strings
            
List of provided bin names. If set to None, a default set of names will be used. Suitable to specify where a small number of bins are used and the defaults are not appropriate, for example, where non-English names should be used.

**max_dimension**s: int

The maximum number of columns examined at any time. If set to, for example, 4, then the detector will check for 1d, 2d, 3d, and 4d outliers, but not outliers in higher dimensions.
            
**threshold**: float
        
Used to determine which values or combinations of values are considered rare. Any set of values that has a count less than threshold * the expected count under a uniform distribution are flagged as outliers. For example, if considering a set of three columns, if the cardinalities are 4, 8, and 3, then there are 4*8*3=96 potential combinations. If there are 10,000 rows, we would expect (under a uniform distribution)
            10000/96 = 104.16 rows in each combination. If threshold is set to 0.25, we flag any combinations that have
            less than 104.16 * 0.25 = 26 rows. When threshold is set to a very low value, only very unusual values and
            combinations of values will be flagged. When set to a higher value, many more will be flagged, and rows
            will be differentiated more by their total scores than if they have any extreme anomalies.
            
**check_marginal_probs**: bool

If set true, values will be flagged only if they are both rare and rare given the marginal probabilities of
            the relevant feature values.  
            
**max_num_combinations**: int
        
This, as well as max_dimensions, determines the maximum number of dimensions that may be examined at a time.
            When determining if the detector considers, for example, 3d outliers, it examines the number of columns and
            average number of unique values per column and estimates the total number of combinations. If this exceeds
            max_num_combinations, the detector will not consider spaces of this dimensionality or higher. This parameter
            may be set to reduce the time or memory required or to limit the flagged values to lower dimensions for
            greater interpretability. It may also be set higher to help identify more outliers where desired.
            
**min_values_per_column**: int
        
The detector excludes from examination any columns with less than this number of unique values
            
**max_values_per_column**: int
        
The detector excludes from examination any columns with more than this number of unique values
            
**results_folder**: string
        
If specified, the output will be written to a .csv file in this folder. If unspecified,  no output file will
            be written. Required if results_name is specified.
            
**results_name**: string
        
Optional text to be included in the names of the output files, if created. The output file  names will also
            include the date and time, to allow multiple to be created without over-writing previous output files.
            
**run_parallel**: bool
        
If set True, the process will execute in parallel, typically allowing some performance gain.
            
**verbose**:
        
If set True, progress messages will be displayed to indicate how far through the process the detector is.


&nbsp;&nbsp;

### predict()
```python
predict(input_data)
```
This is the main API. This determines the outlier score of all rows in the data.

**input_data**: pandas dataframe, or data structure that may be converted to a pandas dataframe, such as
            numpy array, 2d python array, or dictionary

**Return**: dictionary
            
Returns a dictionary with the following elements:

'Scores': list

An array with an element for each row in the dataset. This is the detector's estimate of the
    outlierness of each row.

'Breakdown All Rows': pandas dataframe

This contains a row for each row in the original data, along with columns indicating the number of times
    each row was flagged based on 1d, 2d, 3d,... tests. A short explanation of each is also provided
    giving the columns, values (or bin number), and the fraction of rows having this combination. This is
    not intended to be readable, as is useful only for further analysis of the outliers.

'Breakdown Flagged Rows': pandas dataframe
    
This is a condensed form of the above dataframe, including only the rows flagged at least once, and
    only the relevant columns.

'Flagged Summary': pandas dataframe

This provides a high level view of the numbers of values, or combinations of values, flagged in 1d,
    2d, 3d... spaces, as well as the number of dimensions checked.

&nbsp;&nbsp;

### get_most_flagged_rows

```python
get_most_flagged_rows()
```

This is used to get the rows from the original data with the highest total scores.

Parameters: None

**returns**: pandas dataframe

This returns a dataframe with the set of rows matching all rows from the original data that received any
            non-zero score, ordered from highest score to lowest. This has the full set of columns from the original
            data as well as a SCORE column indicating the total score of the column

&nbsp;&nbsp;

### plot_scores_distribution
```python
plot_scores_distribution()
```

This presents three plots. The first two are bar plots providing the count of each score, the second with scores of zero omitted. The third presents this as a rank plot. This allows determining if there are any distinctions between low and high-scoring flagged rows, such that beyond some score flagged rows may be particularly considered as outliers. 

Parameters: None

Return Values: None

&nbsp;&nbsp;

### print_run_summary
```python
print_run_summary()
```
This prints a string summary of the execution of the detection process, describing the anomalies found in each dimensionality. 

&nbsp;&nbsp;

### explain_row
``` python
explain_row(row_index, max_plots=50)
```

This provides an explanation for a single row, in the form of plots. One or more plots will be drawn for each anomaly found in the row. Where anomalies were found in 1d, these will be displayed as a bar plot. Any in 2d will be displayed as a scatter plot, heatmap, or strip plot depending if the features were categorical or numeric. Any in three or more dimensions will be displayed as a bar plot or histogram. 

**row_index**: int

Row index (zero-based) of the row for which this will provide an explanation. The relevant rows can be found by first calling get_most_flagged_rows()

**max_plots**: int

The maximum number of plots which can be displayed. In some cases, rows may be flagged in many different ways, and plotting all is unnecessary. 

&nbsp;&nbsp;

### explain_features
``` python
explain_features(self, features_arr)
```
Display the counts for each combination of values within a specified set of columns. This would typically be
        called to follow up a call to explain_row(), where a set of 3 or more features were identified with at least
        one unusual combination of values.

**features_arr**: list of strings

A list of features within the dataset

**Returns**: None

&nbsp;&nbsp;

