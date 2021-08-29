# CountsOutlierDetector
CountsOutlierDetector is an interpretable, intuitive, efficient outlier detector intended for categorical data (all numeric data will be automatically binned). It is designed to provide clear explanations of the rows flagged as outliers. Despite the common business need to understand and assess outliers within tabular datasets, most outlier detectors are very much blackboxes, often much more so than classifiers and regressors. Although the algorithms themselves employed by detectors are often interpretable, the individual predictions are generally not. For example, particularly with high-dimensional data, such standard detectors as IsolationForests, Local Outlier Factor, and kth Nearest Neighbors, produce scores that may be difficult to assess. Futher, it's particularly difficult to assess the counterfactuals: the outlierness of the rows not flagged by the detectors. 

CountsOutlierDetector is based on the intuition that all outliers are essentially based on either unusual single values, or unusual combinations of values. It works by simply first examing each column indivually and identifying all values that are unusual with respect to their columns. These are known as *1d outliers*, that is: outliers based on considering a single dimension at a time. It then examines each pair of columns, identifying the rows with pairs of unusual values. For example, having fur may be common as well as laying eggs, but the combination is rare. Or, with numeric columns, an age of 1yr may be common and a height of 6', but the combination rare, most likely flagging an error. These are known as *2d outliers*. The detector then considers sets of 3 columns, and so on. 

The detector may examine up to a specified number of dimensions, but our testing indicates it is rarely necessary to go beyond 3 or 4 dimensions. This ensures the outlier detection process is quite tractable, but also ensures that the outliers identified are interpretable. 

Most detectors, in some sense, attempt to find these, though use algorithms optimized for a variety of concerns, such as speed and scalability. Doing so, they more or less approximate the process here, not necessarily examining all combinations; stocastic processes are often used, which lead to faster execution times and probably approximately correct solutions, but can miss some outliers. Many do, however, go in a sense beyond what CountsOutlierDector does, and check for higher-dimensional outliers, which this process may miss. We argue that is acceptable most of the time as it's an acceptable trade-off for more complete (and as such, less biased) examination of lower-dimensional spaces, more comprehensible explanations, and the sense that outliers in lower dimensions are most often more relevant than those in higher dimensions. For example, a row that is an unusual combination of, say, 10 values (but not of 1, 2,...,9 values) is likely to be less relevant than a row with a single unusual value, or single pair, or triple of values. 

This detector, then, has the advantages of: 1) being able to provide as clear as possible explanations (explanations are based on a single column if possible, two columns if necessary, 3 if necessary, and so on); and 2) being able to provide full statistics about each space to provide full context of the outlierness of each row. 

# Example

'''python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
det = CountsOutlierDetector()
flagged_rows_df, row_explanations, output_msg, run_summary_df = det.predict(X)

The 4 returned pandas dataframes and strings provide information about what rows where flagged and why, as well as summary statistics about the dataset's outliers as a whole. 

# Example Files
[Example Counts Outlier Detector](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/Examples_Counts_Outlier_Detector.ipynb) is a simple notebook providing some examples of use of the detector.

[Tests Counts Outlier Detector](https://github.com/Brett-Kennedy/CountsOutlierDetector/blob/main/examples/Test_CountsOutlierDetector.py) is a python file that tests the outlier detector over a large number of random datasets (100 by default). This uses the [DatasetsEvaluator](https://github.com/Brett-Kennedy/DatasetsEvaluator) tool to aid with collecting and filtering datasets from openml.org. Some results from the execution of this are included below.

# Statistics 

link to results
