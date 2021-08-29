# CountsOutlierDetector
CountsOutlierDetector is an interpretable, intuitive, efficient outlier detector intended for categorical data (all numeric data will be automatically binned). It is designed to provide clear explanations of the rows flagged as outliers. Despite the common business need to understand and assess outliers within tabular datasets, most outlier detectors are very much blackboxes, often much more so than classifiers and regressors. Although the algorithms employed by detectors are often interpretable, the individual predictions are generally not. For example, particularly with high-dimensional data, such standard detectors as IsolationForests, Local Outlier Factor, and kthNN, produce scores that may be difficult to assess. Futher, it's particularly difficult to assess the counterfactuals: the outlierness of the rows not flagged by the detectors. 

CountsOutlierDetector is based on the intuition that all outliers are essentiall based on either unusual single values, or unusual combinations of values. It works by simply examing each column indivually and identifying all values that are unusual with respect to their columns. These are known as *1d outliers*, that is: outliers based on considering a single dimension at a time. It then examines each pair of columns, identifying the rows with pairs of unusual values. For example, having fur may be common as well as laying eggs, but the combination is rare. Or an age of 1yr may be common and a height of 6', but the combination rare, most likely flagging an error. These are known as *2d outliers*. T

The detector may examine up to a specified number of dimensions, but testing indicated it's rarely necessary to go beyond 3 or 4 dimensions. This ensures the outlier detection process is quite tractable, but also ensures that the outliers identified are interpretable. 

Most detectors, in some sense, attempt to find these, though use algorithms optimized for a variety of concerns, such as speed and scalability. Doing so, they more or less approximate the process here, not necessarily examining all combinations. 

not stocastic. doesn't miss anything, but doesn't go up to hgh dimensions.

# Example

# Example Files

# Statistics 

link to results
