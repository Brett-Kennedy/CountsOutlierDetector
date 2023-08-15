import os
import pandas as pd
import numpy as np
import random
from pandas.api.types import is_numeric_dtype
from datetime import datetime, timedelta
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
import concurrent
from statistics import mean
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from IPython import get_ipython
from IPython.display import display, Markdown

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class CountsOutlierDetector:
    def __init__(self,
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
                 verbose=False):
        """
        :param n_bins: int
            The number of bins used to reduce numeric columns to a small set of ordinal values.
        :param bin_names: list of strings
            todo: fill in
        :param max_dimensions: int
            The maximum number of columns examined at any time. If set to, for example, 4, then the detector will check
            for 1d, 2d, 3d, and 4d outliers, but not outliers in higher dimensions.
        :param threshold: float
            Used to determine which values or combinations of values are considered rare. Any set of values that has
            a count less than threshold * the expected count under a uniform distribution are flagged as outliers. For
            example, if considering a set of three columns, if the cardinalities are 4, 8, and 3, then there are
            4*8*3=96 potential combinations. If there are 10,000 rows, we would expect (under a uniform distribution)
            10000/96 = 104.16 rows in each combination. If threshold is set to 0.25, we flag any combinations that have
            less than 104.16 * 0.25 = 26 rows. When threshold is set to a very low value, only very unusual values and
            combinations of values will be flagged. When set to a higher value, many more will be flagged, and rows
            will be differentiated more by their total scores than if they have any extreme anomalies.
        :param check_marginal_probs: bool
            If set true, values will be flagged only if they are both rare and rare given the marginal probabilities of
            the relevant feature values.  
        :param max_num_combinations: int
            This, as well as max_dimensions, determines the maximum number of dimensions that may be examined at a time.
            When determining if the detector considers, for example, 3d outliers, it examines the number of columns and
            average number of unique values per column and estimates the total number of combinations. If this exceeds
            max_num_combinations, the detector will not consider spaces of this dimensionality or higher. This parameter
            may be set to reduce the time or memory required or to limit the flagged values to lower dimensions for
            greater interpretability. It may also be set higher to help identify more outliers where desired.
        :param min_values_per_column: int
            The detector excludes from examination any columns with less than this number of unique values
        :param max_values_per_column: int
            The detector excludes from examination any columns with more than this number of unique values
        :param results_folder: string
            If specified, the output will be written to a .csv file in this folder. If unspecified,  no output file will
            be written. Required if results_name is specified.
        :param results_name: string
            Optional text to be included in the names of the output files, if created. The output file  names will also
            include the date and time, to allow multiple to be created without over-writing previous output files.
        :param run_parallel: bool
            If set True, the process will execute in parallel, typically allowing some performance gain.
        :param verbose:
            If set True, progress messages will be displayed to indicate how far through the process the detector is.
        """

        if n_bins < 2:
            print("The minimum value for n_bins is 2. Using n_bins=2 for outlier detection.")
            n_bins = 2

        if max_dimensions > 6:
            print("The maximum value for max_dimensions is 6. Using max_dimensions=6 for outlier detection.")
            max_dimensions = 6

        if results_folder == "" and results_name != "":
            print(("Error: results_folder is required when results_file is specified. Specify both in order to save "
                   "output to a .csv file."))
            return

        self.n_bins = n_bins
        self.max_dimensions = max_dimensions
        self.threshold = threshold
        self.check_marginal_probs = check_marginal_probs
        self.max_num_combinations = max_num_combinations
        self.min_values_per_column = min_values_per_column
        self.min_values_per_column = min_values_per_column
        self.max_values_per_column = max_values_per_column
        self.results_folder = results_folder
        self.results_name = results_name
        self.run_parallel = run_parallel
        self.verbose = verbose

        # Determine the bin names. If specified as None, a default set of names will be used based on the number of
        # bins
        if bin_names is not None:
            if len(bin_names) != n_bins:
                print("The number of bin names must match n_bins")
                return
            self.bin_names = bin_names
        else:
            if n_bins == 2:
                self.bin_names = ['Low', 'High']
            elif n_bins == 3:
                self.bin_names = ['Low', 'Medium', 'High']
            elif n_bins == 4:
                self.bin_names = ['Low', 'Medium-Low', 'Medium-High', 'High']
            elif n_bins == 5:
                self.bin_names = ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
            elif n_bins == 6:
                self.bin_names = ['Very Low', 'Low', 'Medium-Low', 'Medium-High', 'High', 'Very High']
            elif n_bins == 7:
                self.bin_names = ['Very Low', 'Low', 'Medium-Low', 'Medium', 'Medium-High', 'High', 'Very High']
            else:
                self.bin_names = ['Bin ' + str(x) for x in range(n_bins)]

        # An array indicating the type of each column in the dataset. All columns are considered as either numeric
        # or categorical.
        self.col_types_arr = []

        # List of the numeric columns, in their original order
        self.numeric_col_names = None

        # An encoder used to convert all categorical values to numbers. An ordinal representation is used for each
        # categorical value for efficiency.
        self.ordinal_encoders_arr = []

        # Copy of the input dataframe.
        self.orig_df = None

        # Modified version of the input dataframe, with binned numeric values and removed high-cardinality columns
        self.data_df = None

        # Equivalent to self.data_df, in numpy format
        self.data_np = None

        # Array of bin edges. One element per numeric column
        self.bin_edges = None

        # Dataframe with the same rows as self.data_df, but columns indicating the counts of flagged outliers and
        # descriptions of these.
        self.flagged_rows_df = None
        
        # String representation of the progress of the call to predict()
        self.run_summary = None

        # Data structure describing each outlier pattern
        self.results = None

        # The number of dimension actually examined. This may be less than max_dimensions, given max_num_combinations
        self.dimensions_examined = None

        # Boolean values indicated which values or pairs of values are flagged as outliers
        self.rare_1d_values = None
        self.rare_2d_values = None

        # The set of unique values and count of unique values in each column
        self.unique_vals = None
        self.num_unique_vals = None

        # Colormap used for heatmaps. This is a standard colour map, shifted, to center between flagged and non-flagged
        # values
        self.shifted_cmap = None

        # Set the seed to ensure consistent results from run to run on the same dataset.
        np.random.seed(0)
        random.seed(0)

        # Set display options for dataframes. This is not done with notebooks, as in some cases it may have performance
        # implications for jupyter.
        if not is_notebook():
            pd.options.display.max_columns = 1000
            pd.options.display.max_rows = 1000
            pd.options.display.width = 10000

    # Using numba appears to give similar performance results. The current implementation is without numba.
    def predict(self, input_data):
        """
        Determine the outlier score of all rows in the data

        :param input_data: pandas dataframe, or data structure that may be converted to a pandas dataframe, such as
            numpy array, 2d python array, or dictionary

        :return: dictionary
            Returns a dictionary with the following elements:

            'Scores: list
                todo: fill in

            'Breakdown All Rows': pandas dataframe
                This contains a row for each row in the original data, along with columns indicating the number of times
                each row was flagged based on 1d, 2d, 3d,... tests. A short explanation of each is also provided
                giving the columns, values (or bin number), and the fraction of rows having this combination.

            'Breakdown Flagged Rows': pandas dataframe
                This is a condensed form of the above dataframe, including only the rows flagged at least once, and
                only the relevant columns.

            'Flagged Summary': pandas dataframe
                This provides a high level view of the numbers of values, or combinations of values, flagged in 1d,
                2d, 3d... spaces.
        """

        def update_run_summary(arr):
            """
            Updates the msg used to describe each execution of predict(). This gives the counts of the number of rows
            flagged n times for a given dimensionality, for each n. For example, the number of rows flagged 0 times,
            1 times, 2 times, and so on.

            :param arr:
                An array representing the number of outliers detected per row for a given dimensionality. arr should
                have one element for each row in the full dataset.
            """
            unique_counts = sorted(list(set(arr)))
            for uc in unique_counts:
                self.run_summary += f"\nNumber of rows given score: {uc:2}: {arr.count(uc):5}"

        def get_unique_vals():
            """
            This finds the unique values in each column.

            :return:
            unique_vals: A 2d array, with a list of unique values for each feature.
            num_unique_values: A 1d array, with a count of unique values for each feature.
            """

            # An element for each column. For categorical columns, lists the unique values. This is used to maintain a
            # consistent order of the values within each column.
            unique_vals = []
            num_unique_vals = []

            for i in range(num_cols):
                uv = sorted(self.data_df.iloc[:, i].unique())
                unique_vals.append(uv)
                num_unique_vals.append(len(uv))
            return unique_vals, num_unique_vals

        def get_2d_fractions(i, j):
            """
            Given two columns i and j: get, for each pair of values, the fraction of the dataset and the row numbers.
            Returns:
                two_d_fractions: a 2d array, with a row for each value in column i, and a column for each value in
                    column j. Each cell contains the count for that combination.
                two_2_row_nums:
            """

            two_d_fractions = []
            two_d_row_nums = []
            for i_val in self.unique_vals[i]:
                i_vals_fractions = []
                i_vals_row_nums = []
                cond1 = (self.data_np[:, i] == i_val)
                for j_val in self.unique_vals[j]:
                    cond2 = (self.data_np[:, j] == j_val)
                    rows_both = np.where(cond1 & cond2)
                    i_vals_fractions.append(len(rows_both[0]) / num_rows)
                    i_vals_row_nums.append(rows_both[0])
                two_d_fractions.append(i_vals_fractions)
                two_d_row_nums.append(i_vals_row_nums)
            return two_d_fractions, two_d_row_nums

        def get_1d_stats():

            # Parallel 2d array to the unique_vals array. Indicates the fraction of the total dataset for this value in
            # this column.
            fractions_1d = [[]] * num_cols

            # Parallel 2d array to the unique_vals array. Boolean value indicates if the value is considered a 1d
            # outlier.
            rare_1d_values = [[]] * num_cols

            # Integer value for each row in the dataset indicating how many individual columns have values considered
            # outliers.
            outliers_1d_arr = [0] * num_rows

            # Text explanation for each row explaining each 1d outlier.
            outliers_explanation_arr = [[]] * num_rows

            if self.verbose:
                print("Examining subspaces of dimension 1")

            for i in range(num_cols):

                col_threshold = (1.0 / self.num_unique_vals[i]) * self.threshold

                # Array with an element for each unique value in column i. Indicates the fraction of the total dataset
                # held by that value.
                col_fractions_1d = []

                # Array with an element for each unique value in column i. Indicates if that value is considered rare.
                col_rare_1d_values = []

                for v in self.unique_vals[i]:  # loop through each unique value in the current column.
                    frac = self.data_df.iloc[:, i].tolist().count(v) / num_rows
                    col_fractions_1d.append(frac)
                    rare_values_flag = (frac < col_threshold) and (frac < 0.01)
                    if rare_values_flag:
                        rows_matching = np.where(self.data_np[:, i] == v)
                        for r in rows_matching[0]:
                            outliers_1d_arr[r] += 1
                            expl = [[self.data_df.columns[i]], [self._get_col_value(i, v)]]
                            if outliers_explanation_arr[r] == []:
                                outliers_explanation_arr[r] = [expl]
                            else:
                                outliers_explanation_arr[r].append(expl)
                    col_rare_1d_values.append(rare_values_flag)
                fractions_1d[i] = col_fractions_1d
                rare_1d_values[i] = col_rare_1d_values

            self.run_summary += (f"\n\n1d: Number of common values (over all columns): "
                                 f"{flatten(rare_1d_values).count(False)}")
            self.run_summary += f"\n1d: Number of rare values: {flatten(rare_1d_values).count(True)}"
            update_run_summary(outliers_1d_arr)
            return fractions_1d, rare_1d_values, outliers_1d_arr, outliers_explanation_arr

        def get_2d_stats():
            """
            This returns 2 parallel 4d arrays: fractions_2d and rare_2d_values, both with the dimensions:
            i column, j column, value in i column, value in j column.

            This also returns: outliers_2d_arr, outliers_explanation_arr
            """

            # Each element stores the fraction of the total dataset with this combination of values.
            fractions_2d = [] * num_cols

            # Each element stores a boolean indicating if this combination of values is considered rare in the 2d sense.
            rare_2d_values = [] * num_cols

            # Integer value for each row in the dataset indicating how many pairs of columns have combinations
            # considered outliers.
            outliers_2d_arr = [0] * num_rows

            # String for each row in the dataset describing why it was flagged. Blank if not flagged. A list if flagged
            # multiple times.
            outliers_explanation_arr = [[]] * num_rows

            if self.verbose:
                print("Examining subspaces of dimension 2")

            for i in range(num_cols):
                fractions_2d.append([[]] * num_cols)
                rare_2d_values.append([[]] * num_cols)

            dt_display_prev = datetime.now()
            for i in range(num_cols - 1):
                if self.verbose and i > 0 and len(self.data_df.columns) > 20:
                    num_combinations_left = self.__get_num_combinations(dim=2, num_cols_processed=i)
                    percent_complete = (num_combinations - num_combinations_left) * 100.0 / num_combinations
                    dt_display = datetime.now()
                    if (dt_display - dt_display_prev) > timedelta(seconds=5):
                        print(f"  {percent_complete:.2f}% complete")
                        dt_display_prev = dt_display

                for j in range(i + 1, num_cols):
                    local_fractions, two_d_row_nums = get_2d_fractions(i, j)
                    fractions_2d[i][j] = local_fractions

                    # Determine which of these fractions would be considered rare in the 2d sense
                    i_rare_arr = []
                    expected_under_uniform = 1.0 / (len(self.unique_vals[i]) * len(self.unique_vals[j]))
                    for i_vals_idx in range(len(fractions_2d[i][j])):
                        j_rare_arr = []
                        for j_vals_idx in range(len(fractions_2d[i][j][i_vals_idx])):
                            current_fraction = fractions_2d[i][j][i_vals_idx][j_vals_idx]
                            if self.check_marginal_probs:
                                expected_given_marginal = fractions_1d[i][i_vals_idx] * \
                                                          fractions_1d[j][j_vals_idx] * \
                                                          self.threshold
                            else:
                                expected_given_marginal = np.inf
                            rare_value_flag = (not rare_1d_values[i][i_vals_idx]) and \
                                              (not rare_1d_values[j][j_vals_idx]) and \
                                              (current_fraction < (expected_under_uniform * self.threshold)) and \
                                              (current_fraction < expected_given_marginal) and \
                                              (current_fraction < 0.01)

                            if rare_value_flag:
                                row_nums = two_d_row_nums[i_vals_idx][j_vals_idx]
                                assert len(row_nums) == round(current_fraction * num_rows), \
                                    (f"len of matching rows: {len(row_nums)}, fraction*num_rows: " 
                                     f"current_fraction*num_rows: {current_fraction * num_rows}")
                                for r in row_nums:
                                    outliers_2d_arr[r] += 1
                                    expl = [[self.data_df.columns[i], self.data_df.columns[j]],
                                            [self._get_col_value(i, i_vals_idx), self._get_col_value(j, j_vals_idx)]]
                                    if not outliers_explanation_arr[r]:
                                        outliers_explanation_arr[r] = [expl]
                                    else:
                                        outliers_explanation_arr[r].append(expl)
                            j_rare_arr.append(rare_value_flag)
                        i_rare_arr.append(j_rare_arr)
                    rare_2d_values[i][j] = i_rare_arr

            out = flatten(rare_2d_values)
            self.run_summary += f"\n\n2d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n2d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_2d_arr)
            return fractions_2d, rare_2d_values, outliers_2d_arr, outliers_explanation_arr

        def get_3d_stats(num_combinations):
            """
            This returns 2 parallel 6d arrays: fractions_3d and rare_3d_values, both with the dimensions:
            i column, j column, k column, value in i column, value in j column, value in the k column

            It also returns: outliers_3d_arr, outliers_explanation_arr, and column_combos_checked.

            todo: explain each of these
            """

            fractions_3d = [[]] * num_cols
            rare_3d_values = [[]] * num_cols
            outliers_3d_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows
            column_combos_checked = 0

            run_parallel_3d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_3d = False

            if self.verbose:
                print("Examining subspaces of dimension 3")

            if run_parallel_3d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_3d,
                                            self,
                                            i,
                                            self.data_np,
                                            num_cols,
                                            num_rows,
                                            self.unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values,
                                            self.threshold)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_3d_values[f_idx] = rare_arr_for_i
                        outliers_3d_arr = [x + y for x, y in zip(outliers_3d_arr, outliers_3d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
            else:
                dt_display_prev = datetime.now()
                for i in range(num_cols):
                    if self.verbose and i > 0 and len(self.data_df.columns) > 20:
                        num_combinations_left = self.__get_num_combinations(dim=3, num_cols_processed=i)
                        percent_complete = (num_combinations - num_combinations_left) * 100.0 / num_combinations
                        dt_display = datetime.now()
                        if (dt_display - dt_display_prev) > timedelta(seconds=5):
                            print(f"  {percent_complete:.2f}% complete")
                            dt_display_prev = dt_display

                    rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_3d(
                        self,
                        i,
                        self.data_np,
                        num_cols,
                        num_rows,
                        self.unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        self.threshold
                    )
                    rare_3d_values[i] = rare_arr_for_i
                    outliers_3d_arr = [x + y for x, y in zip(outliers_3d_arr, outliers_3d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_3d_values)
            self.run_summary += f"\n\n3d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n3d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_3d_arr)
            return fractions_3d, rare_3d_values, outliers_3d_arr, outliers_explanation_arr, column_combos_checked

        def get_4d_stats(num_combinations):
            """
            This returns 2 parallel 8d arrays: fractions_4d and rare_4d_values, both with the dimensions:
            i column, j column, k column, m column,
            value in i column, value in j column, value in the k column, value in the m column

            It also returns: outliers_4d_arr, outliers_explanation_arr, and column_combos_checked.

            These are analogous to get_3d_stats()
            """

            fractions_4d = [[]] * num_cols
            rare_4d_values = [[]] * num_cols
            outliers_4d_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows
            column_combos_checked = 0

            run_parallel_4d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_4d = False

            if self.verbose:
                print("Examining subspaces of dimension 4")

            if run_parallel_4d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_4d,
                                            self,
                                            i,
                                            self.data_np,
                                            num_cols,
                                            num_rows,
                                            self.unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values,
                                            rare_3d_values)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_4d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_4d_values[f_idx] = rare_arr_for_i
                        outliers_4d_arr = [x + y for x, y in zip(outliers_4d_arr, outliers_4d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
            else:
                dt_display_prev = datetime.now()
                for i in range(num_cols):
                    if self.verbose and i > 0:
                        num_combinations_left = self.__get_num_combinations(dim=4, num_cols_processed=i)
                        percent_complete = (num_combinations - num_combinations_left) * 100.0 / num_combinations
                        dt_display = datetime.now()
                        if (dt_display - dt_display_prev) > timedelta(seconds=5):
                            print(f"  {percent_complete:.2f}% complete")
                            dt_display_prev = dt_display

                    rare_arr_for_i, outliers_4d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_4d(
                        self,
                        i,
                        self.data_np,
                        num_cols,
                        num_rows,
                        self.unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        rare_3d_values,
                        self.threshold
                    )
                    rare_4d_values[i] = rare_arr_for_i
                    outliers_4d_arr = [x + y for x, y in zip(outliers_4d_arr, outliers_4d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_4d_values)
            self.run_summary += f"\n\n4d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n4d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_4d_arr)
            return fractions_4d, rare_4d_values, outliers_4d_arr, outliers_explanation_arr, column_combos_checked

        def get_5d_stats(num_combinations):
            """
            This returns 2 parallel 10d arrays: fractions_5d and rare_5d_values, both with the dimensions:
            i column, j column, k column, m column, n column
            value in i column, value in j column, value in the k column, value in the m column, values in the n column

            It also returns: outliers_5d_arr,  outliers_explanation_arr, and column_combos_checked.

            These are analogous to get_3d_stats()
            """

            # todo: update this comment. Make more general, so don't repeat it
            fractions_5d = [[]] * num_cols
            rare_5d_values = [[]] * num_cols
            outliers_5d_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows
            column_combos_checked = 0

            run_parallel_5d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_5d = False

            if self.verbose:
                print("Examining subspaces of dimension 5")

            if run_parallel_5d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_5d,
                                            self,
                                            i,
                                            self.data_np,
                                            num_cols,
                                            num_rows,
                                            self.unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values,
                                            rare_3d_values,
                                            rare_4d_values,
                                            self.threshold)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_5d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_5d_values[f_idx] = rare_arr_for_i
                        outliers_5d_arr = [x + y for x, y in zip(outliers_5d_arr, outliers_5d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
            else:
                dt_display_prev = datetime.now()
                for i in range(num_cols):
                    if self.verbose and i > 0:
                        num_combinations_left = self.__get_num_combinations(dim=5, num_cols_processed=i)
                        percent_complete = (num_combinations - num_combinations_left) * 100.0 / num_combinations
                        dt_display = datetime.now()
                        if (dt_display - dt_display_prev) > timedelta(seconds=5):
                            print(f"  {percent_complete:.2f}% complete")
                            dt_display_prev = dt_display

                    rare_arr_for_i, outliers_5d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_5d(
                        self,
                        i,
                        self.data_np,
                        num_cols,
                        num_rows,
                        self.unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        rare_3d_values,
                        rare_4d_values,
                        self.threshold
                    )
                    rare_5d_values[i] = rare_arr_for_i
                    outliers_5d_arr = [x + y for x, y in zip(outliers_5d_arr, outliers_5d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in
                                                zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i
            out = flatten(rare_5d_values)
            self.run_summary += f"\n\n5d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n5d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_5d_arr)
            return fractions_5d, rare_5d_values, outliers_5d_arr, outliers_explanation_arr, column_combos_checked

        def get_6d_stats(num_combinations):
            """
            This returns 2 parallel 12d arrays: fractions_6d and rare_6d_values, both with the dimensions:
            i column, j column, k column, m column, todo: more??
            value in i column, value in j column, value in the k column, value in the m column # todo: more??

            It also returns outliers_6d_arr, outliers_explanation_arr, and column_combos_checked.

            These are analogous to get_3d_stats()
            """

            fractions_6d = [[]] * num_cols
            rare_6d_values = [[]] * num_cols
            outliers_6d_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows
            column_combos_checked = 0

            run_parallel_6d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_6d = False

            if self.verbose:
                print("Examining subspaces of dimension 6")

            if run_parallel_6d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_6d,
                                            self,
                                            i,
                                            self.data_np,
                                            num_cols,
                                            num_rows,
                                            self.unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values,
                                            rare_3d_values,
                                            rare_4d_values,
                                            rare_5d_values,
                                            self.threshold)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_6d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_6d_values[f_idx] = rare_arr_for_i
                        outliers_6d_arr = [x + y for x, y in zip(outliers_6d_arr, outliers_6d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
            else:
                dt_display_prev = datetime.now()
                for i in range(num_cols):
                    if self.verbose and i > 0:
                        num_combinations_left = self.__get_num_combinations(dim=6, num_cols_processed=i)
                        percent_complete = (num_combinations - num_combinations_left) * 100.0 / num_combinations
                        dt_display = datetime.now()
                        if (dt_display - dt_display_prev) > timedelta(seconds=5):
                            print(f"  {percent_complete:.2f}% complete")
                            dt_display_prev = dt_display

                    rare_arr_for_i, outliers_6d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_6d(
                        self,
                        i,
                        self.data_np,
                        num_cols,
                        num_rows,
                        self.unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        rare_3d_values,
                        rare_4d_values,
                        rare_5d_values,
                        self.threshold
                    )
                    rare_6d_values[i] = rare_arr_for_i
                    outliers_6d_arr = [x + y for x, y in zip(outliers_6d_arr, outliers_6d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_6d_values)
            self.run_summary += f"\n\n6d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n6d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_6d_arr)
            return fractions_6d, rare_6d_values, outliers_6d_arr, outliers_explanation_arr, column_combos_checked

        def create_output_csv(outliers_1d_arr,
                              outliers_2d_arr,
                              outliers_3d_arr,
                              outliers_4d_arr,
                              outliers_5d_arr,
                              outliers_6d_arr,
                              explanations_1d_arr,
                              explanations_2d_arr,
                              explanations_3d_arr,
                              explanations_4d_arr,
                              explanations_5d_arr,
                              explanations_6d_arr):

            # todo: de-encode from the ordinal values in the explanations
            df = pd.DataFrame({"1d Counts": outliers_1d_arr,
                               "2d Counts": outliers_2d_arr,
                               "3d Counts": outliers_3d_arr,
                               "4d Counts": outliers_4d_arr,
                               "5d Counts": outliers_5d_arr,
                               "6d Counts": outliers_6d_arr,
                               "1d Explanations": explanations_1d_arr,
                               "2d Explanations": explanations_2d_arr,
                               "3d Explanations": explanations_3d_arr,
                               "4d Explanations": explanations_4d_arr,
                               "5d Explanations": explanations_5d_arr,
                               "6d Explanations": explanations_6d_arr,
                               })
            df['Any at 1d'] = df['1d Counts'] > 0
            df['Any at 2d'] = df['2d Counts'] > 0
            df['Any at 3d'] = df['3d Counts'] > 0
            df['Any at 4d'] = df['4d Counts'] > 0
            df['Any at 5d'] = df['5d Counts'] > 0
            df['Any at 6d'] = df['6d Counts'] > 0

            df['Any up to 1d'] = df['1d Counts'] > 0
            df['Any up to 2d'] = df['Any up to 1d'] | df['2d Counts'] > 0
            df['Any up to 3d'] = df['Any up to 2d'] | df['3d Counts'] > 0
            df['Any up to 4d'] = df['Any up to 3d'] | df['4d Counts'] > 0
            df['Any up to 5d'] = df['Any up to 4d'] | df['5d Counts'] > 0
            df['Any up to 6d'] = df['Any up to 5d'] | df['6d Counts'] > 0

            df['Any Scored'] = (df['1d Counts'] + df['2d Counts'] + df['3d Counts'] + df['4d Counts'] + df['5d Counts'] + df['6d Counts']) > 0

            if self.results_folder != "":
                os.makedirs(self.results_folder, exist_ok=True)
                n = datetime.now()
                dt_string = n.strftime("%d_%m_%Y_%H_%M_%S")
                #file_name = self.results_folder + "\\" + self.results_name + "_results_" + dt_string + ".csv"
                file_name = os.path.join(self.results_folder, f"{self.results_name}_results_{dt_string}.csv")
                df.to_csv(file_name)

            return df

        def create_return_dict():
            return {
                'Scores': self.flagged_rows_df['TOTAL SCORE'],
                'Breakdown All Rows': self.flagged_rows_df,
                'Breakdown Flagged Rows': self.__output_explanations(),
                'Flagged Summary': run_summary_df
            }

        # Copy the passed in data and ensure it is in pandas format. Keep both the original data and a format that
        # will modify the data, binning numeric columns and so on.
        self.orig_df = pd.DataFrame(input_data).copy()
        self.data_df = pd.DataFrame(input_data).copy()

        # Ensure the dataframe has a predictable index
        self.orig_df = self.orig_df.reset_index(drop=True)
        self.data_df = self.data_df.reset_index(drop=True)

        # Ensure the column names are strings
        self.orig_df.columns = [str(x) for x in self.orig_df.columns]
        self.data_df.columns = [str(x) for x in self.data_df.columns]

        # Get the column types and create a list of the numeric columns
        self.col_types_arr = self.__get_col_types_arr()
        self.numeric_col_names = [self.data_df.columns[x]
                                  for x in range(len(self.col_types_arr)) if self.col_types_arr[x] == 'N']

        # Remove any columns with too few (less than min_values_per_column) unique values or too many (more than
        # max_values_per_column) unique values.
        drop_col_names_arr = []
        for c in range(len(self.data_df.columns)):
            if self.col_types_arr[c] == 'C':
                if self.data_df[self.data_df.columns[c]].nunique() < self.min_values_per_column or \
                        self.data_df[self.data_df.columns[c]].nunique() > self.max_values_per_column:
                    drop_col_names_arr.append(input_data.columns[c])
        self.data_df = self.data_df.drop(columns=drop_col_names_arr)

        # Re-create a list of the numeric columns given some columns may have been removed
        self.col_types_arr = self.__get_col_types_arr()
        self.numeric_col_names = [self.data_df.columns[x]
                                  for x in range(len(self.col_types_arr)) if self.col_types_arr[x] == 'N']

        # Fill any null values with the mode/median value. Doing this ensures null values are not typically flagged.
        for col_idx, col_name in enumerate(self.data_df.columns):
            if self.col_types_arr[col_idx] == 'C':
                self.data_df[col_name] = self.data_df[col_name].fillna(self.data_df[col_name].mode())
            else:
                self.data_df[col_name] = self.data_df[col_name].fillna(self.data_df[col_name].median())

        # Fill any infinite values with the column maximum, and negative infinite values with the minimum
        for col_idx, col_name in enumerate(self.data_df.columns):
            if self.col_types_arr[col_idx] == 'N':
                self.data_df[col_name] = self.data_df[col_name].replace(-np.inf, self.data_df[col_name].min())
                self.data_df[col_name] = self.data_df[col_name].replace(np.inf, self.data_df[col_name].max())

        # Bin any numeric columns
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        if len(self.numeric_col_names):
            x_num = self.data_df[self.numeric_col_names]
            xt = est.fit_transform(x_num)
            self.bin_edges = est.bin_edges_
            for col_idx, col_name in enumerate(self.numeric_col_names):
                self.data_df[col_name] = xt[:, col_idx].astype(int)

        num_cols = len(self.data_df.columns)
        num_rows = len(self.data_df)

        self.run_summary = f"\nNumber of rows: {num_rows}"
        self.run_summary += f"\nNumber of columns: {num_cols}"

        # Create a summary of this run, giving statistics about the outliers found
        run_summary_df = pd.DataFrame(columns=[
            # Binary indicators giving the size of subspaces checked. These are False if there are too many
            # combinations to check
            'Checked_2d',
            'Checked_3d',
            'Checked_4d',
            'Checked_5d',
            'Checked_6d',

            'Percent Flagged as 1d',
            'Percent Flagged as 2d',
            'Percent Flagged as 3d',
            'Percent Flagged as 4d',
            'Percent Flagged as 5d',
            'Percent Flagged as 6d',

            'Percent Flagged up to 1d',
            'Percent Flagged up to 2d',
            'Percent Flagged up to 3d',
            'Percent Flagged up to 4d',
            'Percent Flagged up to 5d',
            'Percent Flagged up to 6d',

            'Percent Flagged'])

        if num_cols < 2:
            self.run_summary += \
                ("\nLess than two columns found (after dropping columns with too few or too many unique values. Cannot "
                 "determine outliers. Consider adjusting min_values_per_column or max_values_per_column")
            return create_return_dict()

        self.__ordinal_encode()
        self.data_np = self.data_df.values
        self.unique_vals, self.num_unique_vals, = get_unique_vals()
        self.run_summary += f"\nCardinality of the columns (after binning numeric columns): {self.num_unique_vals}"

        # Determine the 1d stats
        fractions_1d, rare_1d_values, outliers_1d_arr, explanations_1d_arr = get_1d_stats()
        self.rare_1d_values = rare_1d_values
        self.dimensions_examined = 1

        # Determine the 2d stats
        checked_2d = False
        outliers_2d_arr = [0] * num_rows
        explanations_2d_arr = [""] * num_rows
        if self.max_dimensions >= 2:
            num_combinations = self.__get_num_combinations(dim=2)
            if num_combinations > self.max_num_combinations:
                self.run_summary += (
                    f"\n\nCannot determine 2d outliers given the number of columns ({num_cols}) and number of unique "
                    f"values in each. Estimated number of combinations: {round(num_combinations):,}")
            else:
                fractions_2d, rare_2d_values, outliers_2d_arr, explanations_2d_arr = get_2d_stats()
                checked_2d = True
                self.rare_2d_values = rare_2d_values
                self.dimensions_examined = 2

        # Determine the 3d stats unless there are too many columns and unique values to do so efficiently
        checked_3d = False
        outliers_3d_arr = [0] * num_rows
        explanations_3d_arr = [""] * num_rows
        if self.max_dimensions >= 3:
            num_combinations = self.__get_num_combinations(dim=3)
            if num_cols < 3:
                self.run_summary += f"\n\nCannot determine 3d outliers. Too few columns: {num_cols}."
            if num_combinations > self.max_num_combinations:
                self.run_summary += (
                    f"\n\nCannot determine 3d outliers given the number of columns ({num_cols}) and number of unique "
                    f"values in each. Estimated number of combinations: {round(num_combinations):,}")
            else:
                fractions_3d, rare_3d_values, outliers_3d_arr, explanations_3d_arr, column_combos_checked_3d = \
                    get_3d_stats(num_combinations=num_combinations)
                checked_3d = True
                self.dimensions_examined = 3

        # Determine the 4d stats unless there are too many columns and unique values to do so efficiently
        checked_4d = False
        outliers_4d_arr = [0] * num_rows
        explanations_4d_arr = [""] * num_rows
        if self.max_dimensions >= 4:
            num_combinations = self.__get_num_combinations(dim=4)
            if num_cols < 4:
                self.run_summary += f"\n\nCannot determine 4d outliers. Too few columns: {num_cols}."
            elif num_combinations > self.max_num_combinations:
                self.run_summary += \
                    (f"\n\nCannot determine 4d outliers given the number of columns ({num_cols}) and number of unique "
                     f"values in each. Estimated number of combinations: {round(num_combinations):,}")
            else:
                fractions_4d, rare_4d_values, outliers_4d_arr, explanations_4d_arr, column_combos_checked_4d = \
                    get_4d_stats(num_combinations=num_combinations)
                checked_4d = True
                self.dimensions_examined = 4

        # Determine the 5d stats unless there are too many columns and unique values to do so efficiently
        checked_5d = False
        outliers_5d_arr = [0] * num_rows
        explanations_5d_arr = [""] * num_rows
        if self.max_dimensions >= 5:
            num_combinations = self.__get_num_combinations(dim=5)
            if num_cols < 5:
                self.run_summary += f"\n\nCannot determine 5d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
            elif num_combinations > self.max_num_combinations:
                self.run_summary += (
                    f"\n\nCannot determine 5d outliers given the number of columns ({num_cols}) and number of unique "
                    f"values in each. Estimated number of combinations: {round(num_combinations):,}")
            else:
                fractions_5d, rare_5d_values, outliers_5d_arr, explanations_5d_arr, column_combos_checked_5d = \
                    get_5d_stats(num_combinations=num_combinations)
                checked_5d = True
                self.dimensions_examined = 5

        # Determine the 6d stats unless there are too many columns and unique values to do so efficiently
        checked_6d = False
        outliers_6d_arr = [0] * num_rows
        explanations_6d_arr = [""] * num_rows
        if self.max_dimensions >= 6:
            num_combinations = self.__get_num_combinations(dim=6)
            if num_cols < 6:
                self.run_summary += f"\n\nCannot determine 6d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
            elif num_combinations > self.max_num_combinations:
                self.run_summary += (
                    f"\n\nCannot determine 6d outliers given the number of columns ({num_cols}) and number of unique "
                    f"values in each. Estimated number of combinations: {round(num_combinations):,}")
            else:
                fractions_6d, rare_6d_values, outliers_6d_arr, explanations_6d_arr, column_combos_checked_6d = \
                    get_6d_stats(num_combinations=num_combinations)
                checked_6d = True
                self.dimensions_examined = 6

        # Create the self.flagged_rows_df dataframe and, if specified, save to disk
        self.flagged_rows_df = create_output_csv(
            outliers_1d_arr, outliers_2d_arr, outliers_3d_arr, outliers_4d_arr, outliers_5d_arr, outliers_6d_arr,
            explanations_1d_arr, explanations_2d_arr, explanations_3d_arr, explanations_4d_arr, explanations_5d_arr,
            explanations_6d_arr)

        # Add a TOTAL SCORE column to self.flagged_rows_df
        col_names = []
        for dim in range(self.max_dimensions + 1):
            col_name = f'{dim}d Counts'
            if col_name in self.flagged_rows_df.columns:
                col_names.append(col_name)
        self.flagged_rows_df['TOTAL SCORE'] = self.flagged_rows_df[col_names].sum(axis=1)

        # Generate the final text included in self.run_summary, which summarizes the numbers of flagged values or
        # combinations of values in each dimensionality considered.
        self.run_summary += "\n"
        for dim in range(1, self.max_dimensions + 1):
            num_rows_scored = list(self.flagged_rows_df[f'Any at {dim}d'] > 0).count(True)
            self.run_summary += (
                f"\nNumber of rows flagged as outliers examining {dim}d: {num_rows_scored:3}"
                f" ({round(num_rows_scored*100.0/num_rows, 3)}%)")

        # Fill in flagged_rows_df
        run_summary_df = run_summary_df.append(pd.DataFrame(np.array([[
            checked_2d,
            checked_3d,
            checked_4d,
            checked_5d,
            checked_6d,

            self.flagged_rows_df['Any at 1d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 2d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 3d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 4d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 5d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 6d'].sum() * 100.0 / num_rows,

            self.flagged_rows_df['Any up to 1d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 2d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 3d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 4d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 5d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 6d'].sum() * 100.0 / num_rows,

            self.flagged_rows_df['Any Scored'].sum() * 100.0 / num_rows
            ]]),
            columns=run_summary_df.columns))

        # Set the correct column types
        run_summary_df['Checked_2d'] = run_summary_df['Checked_2d'].astype(bool)
        run_summary_df['Checked_3d'] = run_summary_df['Checked_3d'].astype(bool)
        run_summary_df['Checked_4d'] = run_summary_df['Checked_4d'].astype(bool)
        run_summary_df['Checked_5d'] = run_summary_df['Checked_5d'].astype(bool)
        run_summary_df['Checked_6d'] = run_summary_df['Checked_6d'].astype(bool)

        return create_return_dict()

    def get_most_flagged_rows(self):
        """
        This is used to get the rows from the original data with the highest total scores.

        :return: pandas dataframe
            This returns a dataframe with the set of rows matching all rows from the original data that received any
            non-zero score, ordered from highest score to lowest. This has the full set of columns from the original
            data as well as a SCORE column indicating the total score of the column
        """
        if self.flagged_rows_df is None or self.flagged_rows_df.empty:
            print("No rows were flagged.")
            return

        index_df = (self.flagged_rows_df[self.flagged_rows_df['TOTAL SCORE'] > 0]
                    .copy()
                    .sort_values('TOTAL SCORE', ascending=False))
        ret_df = self.data_df.loc[index_df.index].copy()
        ret_df.insert(0, 'TOTAL SCORE', index_df['TOTAL SCORE'])
        if ret_df.empty:
            print("No rows were flagged.")
        return ret_df

    def plot_scores_distribution(self):
        # Ensure there are no missing values on the x-axis
        scores_arr = self.flagged_rows_df['TOTAL SCORE'].value_counts()
        scores_counts = [(x, y) for x, y in zip(scores_arr.index, scores_arr.values)]
        scores_missing = [(x, 0) for x in range(max(scores_arr.index)) if x not in scores_arr.index]
        scores_counts.extend(scores_missing)
        scores_counts = list(zip(*sorted(scores_counts, key=lambda x: x[0])))
        s = sns.barplot(x=pd.Series(scores_counts[0]), y=pd.Series(scores_counts[1]))

        s.set_title("Distribution of Final Scores by Row Count")
        skip_x_ticks(s)
        plt.show()

        if scores_counts[0][0] == 0:
            scores_counts[0] = scores_counts[0][1:]
            scores_counts[1] = scores_counts[1][1:]
            s = sns.barplot(x=pd.Series(scores_counts[0]), y=pd.Series(scores_counts[1]))
            s.set_title("Distribution of Final Scores by Row Count (Excluding Zero)")
            skip_x_ticks(s)
            plt.show()

        scores_arr = self.flagged_rows_df['TOTAL SCORE'].sort_values()
        s = sns.scatterplot(x=range(len(scores_arr)), y=scores_arr)
        s.set_title("Scores Sorted Lowest to Highest")
        plt.show()

    def print_run_summary(self):
        print(self.run_summary)

    def explain_row(self, row_index, max_plots=50):
        def print_outlier_1d(explanation):
            column_name = explanation[0][0]
            value = explanation[1][0]
            col_idx = self.data_df.columns.tolist().index(column_name)

            title = f"Unusual value in column: {column_name}"
            if is_notebook():
                display(Markdown(f'### {title}'))
            else:
                print(title)

            if self.col_types_arr[col_idx] == 'C':
                pass  # todo: fill in!
            else:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

                s = sns.distplot(x=self.orig_df[column_name], ax=ax[0])
                s.set_title(f"Distribution of actual values in {column_name}")

                # Add a green vertical line for the bin boundaries
                for i in range(self.n_bins):
                    bin_edge = self.bin_edges[col_idx][i]
                    s.axvline(bin_edge, color='green', linewidth=0.5)

                for col_val_idx, col_val in enumerate(self.unique_vals[col_idx]):
                    if self.rare_1d_values[col_idx][col_val_idx]:
                        rows_matching = np.where(self.data_np[:, col_idx] == col_val)[0]
                        for r in rows_matching:
                            orig_value = self.orig_df.loc[r, column_name]
                            s.axvline(orig_value, color='brown', linewidth=0.5)

                # Add a thick red line for the current row
                orig_value = self.orig_df.loc[row_index, column_name]
                s.axvline(orig_value, color='red', linewidth=2.0)
                s.set_xlabel((f"{column_name} \n"
                              f"Row {row_index}: thick red line\n"
                              f"Bin edges in green, other flagged values in red"))

                counts_arr = [len(self.data_df[self.data_df[column_name] == x]) for x in self.unique_vals[col_idx]]
                counts_arr = [max(x, max(counts_arr)/100.0) for x in counts_arr]
                cols = ['blue'] * len(counts_arr)
                i = self.unique_vals[col_idx].index(int(value.replace('Bin ', '')))
                cols[i] = 'red'
                s = sns.barplot(
                    x=pd.Series(self.unique_vals[col_idx]),
                    y=pd.Series(counts_arr),
                    palette=cols,
                )
                s.set_title(f"Distribution of binned values in {column_name}")
                s.set_xlabel(f"{column_name} (binned)")
                plt.show()

        def print_outlier_2d(explanation):
            column_name_1 = explanation[0][0]
            column_name_2 = explanation[0][1]
            value_1 = explanation[1][0]
            value_2 = explanation[1][1]
            col_idx_1 = self.data_df.columns.tolist().index(column_name_1)
            col_idx_2 = self.data_df.columns.tolist().index(column_name_2)

            if self.col_types_arr[col_idx_1] == 'N':
                unique_vals_1 = list(range(0, self.n_bins))
            else:
                unique_vals_1 = sorted(self.data_df[column_name_1].unique())
                orig_vals_1 = self.ordinal_encoders_arr[col_idx_1].\
                    inverse_transform(np.array([unique_vals_1]).reshape(-1, 1)).reshape(1, -1)[0].tolist()

            if self.col_types_arr[col_idx_2] == 'N':
                unique_vals_2 = sorted(range(0, self.n_bins), reverse=True)
            else:
                unique_vals_2 = sorted(self.data_df[column_name_2].unique(), reverse=True)
                orig_vals_2 = self.ordinal_encoders_arr[col_idx_2].\
                    inverse_transform(np.array([unique_vals_2]).reshape(-1, 1)).reshape(1, -1)[0].tolist()

            title = (f"Unusual values in column: {column_name_1} ({self.orig_df.loc[row_index, column_name_1]}) and "
                     f"in column: {column_name_2}: ({self.orig_df.loc[row_index, column_name_2]})")
            if is_notebook():
                display(Markdown(f'### {title}'))
            else:
                print(title)

            # If both columns are categorical, display a heatmap
            if self.col_types_arr[col_idx_1] == 'C' and self.col_types_arr[col_idx_2] == 'C':
                counts_arr = []
                for v2 in unique_vals_2:
                    row_arr = []
                    for v1 in unique_vals_1:
                        row_arr.append(len(self.data_df[(self.data_df[column_name_1] == v1) &
                                                        (self.data_df[column_name_2] == v2)]))
                    counts_arr.append(row_arr)
                counts_df = pd.DataFrame(counts_arr, columns=orig_vals_1, index=orig_vals_2)

                orig_cmap = matplotlib.cm.RdBu
                if self.shifted_cmap is None:
                    self.shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.05, name='shifted')
                fig, ax = plt.subplots(figsize=(max(5, len(unique_vals_1) * 0.6), max(4, len(unique_vals_2) * 0.3)))
                s = sns.heatmap(
                    data=counts_df,
                    annot=True,
                    fmt="d",
                    cmap=self.shifted_cmap,
                    mask=(counts_df == 0),
                    linecolor='green',
                    linewidth=.5,
                    cbar=False)
                s.set_title(f"Distribution of Values in Columns\n{column_name_1} and \n{column_name_2}")
                s.set_xlabel(column_name_1)
                s.set_ylabel(column_name_2)

                # Ensure the frame is visible
                for _, spine in s.spines.items():
                    spine.set_visible(True)

                # Highlight the cell for the current row
                cell_x = orig_vals_1.index(value_1)
                cell_y = orig_vals_2.index(value_2)
                s.add_patch(Rectangle((cell_x, cell_y), 1, 1, fill=False, edgecolor='blue', lw=3))

                plt.tight_layout()
                plt.show()

            # If one column is categorical and one numeric, display a strip plot
            elif ((self.col_types_arr[col_idx_1] == 'C') and (self.col_types_arr[col_idx_2] == 'N') or
                  (self.col_types_arr[col_idx_1] == 'N') and (self.col_types_arr[col_idx_2] == 'C')):
                hue_array = ['Unflagged'] * len(self.orig_df)
                hue_array[row_index] = 'Flagged'
                s = sns.stripplot(
                    x=self.orig_df[column_name_1],
                    y=self.orig_df[column_name_2],
                    hue=hue_array,
                    palette=['blue', 'red'],
                    jitter=True,
                    dodge=True)
                s.set_title(f"Distribution of Values in Columns\n{column_name_1} and \n{column_name_2}")
                s.set_xlabel(column_name_1)
                s.set_ylabel(column_name_2)
                if len("".join([x.get_text() for x in s.get_xticklabels()])) > 20:
                    plt.xticks(rotation=70)
                plt.show()

            # If both columns are numeric, display a scatter plot and a heatmap of the bins
            else:
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4), gridspec_kw={'width_ratios': [1, 1]})

                s = sns.scatterplot(x=self.orig_df[column_name_1].astype(float), y=self.orig_df[column_name_2].astype(float), ax=ax[0])
                s.set_title(f"Distribution of Actual Values in Columns\n{column_name_1} and \n{column_name_2}")
                numeric_col_idx_1 = self.numeric_col_names.index(column_name_1)
                numeric_col_idx_2 = self.numeric_col_names.index(column_name_2)
                s.set_xlim(self.bin_edges[numeric_col_idx_1][0], self.bin_edges[numeric_col_idx_1][-1])
                s.set_ylim(self.bin_edges[numeric_col_idx_2][0], self.bin_edges[numeric_col_idx_2][-1])
                xlim = s.get_xlim()
                ylim = s.get_ylim()

                # Add green grid lines for the bin boundaries
                for i in range(self.n_bins):
                    bin_edge = self.bin_edges[numeric_col_idx_1][i]
                    s.axvline(bin_edge, color='green', linewidth=0.5)
                for i in range(self.n_bins):
                    bin_edge = self.bin_edges[numeric_col_idx_2][i]
                    s.axhline(bin_edge, color='green', linewidth=0.5)

                # Add a red vertical dot for each flagged value
                flags = self.rare_2d_values[col_idx_1][col_idx_2]
                for flags_i in range(len(flags)):
                    for flags_j in range(len(flags[flags_i])):
                        if flags[flags_i][flags_j]:
                            flagged_value_1 = self.unique_vals[col_idx_1][flags_i]
                            flagged_value_2 = self.unique_vals[col_idx_2][flags_j]
                            sub_df = self.data_df[
                                (self.data_df[column_name_1] == int(flagged_value_1)) &
                                (self.data_df[column_name_2] == int(flagged_value_2))
                            ]
                            for i in sub_df.index:
                                orig_value_1 = self.orig_df.loc[i, column_name_1]
                                orig_value_2 = self.orig_df.loc[i, column_name_2]
                                s = sns.scatterplot(x=[orig_value_1], y=[orig_value_2], color='r', ax=ax[0])
                                s.set_xlim(xlim)
                                s.set_ylim(ylim)

                # Add yellow dots for the values flagged as 1d outliers
                for col_idx in [col_idx_1, col_idx_2]:
                    for col_val_idx, col_val in enumerate(self.unique_vals[col_idx]):
                        if self.rare_1d_values[col_idx][col_val_idx]:
                            rows_matching = np.where(self.data_np[:, col_idx] == col_val)[0]
                            for r in rows_matching:
                                orig_value_1 = self.orig_df.loc[r, column_name_1]
                                orig_value_2 = self.orig_df.loc[r, column_name_2]
                                s = sns.scatterplot(x=[orig_value_1], y=[orig_value_2], color='gold', ax=ax[0])
                                s.set_xlim(xlim)
                                s.set_ylim(ylim)

                # Add a star for the point representing the current row
                orig_value_1 = float(self.orig_df.loc[row_index, column_name_1])
                orig_value_2 = float(self.orig_df.loc[row_index, column_name_2])
                s = sns.scatterplot(x=[orig_value_1], y=[orig_value_2], color='darkred', marker='*', s=300, ax=ax[0])
                s.set_xlim(xlim)
                s.set_ylim(ylim)

                s.set_xlabel((f'{column_name_1}\nRow {row_index} represented as star. \nOther 2d anomalies for this '
                              f'pair of features as red dots. \n1d anomalies as yellow dots.'))

                counts_arr = []
                for v2 in unique_vals_2:
                    row_arr = []
                    for v1 in unique_vals_1:
                        row_arr.append(len(self.data_df[(self.data_df[column_name_1] == v1) &
                                                        (self.data_df[column_name_2] == v2)]))
                    counts_arr.append(row_arr)
                counts_df = pd.DataFrame(counts_arr, columns=unique_vals_1, index=unique_vals_2)

                orig_cmap = matplotlib.cm.RdBu
                if self.shifted_cmap is None:
                    self.shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.05, name='shifted')
                s = sns.heatmap(
                    data=counts_df,
                    annot=True,
                    fmt="d",
                    cmap=self.shifted_cmap,
                    mask=(counts_df == 0),
                    linecolor='green',
                    linewidth=.5,
                    cbar=False,
                    ax=ax[1])
                s.set_title(f"Distribution of Binned Values in Columns\n{column_name_1} and \n{column_name_2}")
                s.set_xlabel(column_name_1)
                s.set_ylabel(column_name_2)
                heatmap_xlim = s.get_xlim()
                heatmap_ylim = s.get_ylim()

                # Ensure the frame is visible
                for _, spine in s.spines.items():
                    spine.set_visible(True)

                # Highlight the cell for the current row
                cell_x = -1
                for bin_idx in range(len(self.bin_edges[numeric_col_idx_1])):
                    if orig_value_1 <= self.bin_edges[numeric_col_idx_1][bin_idx + 1]:
                        cell_x = bin_idx
                        break
                cell_y = -1
                for bin_j in range(len(self.bin_edges[numeric_col_idx_1])):
                    if orig_value_2 <= self.bin_edges[numeric_col_idx_2][bin_j + 1]:
                        cell_y = self.n_bins - bin_j - 1
                        break
                s.add_patch(Rectangle((cell_x, cell_y), 1, 1, fill=False, edgecolor='blue', lw=3))

                # Draw the current row as a star
                s = sns.scatterplot(
                    x=[(orig_value_1 - xlim[0]) / (xlim[1] - xlim[0]) * (heatmap_xlim[1] - heatmap_xlim[0])],
                    y=[(orig_value_2 - ylim[1]) / (ylim[0] - ylim[1]) * (heatmap_ylim[0] - heatmap_ylim[1])],
                    color='yellow', marker='*', s=150, zorder=np.inf, ax=ax[1])

                s.set_xlabel(f'{column_name_1}\nRow {row_index} represented as star')

                plt.tight_layout()
                plt.show()

        def print_outlier_multi_d(explanation):
            cols = explanation[0]

            title = "Unusual values in "
            for idx in range(len(cols)):
                title += f"column {cols[idx]} ({self.orig_df.loc[row_index, cols[idx]]}), "
            title = title[:-2]
            if is_notebook():
                display(Markdown(f'### {title}'))
            else:
                print(title)

            # Get the set of all combinations of the values in this set of columns
            vals_arr = []
            for col_name in cols:
                col_idx = self.data_df.columns.tolist().index(col_name)
                vals_arr.append(self.unique_vals[col_idx])
            combinations_arr = list(itertools.product(*vals_arr))

            # Get the counts of each combination
            counts_arr = []
            for combination in combinations_arr:
                cond = [True] * len(self.data_df)
                for idx in range(len(cols)):
                    col_name = cols[idx]
                    col_idx = self.data_df.columns.tolist().index(col_name)
                    cond = cond & (self.data_np[:, col_idx] == combination[idx])
                rows_all = np.where(cond)[0]
                if len(rows_all):
                    counts_arr.append(len(rows_all))

            # Get the count of the current values
            cond = [True] * len(self.data_df)
            for idx in range(len(cols)):
                col_name = cols[idx]
                col_idx = self.data_df.columns.tolist().index(col_name)
                cond = cond & (self.data_np[:, col_idx] == self.data_df.loc[row_index, col_name])
            curr_count = len(np.where(cond)[0])

            counts_arr.sort(reverse=True)

            # Ensure each count is tall enough to be seen
            min_val = counts_arr[0] / 100.0
            counts_arr = [x + min_val for x in counts_arr]

            # Draw the histogram / bar plot
            if len(counts_arr) < 100:
                bar_colors = ['steelblue'] * len(counts_arr)
                bar_colors[counts_arr.index(curr_count + min_val)] = 'red'
                s = sns.barplot(x=list(range(len(counts_arr))), y=counts_arr, palette=bar_colors)
            else:
                s = sns.histplot(counts_arr, color='blue')
            s.set_title(f"Counts by combination of values for \n{cols[0]}, \n{cols[1]}, \n{cols[2]}")
            s.set_xticks([])
            plt.show()

        if self.flagged_rows_df is None:
            print("No results can be displayed")
            return

        if row_index < 0 or row_index > len(self.data_df):
            print("Invalid row index")
            return

        if is_notebook():
            display(Markdown(f"**Explanation row number**: {row_index}"))
        else:
            print(f"Explanation row number: {row_index}")

        row = self.flagged_rows_df.iloc[row_index]
        if is_notebook():
            display(Markdown(f"**Total score**: {row['TOTAL SCORE']}"))
        else:
            print(f"Total score: {row['TOTAL SCORE']}")
        if row['TOTAL SCORE'] > max_plots:
            print(f"Displaying first {max_plots} plots")

        num_plots = 0
        for dim in range(self.max_dimensions + 1):
            col_name = f'{dim}d Explanations'
            if col_name not in self.flagged_rows_df.columns:
                continue
            if row[col_name] == "":
                continue
            expl_arr = row[col_name]
            if dim == 1:
                for expl in expl_arr:
                    num_plots += 1
                    if num_plots > max_plots:
                        return
                    print_outlier_1d(expl)
            elif dim == 2:
                for expl in expl_arr:
                    num_plots += 1
                    if num_plots > max_plots:
                        return
                    print_outlier_2d(expl)
            else:
                for expl in expl_arr:
                    num_plots += 1
                    if num_plots > max_plots:
                        return
                    print_outlier_multi_d(expl)

    def explain_features(self, features_arr):
        """
        Display the counts for each combination of values within a specified set of columns. This would typically be
        called to follow up a call to explain_row(), where a set of 3 or more features were identified with at least
        one unusual combination of values.

        :param features_arr: list of strings
            A list of features within the dataset

        :return: None
        """

        for column_name in features_arr:
            if column_name not in self.orig_df.columns:
                print(f"{column_name} not in the dataframe used for outlier detection.")
                return
            if column_name not in self.data_df.columns:
                print(f"{column_name} was not used for outlier detection.")
                return

        vals_arr = []
        for col_name in features_arr:
            col_idx = self.data_df.columns.tolist().index(col_name)
            vals_arr.append(self.unique_vals[col_idx])
        combinations_arr = list(itertools.product(*vals_arr))

        # Get the counts of each combination
        counts_arr = []
        for combination in combinations_arr:
            col_idx_arr = []
            cond = [True] * len(self.data_df)
            for idx in range(len(features_arr)):
                col_name = features_arr[idx]
                col_idx = self.data_df.columns.tolist().index(col_name)
                col_idx_arr.append(col_idx)
                cond = cond & (self.data_np[:, col_idx] == combination[idx])
            rows_all = np.where(cond)[0]
            if len(rows_all):
                row = []
                for c_idx, c in enumerate(combination):
                    row.append(self._get_col_value(col_idx_arr[c_idx], c))
                row.append(len(rows_all))
                counts_arr.append(row)

        # Display the counts
        counts_df = pd.DataFrame(counts_arr, columns=features_arr + ['Count'])
        counts_df = counts_df.sort_values('Count', ascending=False)
        if is_notebook():
            display(counts_df)
        else:
            print(counts_df)

    def __output_explanations(self):
        """
        Given self.flagged_rows_df, a dataframe with the full information about flagged rows, return a smaller, simpler
        dataframe to summarize this concisely. self.flagged_df should have a row for each row in the original data, and
        a set of columns summarizing the outlier combinations found in each row.

        :return: A dataframe with a row for each row in the original data where at least one combination of values was
            flagged.
        """

        if self.flagged_rows_df is None or self.flagged_rows_df.empty:
            return None

        df_subset = self.flagged_rows_df[self.flagged_rows_df['Any Scored']].copy()
        expl_arr = []
        index_arr = list(df_subset.index)
        for i in range(len(df_subset)):
            row = df_subset.iloc[i]
            row_expl = [index_arr[i], "", "", "", "", "", ""][:self.dimensions_examined+1]
            for j in range(1, self.dimensions_examined+1):
                col_name = f"{j}d Explanations"
                row_expl[j] = row[col_name]
            expl_arr.append(row_expl)
        expl_df = pd.DataFrame(
            expl_arr,
            columns=['Row Index', '1d Explanations', '2d Explanations', '3d Explanations',
                     '4d Explanations', '5d Explanations', '6d Explanations'][:self.dimensions_examined+1])
        return expl_df

    def __get_col_types_arr(self):
        """
        Create an array representing each column of the data, with each coded as 'C' (categorical) or 'N' (numeric).
        """

        col_types_arr = ['N'] * len(self.data_df.columns)

        for col_idx, col_name in enumerate(self.data_df.columns):
            num_unique = self.data_df[col_name].nunique()
            is_numeric = self.__get_is_numeric(col_name)
            if not is_numeric:
                col_types_arr[col_idx] = 'C'
            # Even where the values are numeric, if there are few of them, consider them categorical, though if the
            # values are all float, the column will be cast to 'N' when collecting the unique values. We want at
            # least, on average, 2 values per bin.
            if num_unique <= (2 * self.n_bins):
                col_types_arr[col_idx] = 'C'

        # # If there are a large number of categorical columns, re-determine the types with a more strict cutoff
        # if col_types_arr.count('C') > 50:
        #     col_types_arr = ['N'] * len(self.data_df.columns)
        #     for col_idx, col_name in enumerate(self.data_df.columns):
        #         num_unique = self.data_df[col_name].nunique()
        #         is_numeric = self.__get_is_numeric(col_name)
        #         if not is_numeric:
        #             col_types_arr[col_idx] = 'C'
        #         if num_unique <= min(5, self.n_bins):
        #             col_types_arr[col_idx] = 'C'

        # Ensure any numeric columns are stored in integer or float format
        for col_idx, col_name in enumerate(self.data_df.columns):
            if col_types_arr[col_idx] == 'N':
                if not self.data_df[col_name].dtype in [int, np.int64]:
                    self.data_df[col_name] = self.data_df[col_name].astype(float)

        return col_types_arr

    def __ordinal_encode(self):
        """
        numpy deals with numeric values much more efficiently than text. We convert any categorical columns to ordinal
        values based on either their original values or bin ids.
        """

        self.ordinal_encoders_arr = [None]*len(self.data_df.columns)
        for col_idx, col_name in enumerate(self.data_df.columns):
            if self.col_types_arr[col_idx] == 'C':
                enc = OrdinalEncoder()
                self.ordinal_encoders_arr[col_idx] = enc
                x_np = enc.fit_transform(self.orig_df[col_name].astype(str).values.reshape(-1, 1)).reshape(1, -1)[0]
                self.data_df[col_name] = self.data_df[col_name].astype(str)
                self.data_df[col_name] = x_np
                self.data_df[col_name] = self.data_df[col_name].astype(int)
        return self.data_df

    def _get_col_value(self, col_idx, value_idx):
        """
        Return the original value of a specified column and ordinal value. For binned numeric columns, there is a
        range of values in each bin, and so the bin id is returned to approximate the original value.
        """
        if self.col_types_arr[col_idx] == "C":
            return self.ordinal_encoders_arr[col_idx].inverse_transform([[value_idx]])[0][0]
        else:
            return self.bin_names[value_idx]

    def __get_num_combinations(self, dim, num_cols_processed=None):
        avg_num_unique_vals = mean([len(x) for x in self.unique_vals])
        num_cols = len(self.data_df.columns)
        if num_cols_processed is not None:
            num_cols -= num_cols_processed
        num_combinations = math.comb(num_cols, dim) * pow(avg_num_unique_vals, 2)
        return num_combinations

    def __get_is_numeric(self, col_name):
        is_numeric = is_numeric_dtype((self.data_df[col_name])) or \
                     (self.orig_df[col_name]
                      .astype(str).str.replace('-', '', regex=False)
                      .str.replace('.', '', regex=False)
                      .str.isdigit()
                      .tolist()
                      .count(False) == 0)
        return is_numeric


#####################################################################################################################
# Methods to examine specific subspaces. These methods are outside the class in order to be callable as concurrent
# processes

def process_inner_loop_3d(
        obj,
        i,
        data_np,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values,
        divisor):

    """
    todo: fill in
    :param obj:
    :param i:
    :param data_np:
    :param num_cols:
    :param num_rows:
    :param unique_vals:
    :param fractions_1d:
    :param rare_1d_values:
    :param rare_2d_values:
    :param divisor:
    :return:
    """

    num_unique_vals_i = len(unique_vals[i])
    outliers_3d_arr_for_i = [0] * num_rows
    outliers_explanation_arr_for_i = [[]] * num_rows
    column_combos_checked_for_i = 0

    rare_arr_for_i = [[[] for _ in range(num_cols)] for _ in range(num_cols)]

    for j in range(i + 1, num_cols - 1):
        num_unique_vals_j = len(unique_vals[j])
        for k in range(j + 1, num_cols):
            num_unique_vals_k = len(unique_vals[k])

            expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k]))
            expected_count_under_uniform = num_rows * expected_under_uniform
            if expected_count_under_uniform < 10:
                continue
            column_combos_checked_for_i += 1

            local_rare_arr = [[[False
                                for _ in range(num_unique_vals_k)]
                               for _ in range(num_unique_vals_j)]
                              for _ in range(num_unique_vals_i)]
            for i_vals_idx in range(num_unique_vals_i):
                if rare_1d_values[i][i_vals_idx]:
                    continue
                i_val = unique_vals[i][i_vals_idx]
                cond1 = (data_np[:, i] == i_val)
                for j_vals_idx in range(num_unique_vals_j):
                    if rare_1d_values[j][j_vals_idx]:
                        continue
                    if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                        continue
                    j_val = unique_vals[j][j_vals_idx]
                    cond2 = (data_np[:, j] == j_val)
                    for k_vals_idx in range(num_unique_vals_k):
                        if rare_1d_values[k][k_vals_idx]:
                            continue
                        if rare_2d_values[i][k][i_vals_idx][k_vals_idx]:
                            continue
                        if rare_2d_values[j][k][j_vals_idx][k_vals_idx]:
                            continue
                        k_val = unique_vals[k][k_vals_idx]
                        cond3 = (data_np[:, k] == k_val)
                        rows_all = np.where(cond1 & cond2 & cond3)
                        current_fraction = len(rows_all[0]) / num_rows
                        three_d_row_nums = rows_all[0]

                        if obj.check_marginal_probs:
                            expected_given_marginal = \
                                fractions_1d[i][i_vals_idx] * \
                                fractions_1d[j][j_vals_idx] * \
                                fractions_1d[k][k_vals_idx] * \
                                divisor
                        else:
                            expected_given_marginal = np.inf
                        rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                          (current_fraction < expected_given_marginal) and \
                                          (current_fraction < 0.01)
                        if rare_value_flag:
                            row_nums = three_d_row_nums  # todo: can remove some variables here
                            assert len(row_nums) == round(current_fraction * num_rows), \
                                (f"len of matching rows: {len(row_nums)}, fraction*num_rows: current_fraction*num_rows: " 
                                 f"{current_fraction * num_rows}")
                            for r in row_nums:
                                # todo: i doubt this is threadsafe
                                outliers_3d_arr_for_i[r] += 1
                                expl = [[obj.data_df.columns[i], obj.data_df.columns[j], obj.data_df.columns[k]],
                                        [obj._get_col_value(i, i_vals_idx),
                                         obj._get_col_value(j, j_vals_idx),
                                         obj._get_col_value(k, k_vals_idx)]]
                                if not outliers_explanation_arr_for_i[r]:
                                    outliers_explanation_arr_for_i[r] = [expl]
                                else:
                                    outliers_explanation_arr_for_i[r].append(expl)
                        local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx] = rare_value_flag
            rare_arr_for_i[j][k] = local_rare_arr
    return rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def process_inner_loop_4d(
        obj,
        i,
        data_np,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values,
        rare_3d_values,
        divisor):

    num_unique_vals_i = len(unique_vals[i])
    outliers_4d_arr_for_i = [0] * num_rows
    outliers_explanation_arr_for_i = [[]] * num_rows
    rare_arr_for_i = [[[[] for _ in range(num_cols)] for _ in range(num_cols)] for _ in range(num_cols)]
    column_combos_checked_for_i = 0

    for j in range(i+1, num_cols-2):
        num_unique_vals_j = len(unique_vals[j])
        for k in range(j+1, num_cols-1):
            num_unique_vals_k = len(unique_vals[k])
            for m in range(k+1, num_cols):
                num_unique_vals_m = len(unique_vals[m])

                expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k]) * len(unique_vals[m]))
                expected_count_under_uniform = num_rows * expected_under_uniform
                if expected_count_under_uniform < 10:
                    continue
                column_combos_checked_for_i += 1

                local_rare_arr = [[[[False
                                     for _ in range(num_unique_vals_m)]
                                    for _ in range(num_unique_vals_k)]
                                   for _ in range(num_unique_vals_j)]
                                  for _ in range(num_unique_vals_i)]
                for i_vals_idx in range(num_unique_vals_i):
                    if rare_1d_values[i][i_vals_idx]:
                        continue
                    i_val = unique_vals[i][i_vals_idx]
                    cond1 = (data_np[:, i] == i_val)
                    for j_vals_idx in range(num_unique_vals_j):
                        if rare_1d_values[j][j_vals_idx]:
                            continue
                        if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                            continue
                        j_val = unique_vals[j][j_vals_idx]
                        cond2 = (data_np[:, j] == j_val)
                        for k_vals_idx in range(num_unique_vals_k):
                            if rare_1d_values[k][k_vals_idx]:
                                continue
                            if rare_2d_values[i][k][i_vals_idx][k_vals_idx]:
                                continue
                            if rare_2d_values[j][k][j_vals_idx][k_vals_idx]:
                                continue
                            if rare_3d_values[i][j][k][i_vals_idx][j_vals_idx][k_vals_idx]:
                                continue
                            k_val = unique_vals[k][k_vals_idx]
                            cond3 = (data_np[:, k] == k_val)
                            for m_vals_idx in range(num_unique_vals_m):
                                if rare_1d_values[m][m_vals_idx]:
                                    continue
                                if rare_2d_values[i][m][i_vals_idx][m_vals_idx]:
                                    continue
                                if rare_2d_values[j][m][j_vals_idx][m_vals_idx]:
                                    continue
                                if rare_2d_values[k][m][k_vals_idx][m_vals_idx]:
                                    continue
                                if rare_3d_values[i][j][m][i_vals_idx][j_vals_idx][m_vals_idx]:
                                    continue
                                if rare_3d_values[i][k][m][i_vals_idx][k_vals_idx][m_vals_idx]:
                                    continue
                                if rare_3d_values[j][k][m][j_vals_idx][k_vals_idx][m_vals_idx]:
                                    continue
                                m_val = unique_vals[m][m_vals_idx]
                                cond4 = (data_np[:, m] == m_val)
                                rows_all = np.where(cond1 & cond2 & cond3 & cond4)
                                current_fraction = len(rows_all[0]) / num_rows
                                four_d_row_nums = rows_all[0]  # todo: use less variables

                                if obj.check_marginal_probs:
                                    expected_given_marginal = \
                                        fractions_1d[i][i_vals_idx] * \
                                        fractions_1d[j][j_vals_idx] * \
                                        fractions_1d[k][k_vals_idx] * \
                                        fractions_1d[m][m_vals_idx] * \
                                        divisor
                                else:
                                    expected_given_marginal = np.inf
                                rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                                  (current_fraction < expected_given_marginal) and \
                                                  (current_fraction < 0.01)
                                if rare_value_flag:
                                    row_nums = four_d_row_nums  # todo: can remove some variables here
                                    assert len(row_nums) == round(current_fraction * num_rows), \
                                        f"len of matching rows: {len(row_nums)}, " \
                                        f"fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                                    for r in row_nums:
                                        # todo: i doubt this is threadsafe
                                        outliers_4d_arr_for_i[r] += 1
                                        expl = [[obj.data_df.columns[i],
                                                 obj.data_df.columns[j],
                                                 obj.data_df.columns[k],
                                                 obj.data_df.columns[m]],
                                                [obj._get_col_value(i, i_vals_idx),
                                                 obj._get_col_value(j, j_vals_idx),
                                                 obj._get_col_value(k, k_vals_idx),
                                                 obj._get_col_value(m, m_vals_idx)]
                                                ]
                                        if not outliers_explanation_arr_for_i[r]:
                                            outliers_explanation_arr_for_i[r] = [expl]
                                        else:
                                            outliers_explanation_arr_for_i[r].append(expl)
                                local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx] = rare_value_flag
                rare_arr_for_i[j][k][m] = local_rare_arr.copy()
    return rare_arr_for_i, outliers_4d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def process_inner_loop_5d(
        obj,
        i,
        data_np,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values,
        rare_3d_values,
        rare_4d_values,
        divisor):

    num_unique_vals_i = len(unique_vals[i])
    outliers_5d_arr_for_i = [0] * num_rows
    outliers_explanation_arr_for_i = [[]] * num_rows
    rare_arr_for_i = [[[[[]
                         for _ in range(num_cols)]
                        for _ in range(num_cols)]
                       for _ in range(num_cols)]
                      for _ in range(num_cols)]
    column_combos_checked_for_i = 0

    for j in range(i+1, num_cols-3):
        num_unique_vals_j = len(unique_vals[j])
        for k in range(j+1, num_cols-2):
            num_unique_vals_k = len(unique_vals[k])
            for m in range(k+1, num_cols-1):
                num_unique_vals_m = len(unique_vals[m])
                for n in range(m+1, num_cols):
                    num_unique_vals_n = len(unique_vals[n])

                    expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k]) * len(unique_vals[m]) * len(unique_vals[n]))
                    expected_count_under_uniform = num_rows * expected_under_uniform
                    if expected_count_under_uniform < 10:
                        continue
                    column_combos_checked_for_i += 1

                    # local_rare_arr represents the current set of columns. It's a 5d array, with a dimension
                    # for each value.
                    local_rare_arr = [[[[[False
                                          for _ in range(num_unique_vals_n)]
                                         for _ in range(num_unique_vals_m)]
                                        for _ in range(num_unique_vals_k)]
                                       for _ in range(num_unique_vals_j)]
                                      for _ in range(num_unique_vals_i)]
                    for i_vals_idx in range(num_unique_vals_i):
                        if rare_1d_values[i][i_vals_idx]:
                            continue
                        i_val = unique_vals[i][i_vals_idx]
                        cond1 = (data_np[:, i] == i_val)
                        for j_vals_idx in range(num_unique_vals_j):
                            if rare_1d_values[j][j_vals_idx]:
                                continue
                            if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                                continue
                            j_val = unique_vals[j][j_vals_idx]
                            cond2 = (data_np[:, j] == j_val)
                            for k_vals_idx in range(num_unique_vals_k):
                                if rare_1d_values[k][k_vals_idx]:
                                    continue
                                if rare_2d_values[i][k][i_vals_idx][k_vals_idx]:
                                    continue
                                if rare_2d_values[j][k][j_vals_idx][k_vals_idx]:
                                    continue
                                if rare_3d_values[i][j][k][i_vals_idx][j_vals_idx][k_vals_idx]:
                                    continue
                                k_val = unique_vals[k][k_vals_idx]
                                cond3 = (data_np[:, k] == k_val)
                                for m_vals_idx in range(num_unique_vals_m):
                                    if rare_1d_values[m][m_vals_idx]:
                                        continue
                                    if rare_2d_values[i][m][i_vals_idx][m_vals_idx]:
                                        continue
                                    if rare_2d_values[j][m][j_vals_idx][m_vals_idx]:
                                        continue
                                    if rare_2d_values[k][m][k_vals_idx][m_vals_idx]:
                                        continue
                                    if rare_3d_values[i][j][m][i_vals_idx][j_vals_idx][m_vals_idx]:
                                        continue
                                    if rare_3d_values[i][k][m][i_vals_idx][k_vals_idx][m_vals_idx]:
                                        continue
                                    if rare_3d_values[j][k][m][j_vals_idx][k_vals_idx][m_vals_idx]:
                                        continue
                                    m_val = unique_vals[m][m_vals_idx]
                                    cond4 = (data_np[:, m] == m_val)
                                    for n_vals_idx in range(num_unique_vals_n):
                                        if rare_1d_values[n][n_vals_idx]:
                                            continue
                                        if rare_2d_values[i][n][i_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_2d_values[j][n][j_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_2d_values[k][n][k_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_2d_values[m][n][m_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_3d_values[i][j][n][i_vals_idx][j_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_3d_values[i][k][n][i_vals_idx][k_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_3d_values[i][m][n][i_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_3d_values[j][k][n][j_vals_idx][k_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_3d_values[j][m][n][j_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_3d_values[k][m][n][k_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_4d_values[i][j][k][m][i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx]:
                                            continue
                                        if rare_4d_values[i][j][k][n][i_vals_idx][j_vals_idx][k_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_4d_values[i][j][m][n][i_vals_idx][j_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_4d_values[i][k][m][n][i_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        if rare_4d_values[j][k][m][n][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        n_val = unique_vals[n][n_vals_idx]
                                        cond5 = (data_np[:, n] == n_val)

                                        rows_all = np.where(cond1 & cond2 & cond3 & cond4 & cond5)
                                        current_fraction = len(rows_all[0]) / num_rows
                                        five_d_row_nums = rows_all[0]  # todo: use less variables

                                        if obj.check_marginal_probs:
                                            expected_given_marginal = \
                                                fractions_1d[i][i_vals_idx] * \
                                                fractions_1d[j][j_vals_idx] * \
                                                fractions_1d[k][k_vals_idx] * \
                                                fractions_1d[m][m_vals_idx] * \
                                                fractions_1d[n][n_vals_idx] * \
                                                divisor
                                        else:
                                            expected_given_marginal = np.inf
                                        rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                                          (current_fraction < expected_given_marginal) and \
                                                          (current_fraction < 0.01)
                                        if rare_value_flag:
                                            row_nums = five_d_row_nums  # todo: can remove some variables here
                                            assert len(row_nums) == round(current_fraction * num_rows), \
                                                (f"len of matching rows: {len(row_nums)}, fraction*num_rows: " 
                                                 f"current_fraction*num_rows: {current_fraction * num_rows}")
                                            for r in row_nums:
                                                # todo: i doubt this is threadsafe
                                                outliers_5d_arr_for_i[r] += 1
                                                expl = [[obj.data_df.columns[i],
                                                         obj.data_df.columns[j],
                                                         obj.data_df.columns[k],
                                                         obj.data_df.columns[m],
                                                         obj.data_df.columns[n]],
                                                        [obj._get_col_value(i, i_vals_idx),
                                                         obj._get_col_value(j, j_vals_idx),
                                                         obj._get_col_value(k, k_vals_idx),
                                                         obj._get_col_value(m, m_vals_idx),
                                                         obj._get_col_value(n, n_vals_idx)]
                                                        ]
                                                if not outliers_explanation_arr_for_i[r]:
                                                    outliers_explanation_arr_for_i[r] = [expl]
                                                else:
                                                    outliers_explanation_arr_for_i[r].append(expl)

                                        local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx] = rare_value_flag
                    rare_arr_for_i[j][k][m][n] = local_rare_arr
    return rare_arr_for_i, outliers_5d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def process_inner_loop_6d(
        obj,
        i,
        data_np,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values,
        rare_3d_values,
        rare_4d_values,
        rare_5d_values,
        divisor):

    num_unique_vals_i = len(unique_vals[i])
    outliers_6d_arr_for_i = [0] * num_rows
    outliers_explanation_arr_for_i = [[]] * num_rows
    rare_arr_for_i = [[[[[[]
                          for _ in range(num_cols)]
                         for _ in range(num_cols)]
                        for _ in range(num_cols)]
                       for _ in range(num_cols)]
                      for _ in range(num_cols)]
    column_combos_checked_for_i = 0

    for j in range(i+1, num_cols-4):
        num_unique_vals_j = len(unique_vals[j])
        for k in range(j+1, num_cols-3):
            num_unique_vals_k = len(unique_vals[k])
            for m in range(k+1, num_cols-2):
                num_unique_vals_m = len(unique_vals[m])
                for n in range(m+1, num_cols-1):
                    num_unique_vals_n = len(unique_vals[n])
                    for p in range(n+1, num_cols):
                        num_unique_vals_p = len(unique_vals[p])

                        expected_under_uniform = 1.0 / \
                            (len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k]) * len(unique_vals[m]) *
                             len(unique_vals[n]) * len(unique_vals[p]))
                        expected_count_under_uniform = num_rows * expected_under_uniform
                        if expected_count_under_uniform < 10:
                            continue
                        column_combos_checked_for_i += 1

                        # local_rare_arr represents the current set of columns. It's a 5d array, with a dimension
                        # for each value.
                        local_rare_arr = [[[[[[False
                                              for _ in range(num_unique_vals_p)]
                                             for _ in range(num_unique_vals_n)]
                                            for _ in range(num_unique_vals_m)]
                                           for _ in range(num_unique_vals_k)]
                                          for _ in range(num_unique_vals_j)]
                                         for _ in range(num_unique_vals_i)]
                        for i_vals_idx in range(num_unique_vals_i):
                            if rare_1d_values[i][i_vals_idx]:
                                continue
                            i_val = unique_vals[i][i_vals_idx]
                            cond1 = (data_np[:, i] == i_val)
                            for j_vals_idx in range(num_unique_vals_j):
                                if rare_1d_values[j][j_vals_idx]:
                                    continue
                                if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                                    continue
                                j_val = unique_vals[j][j_vals_idx]
                                cond2 = (data_np[:, j] == j_val)
                                for k_vals_idx in range(num_unique_vals_k):
                                    if rare_1d_values[k][k_vals_idx]:
                                        continue
                                    if rare_2d_values[i][k][i_vals_idx][k_vals_idx]:
                                        continue
                                    if rare_2d_values[j][k][j_vals_idx][k_vals_idx]:
                                        continue
                                    if rare_3d_values[i][j][k][i_vals_idx][j_vals_idx][k_vals_idx]:
                                        continue
                                    k_val = unique_vals[k][k_vals_idx]
                                    cond3 = (data_np[:, k] == k_val)
                                    for m_vals_idx in range(num_unique_vals_m):
                                        if rare_1d_values[m][m_vals_idx]:
                                            continue
                                        if rare_2d_values[i][m][i_vals_idx][m_vals_idx]:
                                            continue
                                        if rare_2d_values[j][m][j_vals_idx][m_vals_idx]:
                                            continue
                                        if rare_2d_values[k][m][k_vals_idx][m_vals_idx]:
                                            continue
                                        if rare_3d_values[i][j][m][i_vals_idx][j_vals_idx][m_vals_idx]:
                                            continue
                                        if rare_3d_values[i][k][m][i_vals_idx][k_vals_idx][m_vals_idx]:
                                            continue
                                        if rare_3d_values[j][k][m][j_vals_idx][k_vals_idx][m_vals_idx]:
                                            continue
                                        m_val = unique_vals[m][m_vals_idx]
                                        cond4 = (data_np[:, m] == m_val)
                                        for n_vals_idx in range(num_unique_vals_n):
                                            if rare_1d_values[n][n_vals_idx]:
                                                continue
                                            if rare_2d_values[i][n][i_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_2d_values[j][n][j_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_2d_values[k][n][k_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_2d_values[m][n][m_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_3d_values[i][j][n][i_vals_idx][j_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_3d_values[i][k][n][i_vals_idx][k_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_3d_values[i][m][n][i_vals_idx][m_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_3d_values[j][k][n][j_vals_idx][k_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_3d_values[j][m][n][j_vals_idx][m_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_3d_values[k][m][n][k_vals_idx][m_vals_idx][n_vals_idx]:
                                                continue
                                            if rare_4d_values[i][j][k][m][i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx]:
                                                continue
                                            n_val = unique_vals[n][n_vals_idx]
                                            cond5 = (data_np[:, n] == n_val)
                                            for p_vals_idx in range(num_unique_vals_p):
                                                if rare_1d_values[p][p_vals_idx]:
                                                    continue
                                                if rare_2d_values[i][p][i_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_2d_values[j][p][j_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_2d_values[k][p][k_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_2d_values[m][p][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_2d_values[n][p][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[i][j][p][i_vals_idx][j_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[i][k][p][i_vals_idx][k_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[i][m][p][i_vals_idx][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[i][n][p][i_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[j][k][p][j_vals_idx][k_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[j][m][p][j_vals_idx][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[j][n][p][j_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[k][m][p][k_vals_idx][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_3d_values[k][n][p][k_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[i][j][k][p][i_vals_idx][j_vals_idx][k_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[i][j][m][p][i_vals_idx][j_vals_idx][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[i][j][n][p][i_vals_idx][j_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[j][k][m][p][j_vals_idx][k_vals_idx][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[j][k][n][p][j_vals_idx][k_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[j][m][n][p][j_vals_idx][m_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_4d_values[k][m][n][p][k_vals_idx][m_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_5d_values[i][j][k][m][p][i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_5d_values[i][j][k][n][p][i_vals_idx][j_vals_idx][k_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_5d_values[i][j][m][n][p][i_vals_idx][j_vals_idx][m_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_5d_values[i][k][m][n][p][i_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue
                                                if rare_5d_values[j][k][m][n][p][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx][p_vals_idx]:
                                                    continue

                                                p_val = unique_vals[n][n_vals_idx]
                                                cond6 = (data_np[:, p] == p_val)

                                                rows_all = np.where(cond1 & cond2 & cond3 & cond4 & cond5 & cond6)
                                                current_fraction = len(rows_all[0]) / num_rows
                                                six_d_row_nums = rows_all[0]  # todo: use less variables

                                                if obj.check_marginal_probs:
                                                    expected_given_marginal = fractions_1d[i][i_vals_idx] * \
                                                                              fractions_1d[j][j_vals_idx] * \
                                                                              fractions_1d[k][k_vals_idx] * \
                                                                              fractions_1d[m][m_vals_idx] * \
                                                                              fractions_1d[n][n_vals_idx] * \
                                                                              fractions_1d[p][p_vals_idx] * \
                                                                              divisor
                                                else:
                                                    expected_given_marginal = np.inf
                                                rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                                                  (current_fraction < expected_given_marginal) and \
                                                                  (current_fraction < 0.01)
                                                if rare_value_flag:
                                                    row_nums = six_d_row_nums  # todo: can remove some variables here
                                                    assert len(row_nums) == round(current_fraction * num_rows), \
                                                        f"len of matching rows: {len(row_nums)}, fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                                                    for r in row_nums:
                                                        # todo: i doubt this is threadsafe
                                                        outliers_6d_arr_for_i[r] += 1
                                                        expl = [[obj.data_df.columns[i],
                                                                 obj.data_df.columns[j],
                                                                 obj.data_df.columns[k],
                                                                 obj.data_df.columns[m],
                                                                 obj.data_df.columns[n],
                                                                 obj.data_df.columns[p]],
                                                                [obj._get_col_value(i, i_vals_idx),
                                                                 obj._get_col_value(j, j_vals_idx),
                                                                 obj._get_col_value(k, k_vals_idx),
                                                                 obj._get_col_value(m, m_vals_idx),
                                                                 obj._get_col_value(n, n_vals_idx),
                                                                 obj._get_col_value(p, p_vals_idx)]
                                                                ]
                                                        if not outliers_explanation_arr_for_i[r]:
                                                            outliers_explanation_arr_for_i[r] = [expl]
                                                        else:
                                                            outliers_explanation_arr_for_i[r].append(expl)

                                                local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx][p_vals_idx] = rare_value_flag
                        rare_arr_for_i[j][k][m][n][p] = local_rare_arr
    return rare_arr_for_i, outliers_6d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


###################################################################################################################
# Simple utility methods

def flatten(arr):
    """
    Flatten a python array of any dimensionality into a 1d python array.
    """
    while True:
        if len(arr) == 0:
            return arr
        if not any(1 for x in arr if type(x) is list):
            return arr
        #arr = [x[0] if (type(x) is list) and (len(x) == 1) else x for x in arr]
        arr = tuple(i for row in arr for i in row)


def is_float(v):
    """
    Determine if a passed value is float or not. Integers are not considered float.
    """
    if str(v).isdigit():
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False


def skip_x_ticks(s):
    max_ticks = 15
    num_ticks = len(s.xaxis.get_ticklabels())
    if num_ticks < max_ticks:
        return
    step = num_ticks // max_ticks
    for label_idx, label in enumerate(s.xaxis.get_ticklabels()):
        if label_idx % step != 0:
            label.set_visible(False)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


# From https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    new_cmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=new_cmap)

    return new_cmap
