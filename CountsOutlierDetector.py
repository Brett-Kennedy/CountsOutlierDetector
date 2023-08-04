import os
import pandas as pd
import numpy as np
import random
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
import concurrent
from statistics import mean


class CountsOutlierDetector:
    def __init__(self,
                 n_bins=7,
                 max_dimensions=6,
                 max_num_combinations=100_000_000,
                 min_values_per_column=2,
                 max_values_per_column=50,
                 results_folder="",
                 results_name="",
                 run_parallel=False):
        """
        :param n_bins: int
            The number of bins used to reduce numeric columns to a small set of ordinal values.
        :param max_dimensions: int
            The maximum number of columns examined at any time. If set to, for example, 4, then the detector will check
            for 1d, 2d, 3d, and 4d outliers, but not outliers in higher dimensions.
        :param max_num_combinations: int
            This, as well as max_dimensions, determines the maximum number of dimensions that may be examined at a time.
            When determining if the detector considers, for example, 3d outliers, it examines the number of columns and
            average number of unique values per column and estimates the total number of combinations. If this exceeds
            max_num_combinations, the detector will not consider spaces of this dimensionality or higher.
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
        """

        if max_dimensions > 6:
            print("The maximum value for max_dimensions is 6. Using max_dimensions=6 for outlier detection.")
            max_dimensions = 6

        if results_folder == "" and results_name != "":
            print(("results_folder is required when results_file is specified. Specify both in order to save output"
                   "to a .csv file. Exiting."))
            return

        self.n_bins = n_bins
        self.max_dimensions = max_dimensions
        self.max_num_combinations = max_num_combinations
        self.min_values_per_column = min_values_per_column
        self.max_values_per_column = max_values_per_column
        self.results_folder = results_folder
        self.results_name = results_name
        self.run_parallel = run_parallel

        # An array indicating the type of each column in the dataset. All columns are considered as either numeric
        # or categorical.
        self.col_types_arr = []

        # An encoder used to convert all categorical values to numbers. An ordinal representation is used for each
        # categorical value for efficiency.
        self.ordinal_encoders_arr = []

        # todo: fill in
        self.divisor = 0.25  # todo: loop through different values of this to see how it affects the results.

        # Copy of the input dataframe.
        self.data_df = None

        # The number of dimension actually examined. This may be less than max_dimensions, given max_num_combinations
        self.dimensions_examined = None

        # Set the seed to ensure consistent results from run to run on the same dataset.
        np.random.seed(0)
        random.seed(0)

        # Set display options for dataframes
        pd.options.display.max_columns = 1000
        pd.options.display.max_rows = 1000
        pd.options.display.width = 10000

    # Using numba appears to give similar performance results
    # @jit(nopython=True)
    # def get_cond(X_vals, col_idx, val):
    #     cond = (X_vals[:, col_idx] == val)
    #     return cond

    def predict(self, input_data):
        """
        Determine the outlier score of all rows in the data

        :param input_data: pandas dataframe, or data structure that may be converted to a pandas dataframe, such as
            numpy array, 2d python array, or dictionary

        :return: dictionary
            todo: describe here
        """

        def update_output_msg(msg, arr):
            """
            Updates the msg used to describe each execution of predict(). This gives the counts of the number of rows
            flagged n times for a given dimensionality. For example, the number of rows flagged 0 times, 1 times, 2
            times, and so on.

            :param msg: string
                A short text message to be displayed along with each unique count in arr
            :param arr:
                An array representing the number of outliers detected per row for a given dimensionality. arr should
                have one element for each row in the full dataset.
            """
            nonlocal output_msg
            unique_counts = sorted(list(set(arr)))
            for uc in unique_counts:
                output_msg += f"\n{msg}: {uc}: {arr.count(uc):5}"

        def get_2d_fractions(i, j):
            """
            Given two columns i and j: get, for each pair of values, the fraction of the dataset and the row numbers.
            Returns:
                two_d_fractions: a 2d array, with a row for each value in column i, and a column for each value in
                    column j. Each cell contains the count for that combination.
                two_2_row_nums: todo: fill in!
            """

            two_d_fractions = []
            two_d_row_nums = []
            for i_val in unique_vals[i]:
                i_vals_fractions = []
                i_vals_row_nums = []
                cond1 = (data_np[:, i] == i_val)
                for j_val in unique_vals[j]:
                    cond2 = (data_np[:, j] == j_val)
                    rows_both = np.where(cond1 & cond2)
                    i_vals_fractions.append(len(rows_both[0]) / num_rows)
                    i_vals_row_nums.append(rows_both[0])
                two_d_fractions.append(i_vals_fractions)
                two_d_row_nums.append(i_vals_row_nums)
            return two_d_fractions, two_d_row_nums

        def get_unique_vals():
            """
            This finds the unique values in each column.

            :return:
            unique_vals: A 2d array, with a list of unique values for each feature.
            num_unique_values: A 1d array, with a count of unique values for each feature.
            """

            nonlocal output_msg

            # An element for each column. For categorical columns, lists the unique values. This is used to maintain a
            # consistent order of the values within each column.
            unique_vals = []
            num_unique_vals = []

            for i in range(num_cols):
                uv = self.data_df.iloc[:, i].unique()
                # # If there are many unique values, remove the float values
                # # todo: set this threshold as a parameter
                # # todo: need to save the pre-ordinal encoded values to do this.
                # # todo: or could not do it: then don't need to save the unique values, just the count and can assume it's
                # #  0 up to that.
                # #if len(uv) > 25:
                # #    uv = [v for v in uv if not is_float(v)]
                # col_threshold = (1.0 / len(uv)) * self.divisor
                unique_vals.append(uv)
                num_unique_vals.append(len(uv))
            return unique_vals, num_unique_vals

        def get_1d_stats():
            nonlocal output_msg

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
            outliers_explanation_arr = [""] * num_rows

            for i in range(num_cols):

                col_threshold = (1.0 / num_unique_vals[i]) * self.divisor  # todo: make self.divisor a hyperparameter -- need to explain it well!!

                # Array with an element for each unique value in column i. Indicates the fraction of the total dataset
                # held by that value.
                col_fractions_1d = []

                # Array with an element for each unique value in column i. Indicates if that value is considered rare.
                col_rare_1d_values = []

                for v in unique_vals[i]:  # loop through each unique value in the current column.
                    frac = self.data_df.iloc[:, i].tolist().count(v) / num_rows
                    col_fractions_1d.append(frac)
                    rare_values_flag = (frac < col_threshold) and (frac < 0.01)
                    if rare_values_flag:
                        rows_matching = np.where(data_np[:, i] == v)
                        for r in rows_matching[0]:
                            outliers_1d_arr[r] += 1
                            outliers_explanation_arr[r] += f'[Column: {self.data_df.columns[i]}, ' + \
                                                           f'Value: {self.__get_col_value(i,v)}, fraction: {frac}]'
                    col_rare_1d_values.append(rare_values_flag)
                fractions_1d[i] = col_fractions_1d
                rare_1d_values[i] = col_rare_1d_values

            output_msg += f"\n\n1d: num common values: {flatten(rare_1d_values).count(False)}"
            output_msg += f"\n1d: num rare values: {flatten(rare_1d_values).count(True)}"
            update_output_msg("1d: Outlier Counts by score", outliers_1d_arr)
            return fractions_1d, rare_1d_values, outliers_1d_arr, outliers_explanation_arr

        def get_2d_stats():
            """
            This returns 2 parallel 4d arrays: fractions_2d and rare_2d_values, both with the dimensions:
            i column, j column, value in i column, value in j column
            """
            nonlocal output_msg  # todo: make an object scope variable?

            # Each element stores the fraction of the total dataset with this combination of values.
            fractions_2d = [] * num_cols

            # Each element stores a boolean indicating if this combination of values is considered rare in the 2d sense.
            rare_2d_values = [] * num_cols

            # Integer value for each row in the dataset indicating how many pairs of columns have combinations
            # considered outliers.
            outliers_2d_arr = [0] * num_rows
            outliers_explanation_arr = [""] * num_rows

            for i in range(num_cols):
                fractions_2d.append([[]] * num_cols)
                rare_2d_values.append([[]] * num_cols)

            for i in range(num_cols - 1):
                #print("2d i: ",i)
                for j in range(i + 1, num_cols):
                    local_fractions, two_d_row_nums = get_2d_fractions(i, j)
                    fractions_2d[i][j] = local_fractions

                    # Determine which of these fractions would be considered rare in the 2d sense
                    i_rare_arr = []
                    expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]))
                    for i_vals_idx in range(len(fractions_2d[i][j])):
                        j_rare_arr = []
                        for j_vals_idx in range(len(fractions_2d[i][j][i_vals_idx])):
                            current_fraction = fractions_2d[i][j][i_vals_idx][j_vals_idx]
                            expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx]
                            rare_value_flag = (rare_1d_values[i][i_vals_idx] == False) and \
                                              (rare_1d_values[j][j_vals_idx] == False) and \
                                              (current_fraction < (expected_under_uniform * self.divisor)) and \
                                              (current_fraction < (expected_given_marginal * self.divisor)) and \
                                              (current_fraction < 0.01)
                            if rare_value_flag:
                                row_nums = two_d_row_nums[i_vals_idx][j_vals_idx]
                                assert len(row_nums) == round(current_fraction * num_rows), \
                                    (f"len of matching rows: {len(row_nums)}, fraction*num_rows: " \
                                     f"current_fraction*num_rows: {current_fraction * num_rows}")
                                for r in row_nums:
                                    outliers_2d_arr[r] += 1
                                    # todo: format this for 3, 4, 5d too
                                    outliers_explanation_arr[r] += f" [[Columns: {self.data_df.columns[i]} and " +\
                                        f"{self.data_df.columns[j]} Values: {self.__get_col_value(i, i_vals_idx)} and " +\
                                        f"{self.__get_col_value(j, j_vals_idx)}, Fraction: {current_fraction}]]"
                            j_rare_arr.append(rare_value_flag)
                        i_rare_arr.append(j_rare_arr)
                    rare_2d_values[i][j] = i_rare_arr

            out = flatten(rare_2d_values)
            output_msg += f"\n\n2d: num common combinations: {out.count(False):,}"
            output_msg += f"\n2d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            update_output_msg("2d: Outlier Counts by score", outliers_2d_arr)
            return fractions_2d, rare_2d_values, outliers_2d_arr, outliers_explanation_arr

        def get_3d_stats(num_combinations):
            """
            This returns 2 parallel 6d arrays: fractions_3d and rare_3d_values, both with the dimensions:
            i column, j column, k column, value in i column, value in j column, value in the k column

            It also returns: outliers_3d_arr, outliers_explanation_arr, and column_combos_checked.

            todo: explain each of these
            """
            nonlocal output_msg

            fractions_3d = [[]] * num_cols # todo: not used, though will if go to 4d
            rare_3d_values = [[]] * num_cols
            outliers_3d_arr = [0] * num_rows
            outliers_explanation_arr = [""] * num_rows
            column_combos_checked = 0

            run_parallel_3d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_3d = False

            if run_parallel_3d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_3d,
                                            i,
                                            data_np,
                                            num_cols,
                                            num_rows,
                                            unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values,
                                            self.divisor)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_3d_values[f_idx] = rare_arr_for_i
                        outliers_3d_arr = [x + y for x, y in zip(outliers_3d_arr, outliers_3d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
                        #print("outliers_3d_arr_for_i: ", outliers_3d_arr_for_i.count(0), outliers_3d_arr_for_i.count(1))
            else:
                for i in range(num_cols):
                    rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_3d(
                        i,
                        data_np,
                        num_cols,
                        num_rows,
                        unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        self.divisor
                    )
                    rare_3d_values[i] = rare_arr_for_i
                    outliers_3d_arr = [x + y for x, y in zip(outliers_3d_arr, outliers_3d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_3d_values)
            output_msg += f"\n\n3d: num common combinations: {out.count(False):,}"
            output_msg += f"\n3d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            update_output_msg("3d: Outlier Counts by score", outliers_3d_arr)
            return fractions_3d, rare_3d_values, outliers_3d_arr, outliers_explanation_arr, column_combos_checked

        def get_4d_stats(num_combinations):
            """
            This returns 2 parallel 8d arrays: fractions_4d and rare_4d_values, both with the dimensions:
            i column, j column, k column, m column,
            value in i column, value in j column, value in the k column, value in the m column

            It also returns: outliers_4d_arr, outliers_explanation_arr, and column_combos_checked.

            These are analogous to get_3d_stats()
            """
            nonlocal output_msg

            fractions_4d = [[]] * num_cols
            rare_4d_values = [[]] * num_cols
            outliers_4d_arr = [0] * num_rows
            outliers_explanation_arr = [""] * num_rows
            column_combos_checked = 0

            run_parallel_4d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_4d = False

            if run_parallel_4d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_4d,
                                            i,
                                            data_np,
                                            num_cols,
                                            num_rows,
                                            unique_vals,
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
                for i in range(num_cols):
                    rare_arr_for_i, outliers_4d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_4d(
                        i,
                        data_np,
                        num_cols,
                        num_rows,
                        unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        rare_3d_values,
                        self.divisor
                    )
                    rare_4d_values[i] = rare_arr_for_i
                    outliers_4d_arr = [x + y for x, y in zip(outliers_4d_arr, outliers_4d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_4d_values)
            output_msg += f"\n\n4d: num common combinations: {out.count(False):,}"
            output_msg += f"\n4d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            update_output_msg("4d: Outlier Counts by score", outliers_4d_arr)
            return fractions_4d, rare_4d_values, outliers_4d_arr, outliers_explanation_arr, column_combos_checked

        def get_5d_stats(num_combinations):
            """
            This returns 2 parallel 10d arrays: fractions_5d and rare_5d_values, both with the dimensions:
            i column, j column, k column, m column, n column
            value in i column, value in j column, value in the k column, value in the m column, values in the n column

            It also returns: outliers_5d_arr,  outliers_explanation_arr, and column_combos_checked.

            These are analogous to get_3d_stats()
            """
            nonlocal output_msg

            # todo: update this comment. Make more general, so don't repeat it
            fractions_5d = [[]] * num_cols
            rare_5d_values = [[]] * num_cols
            outliers_5d_arr = [0] * num_rows
            outliers_explanation_arr = [""] * num_rows
            column_combos_checked = 0

            run_parallel_5d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_5d = False

            if run_parallel_5d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_5d,
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
                                            self.divisor)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_5d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_5d_values[f_idx] = rare_arr_for_i
                        outliers_5d_arr = [x + y for x, y in zip(outliers_5d_arr, outliers_5d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
            else:
                for i in range(num_cols):
                    rare_arr_for_i, outliers_5d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_5d(
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
                        self.divisor
                    )
                    rare_5d_values[i] = rare_arr_for_i
                    outliers_5d_arr = [x + y for x, y in zip(outliers_5d_arr, outliers_5d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in
                                                zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i
            out = flatten(rare_5d_values)
            output_msg += f"\n\n5d: num common combinations: {out.count(False):,}"
            output_msg += f"\n5d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            update_output_msg("5d: Outlier Counts by score", outliers_5d_arr)
            return fractions_5d, rare_5d_values, outliers_5d_arr, outliers_explanation_arr, column_combos_checked

        def get_6d_stats(num_combinations):
            """
            This returns 2 parallel 12d arrays: fractions_6d and rare_6d_values, both with the dimensions:
            i column, j column, k column, m column, todo: more??
            value in i column, value in j column, value in the k column, value in the m column # todo: more??

            It also returns outliers_6d_arr, outliers_explanation_arr, and column_combos_checked.

            These are analogous to get_3d_stats()
            """
            nonlocal output_msg

            fractions_6d = [[]] * num_cols
            rare_6d_values = [[]] * num_cols
            outliers_6d_arr = [0] * num_rows
            outliers_explanation_arr = [""] * num_rows
            column_combos_checked = 0

            run_parallel_6d = self.run_parallel
            if num_combinations < 1_000_000:
                run_parallel_6d = False

            if run_parallel_6d:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(process_inner_loop_6d,
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
                                            self.divisor)
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_6d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = f.result()
                        rare_6d_values[f_idx] = rare_arr_for_i
                        outliers_6d_arr = [x + y for x, y in zip(outliers_6d_arr, outliers_6d_arr_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                        column_combos_checked += column_combos_checked_for_i
            else:
                for i in range(num_cols):
                    rare_arr_for_i, outliers_6d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i = process_inner_loop_6d(
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
                        self.divisor
                    )
                    rare_6d_values[i] = rare_arr_for_i
                    outliers_6d_arr = [x + y for x, y in zip(outliers_6d_arr, outliers_6d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_6d_values)
            output_msg += f"\n\n6d: num common combinations: {out.count(False):,}"
            output_msg += f"\n6d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            update_output_msg("6d: Outlier Counts by score", outliers_6d_arr)
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
                file_name = self.results_folder + "\\" + self.results_name + "_results_" + dt_string + ".csv"
                df.to_csv(file_name)

            return df

        def create_return_dict():
            return {'Breakdown All Rows': flagged_rows_df,
                    'Breakdown Flagged Rows': self.output_explanations(flagged_rows_df),
                    'Run Summary': output_msg,
                    'run_summary_df': 'run_summary_df'
                    }

        self.data_df = pd.DataFrame(input_data).copy()

        # Create a list of the numeric columns
        self.col_types_arr = self.__get_col_types_arr()
        numeric_col_names = [self.data_df.columns[x]
                             for x in range(len(self.col_types_arr)) if self.col_types_arr[x] == 'N']

        # Bin any numeric columns
        # todo: test with k-means as the strategy
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        if len(numeric_col_names):
            x_num = self.data_df[numeric_col_names]
            xt = est.fit_transform(x_num)
            for col_idx, col_name in enumerate(numeric_col_names):
                self.data_df[col_name] = xt[:, col_idx].astype(int)

        # Remove any columns with too few (less than min_values_per_column) unique values or too many (more than
        # max_values_per_column) unique values.
        drop_col_names_arr = []
        for c in range(len(self.data_df.columns)):
            if self.data_df[self.data_df.columns[c]].nunique() < self.min_values_per_column or \
                    self.data_df[self.data_df.columns[c]].nunique() > self.max_values_per_column:
                drop_col_names_arr.append(input_data.columns[c])
        self.data_df = self.data_df.drop(columns=drop_col_names_arr)
        num_cols = len(self.data_df.columns)
        num_rows = len(self.data_df)

        output_msg = f"\nNumber of rows: {num_rows}"
        output_msg += f"\nNumber of columns: {num_cols}"

        # Create a summary of this run, giving statistics about the outliers found
        run_summary_df = pd.DataFrame(columns=[
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

            # Binary indicators giving the size of subspaces checked. These are False if there are too many
            # combinations to check
            'Checked_3d',
            'Checked_4d',
            'Checked_5d',
            'Checked_6d',

            # Skip column combinations where expected count based on marginal probabilities is too low.
            '3d column combos checked',
            '4d column combos checked',
            '5d column combos checked',
            '6d column combos checked',

            'Percent Flagged'])

        if num_cols < 2:
            output_msg += ("\nLess than two columns found (after dropping columns with too few or too many unique "
                           "values. Cannot determine outliers. Consider adjusting min_values_per_column or "
                           "max_values_per_column")
            #return None, None, output_msg, run_summary_df
            return create_return_dict()

        self.__ordinal_encode()
        data_np = self.data_df.values
        unique_vals, num_unique_vals, = get_unique_vals()
        output_msg += f"\nCardinality of the columns (after binning numeric columns): {num_unique_vals}"

        # Determine the 1d stats
        fractions_1d, rare_1d_values, outliers_1d_arr, explanations_1d_arr = get_1d_stats()

        # Determine the 2d stats
        fractions_2d, rare_2d_values, outliers_2d_arr, explanations_2d_arr = get_2d_stats()
        self.dimensions_examined = 2

        # Determine the 3d stats unless there are too many columns and unique values to do so efficiently
        checked_3d = False
        column_combos_checked_3d = -1
        avg_num_unique_vals = mean([len(x) for x in unique_vals])
        num_combinations = (num_cols * (num_cols-1) * (num_cols-2)) * pow(avg_num_unique_vals, 3)
        if num_combinations > self.max_num_combinations:
            output_msg += (f"\n\nCannot determine 3d outliers given the number of columns ({num_cols}) and",
                           f"number of unique values in each. Estimated number of combinations: "
                           f"{round(num_combinations):,}")
            outliers_3d_arr = [0] * num_rows
            explanations_3d_arr = [""] * num_rows
        else:
            fractions_3d, rare_3d_values, outliers_3d_arr, explanations_3d_arr, column_combos_checked_3d = \
                get_3d_stats(num_combinations=num_combinations)
            checked_3d = True
            self.dimensions_examined = 3

        # Determine the 4d stats unless there are too many columns and unique values to do so efficiently
        # todo here and above just use pow method
        checked_4d = False
        column_combos_checked_4d = -1
        num_combinations = (num_cols*(num_cols-1)*(num_cols-2)*(num_cols-3)) * pow(avg_num_unique_vals, 4)
        outliers_4d_arr = [0] * num_rows
        explanations_4d_arr = [""] * num_rows
        if num_cols < 4:
            output_msg += f"\n\nCannot determine 4d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
        elif num_combinations > self.max_num_combinations:
            output_msg += (f"\n\nCannot determine 4d outliers given the number of columns ({num_cols}) and number of "
                           f"unique values in each. Estimated number of combinations: {round(num_combinations):,}")
        else:
            fractions_4d, rare_4d_values, outliers_4d_arr, explanations_4d_arr, column_combos_checked_4d = \
                get_4d_stats(num_combinations=num_combinations)
            checked_4d = True
            self.dimensions_examined = 4

        # Determine the 5d stats unless there are too many columns and unique values to do so efficiently
        checked_5d = False
        column_combos_checked_5d = -1
        num_combinations = (num_cols*(num_cols-1)*(num_cols-2)*(num_cols-3)*(num_cols-4)) * pow(avg_num_unique_vals, 5)
        outliers_5d_arr = [0] * num_rows
        explanations_5d_arr = [""] * num_rows
        if num_cols < 5:
            output_msg += f"\n\nCannot determine 5d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
        elif num_combinations > self.max_num_combinations:
            output_msg += (f"\n\nCannot determine 5d outliers given the number of columns ({num_cols}) and number of "
                           f"unique values in each. Estimated number of combinations: {round(num_combinations):,}")
        else:
            fractions_5d, rare_5d_values, outliers_5d_arr, explanations_5d_arr, column_combos_checked_5d = \
                get_5d_stats(num_combinations=num_combinations)
            checked_5d = True
            self.dimensions_examined = 5

        # Determine the 6d stats unless there are too many columns and unique values to do so efficiently
        checked_6d = False
        column_combos_checked_6d = -1
        num_combinations = (num_cols*(num_cols-1)*(num_cols-2)*(num_cols-3)*(num_cols-4)*(num_cols-5)) * pow(avg_num_unique_vals, 5)
        outliers_6d_arr = [0] * num_rows
        explanations_6d_arr = [""] * num_rows
        if num_cols < 6:
            output_msg += f"\n\nCannot determine 6d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
        elif num_combinations > self.max_num_combinations:
            output_msg += f"\n\nCannot determine 6d outliers given the number of columns ({num_cols}) and number of unique values in each. Estimated number of combinations: {round(num_combinations):,}"
        else:
            fractions_6d, rare_6d_values, outliers_6d_arr, explanations_6d_arr, column_combos_checked_6d = \
                get_6d_stats(num_combinations=num_combinations)
            checked_6d = True
            self.dimensions_examined = 6

        flagged_rows_df = create_output_csv(
            outliers_1d_arr, outliers_2d_arr, outliers_3d_arr, outliers_4d_arr, outliers_5d_arr, outliers_6d_arr,
            explanations_1d_arr, explanations_2d_arr, explanations_3d_arr, explanations_4d_arr, explanations_5d_arr,
            explanations_6d_arr)

        num_rows_scored = list(flagged_rows_df['Any at 1d'] > 0).count(True)
        output_msg += f"\n\nNumber of rows flagged as outliers examining 1d: {num_rows_scored}" +\
                      f" ({round(num_rows_scored*100.0/num_rows, 3)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 2d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 2d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,3)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 3d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 3d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,3)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 4d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 4d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,3)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 5d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 5d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,3)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 6d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 6d: {num_rows_scored} " + \
                      f"({round(num_rows_scored*100.0/num_rows,3)}%)"

        # Update run_summary_df
        run_summary_df = run_summary_df.append(pd.DataFrame(np.array([[
                flagged_rows_df['Any at 1d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 2d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 3d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 4d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 5d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 6d'].sum() * 100.0 / num_rows,

                flagged_rows_df['Any up to 1d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 2d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 3d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 4d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 5d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 6d'].sum() * 100.0 / num_rows,

                checked_3d,
                checked_4d,
                checked_5d,
                checked_6d,
                column_combos_checked_3d,
                column_combos_checked_4d,
                column_combos_checked_5d,
                column_combos_checked_6d,
                flagged_rows_df['Any Scored'].sum() * 100.0 / num_rows]]),
            columns=run_summary_df.columns))

        return create_return_dict()

    def output_explanations(self, flagged_rows_df):
        """
        Given a dataframe with the full information about flagged rows, return a smaller, simpler dataframe to
        summarize this concisely.

        :param flagged_rows_df a dataframe with a row for each row in the original data, and numerous columns
            summarizing the outlier combinations found in each row.

        :return: A dataframe with a row for each row in the original data where at least one combination of values was
            flagged.
        """

        df_subset = flagged_rows_df[flagged_rows_df['Any Scored']]
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

    def explain_row(self, row_index):
        return "Unimplemented"  # todo: fill in

    def __get_col_types_arr(self):
        """
        Create an array representing each column of the data, with each coded as 'C' (categorical) or 'N' (numeric).
        """

        col_types_arr = ['N'] * len(self.data_df.columns)

        for c in range(len(self.data_df.columns)):
            num_unique = self.data_df[self.data_df.columns[c]].nunique()
            if not is_numeric_dtype(self.data_df[self.data_df.columns[c]]):
                col_types_arr[c] = 'C'
            # Even where the values are numeric, if there are few of them, consider them categorical, though if the
            # values are all float, the column will be cast to 'N' when collecting the unique values.
            elif is_numeric_dtype(self.data_df[self.data_df.columns[c]]) and num_unique <= 25:
                col_types_arr[c] = 'C'

        # If there are a large number of categorical columns, re-determine the types with a more strict cutoff
        if col_types_arr.count('C') > 50:
            col_types_arr = ['N'] * len(self.data_df.columns)
            for c in range(len(self.data_df.columns)):
                num_unique = self.data_df[self.data_df.columns[c]].nunique()
                if not is_numeric_dtype(self.data_df[self.data_df.columns[c]]):
                    col_types_arr[c] = 'C'
                elif is_numeric_dtype(self.data_df[self.data_df.columns[c]]) and num_unique <= 5:
                    col_types_arr[c] = 'C'

        return col_types_arr

    def __ordinal_encode(self):
        """
        numpy deals with numeric values much more efficiently than text. We convert any categorical columns to ordinal
        values based on either their original values or bin ids.
        """

        self.ordinal_encoders_arr = [None]*len(self.data_df.columns)
        for i in range(len(self.data_df.columns)):
            if self.col_types_arr[i] == 'C':
                enc = OrdinalEncoder()
                self.ordinal_encoders_arr[i] = enc
                col_vals = self.data_df[self.data_df.columns[i]].values.reshape(-1, 1)
                X_np = enc.fit_transform(col_vals).astype(int)
                self.data_df[self.data_df.columns[i]] = X_np
        return self.data_df

    def __get_col_value(self, col_idx, value_idx):
        """
        Return the original value of a specified column and ordinal value. For binned numeric columns, there is a
        range of values in each bin, and so the bin id is returned to approximate the original value.
        """
        if self.col_types_arr[col_idx] == "C":
            return self.ordinal_encoders_arr[col_idx].inverse_transform([[value_idx]])[0][0]
        else:
            return f"Bin {value_idx}"


##############################################################################################################
# Methods to examine specific subspaces. These methods are outside the class in order to be callable as concurrent
# processes

def process_inner_loop_3d(
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
    outliers_explanation_arr_for_i = [""] * num_rows
    column_combos_checked_for_i = 0

    rare_arr_for_i = [[]] * num_cols
    for k in range(num_cols):
        rare_arr_for_i[k] = [[]] * num_cols

    for j in range(i + 1, num_cols - 1):
        num_unique_vals_j = len(unique_vals[j])
        for k in range(j + 1, num_cols):
            num_unique_vals_k = len(unique_vals[k])

            expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k]))
            expected_count_under_uniform = num_rows * expected_under_uniform
            if expected_count_under_uniform < 10:
                continue
            column_combos_checked_for_i += 1

            local_rare_arr = [[[False]*num_unique_vals_k]*num_unique_vals_j for _ in range(num_unique_vals_i)]
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

                        expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx] * \
                                                  fractions_1d[k][k_vals_idx]
                        rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                          (current_fraction < (expected_given_marginal * divisor)) and \
                                          (current_fraction < 0.01)
                        if rare_value_flag:
                            row_nums = three_d_row_nums  # todo: can remove some variables here
                            assert len(row_nums) == round(current_fraction * num_rows), \
                                f"len of matching rows: {len(row_nums)}, fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                            for r in row_nums:
                                # todo: i doubt this is threadsafe
                                outliers_3d_arr_for_i[r] += 1
                                outliers_explanation_arr_for_i[r] += f" [[[Columns: {i} {j} {k} Values: {i_vals_idx} {j_vals_idx} {k_vals_idx}, Fraction: {current_fraction}]]]"
                        local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx] = rare_value_flag
            rare_arr_for_i[j][k] = local_rare_arr
    return rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def process_inner_loop_4d(
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
    outliers_explanation_arr_for_i = [""] * num_rows
    rare_arr_for_i = [[[[]]*num_cols]*num_cols for _ in range(num_cols)]
    column_combos_checked_for_i = 0
    max_cardinality = max([len(x) for x in unique_vals])

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

                local_rare_arr = [[[[False]*max_cardinality]*max_cardinality]*max_cardinality for _ in range(max_cardinality)]
                for i_vals_idx in range(num_unique_vals_i):
                    if rare_1d_values[i][i_vals_idx]:
                        continue
                    i_val = unique_vals[i][i_vals_idx]
                    cond1 = (data_np[:, i] == i_val)
                    for j_vals_idx in range(num_unique_vals_j):
                        if rare_1d_values[j][j_vals_idx]:
                            continue
                        j_val = unique_vals[j][j_vals_idx]
                        cond2 = (data_np[:, j] == j_val)
                        if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                            continue
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
                                four_d_row_nums = rows_all[0] # todo: use less variables

                                expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx] * \
                                                          fractions_1d[k][k_vals_idx] * fractions_1d[m][m_vals_idx]
                                rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                                  (current_fraction < (expected_given_marginal * divisor)) and \
                                                  (current_fraction < 0.01)
                                if rare_value_flag:
                                    row_nums = four_d_row_nums  # todo: can remove some variables here
                                    assert len(row_nums) == round(current_fraction * num_rows), \
                                        f"len of matching rows: {len(row_nums)}, " \
                                        f"fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                                    for r in row_nums:
                                        # todo: i doubt this is threadsafe
                                        outliers_4d_arr_for_i[r] += 1
                                        # todo: use the actual values, not their index
                                        outliers_explanation_arr_for_i[r] += \
                                            f" [[[Columns: {i} {j} {k} {m}" \
                                            f"Values: {i_vals_idx} {j_vals_idx} {k_vals_idx}  {m_vals_idx}" \
                                            f"Fraction: {current_fraction}]]]"
                                # todo: remove try-except logic
                                try:
                                    local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx] = rare_value_flag
                                except Exception as e:
                                    print(f"here {i_vals_idx}, {j_vals_idx}, {k_vals_idx}, {m_vals_idx}")
                rare_arr_for_i[j][k][m] = local_rare_arr
    return rare_arr_for_i, outliers_4d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def process_inner_loop_5d(
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
    outliers_explanation_arr_for_i = [""] * num_rows
    rare_arr_for_i = [[[[[]]*num_cols]*num_cols]*num_cols]*num_cols
    column_combos_checked_for_i = 0
    max_cardinality = max([len(x) for x in unique_vals])

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
                    #local_rare_arr = [[[[[False]*num_unique_vals_n]*num_unique_vals_m]*num_unique_vals_k]*num_unique_vals_j]*num_unique_vals_i
                    local_rare_arr = [[[[[False]*max_cardinality]*max_cardinality]*max_cardinality]*max_cardinality for _ in range(max_cardinality)]
                    for i_vals_idx in range(num_unique_vals_i):
                        if rare_1d_values[i][i_vals_idx]:
                            continue
                        i_val = unique_vals[i][i_vals_idx]
                        cond1 = (data_np[:, i] == i_val)
                        for j_vals_idx in range(num_unique_vals_j):
                            if rare_1d_values[j][j_vals_idx]:
                                continue
                            j_val = unique_vals[j][j_vals_idx]
                            cond2 = (data_np[:, j] == j_val)
                            if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                                continue
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
                                        try:
                                            if rare_4d_values[i][j][k][n][i_vals_idx][j_vals_idx][k_vals_idx][n_vals_idx]:
                                                continue
                                        except Exception as e:
                                            print(f" case 2 error: {e}, indexes: {i},{j},{k},{n},{i_vals_idx},{j_vals_idx},{k_vals_idx},{n_vals_idx}")
                                        try:
                                            if rare_4d_values[i][j][m][n][i_vals_idx][j_vals_idx][m_vals_idx][n_vals_idx]:
                                                continue
                                        except Exception as e:
                                            print(f" case 3 error: {e}, indexes: {i},{j},{m},{n},{i_vals_idx},{j_vals_idx},{m_vals_idx},{n_vals_idx}")
                                        if rare_4d_values[i][k][m][n][i_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx]:
                                            continue
                                        try:
                                            if rare_4d_values[j][k][m][n][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx]:
                                                continue
                                        except Exception as e:
                                            print(f"case 5 error: {e}, indexes: {j},{k},{m},{n},{j_vals_idx},{k_vals_idx},{m_vals_idx},{n_vals_idx}")
                                        n_val = unique_vals[n][n_vals_idx]
                                        cond5 = (data_np[:, n] == n_val)

                                        rows_all = np.where(cond1 & cond2 & cond3 & cond4 & cond5)
                                        current_fraction = len(rows_all[0]) / num_rows
                                        five_d_row_nums = rows_all[0] # todo: use less variables

                                        expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx] * \
                                                                  fractions_1d[k][k_vals_idx] * fractions_1d[m][m_vals_idx] * fractions_1d[n][n_vals_idx]
                                        rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                                          (current_fraction < (expected_given_marginal * divisor)) and \
                                                          (current_fraction < 0.01)
                                        if rare_value_flag:
                                            row_nums = five_d_row_nums  # todo: can remove some variables here
                                            assert len(row_nums) == round(current_fraction * num_rows), \
                                                f"len of matching rows: {len(row_nums)}, fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                                            for r in row_nums:
                                                # todo: i doubt this is threadsafe
                                                outliers_5d_arr_for_i[r] += 1
                                                # todo: use the actual values, not their index
                                                outliers_explanation_arr_for_i[r] += f" [[[Columns: {i} {j} {k} {m} {n} Values: {i_vals_idx} {j_vals_idx} {k_vals_idx} {m_vals_idx} {n_vals_idx} Fraction: {current_fraction}]]]"
                                        local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx] = rare_value_flag
                    rare_arr_for_i[j][k][m][n] = local_rare_arr
    return rare_arr_for_i, outliers_5d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def process_inner_loop_6d(
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
    outliers_explanation_arr_for_i = [""] * num_rows
    rare_arr_for_i = [[[[[[]]*num_cols]*num_cols]*num_cols]*num_cols]*num_cols
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

                        expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k]) * len(unique_vals[m]) * len(unique_vals[n]) * len(unique_vals[p]))
                        expected_count_under_uniform = num_rows * expected_under_uniform
                        if expected_count_under_uniform < 10:
                            continue
                        column_combos_checked_for_i += 1

                        # local_rare_arr represents the current set of columns. It's a 5d array, with a dimension
                        # for each value.
                        local_rare_arr = [[[[[[False]*num_unique_vals_p]*num_unique_vals_n]*num_unique_vals_m]*num_unique_vals_k]*num_unique_vals_j]*num_unique_vals_i
                        for i_vals_idx in range(num_unique_vals_i):
                            if rare_1d_values[i][i_vals_idx]:
                                continue
                            i_val = unique_vals[i][i_vals_idx]
                            cond1 = (data_np[:, i] == i_val)
                            for j_vals_idx in range(num_unique_vals_j):
                                if rare_1d_values[j][j_vals_idx]:
                                    continue
                                j_val = unique_vals[j][j_vals_idx]
                                cond2 = (data_np[:, j] == j_val)
                                if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                                    continue
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

                                                expected_given_marginal = fractions_1d[i][i_vals_idx] * \
                                                                          fractions_1d[j][j_vals_idx] * \
                                                                          fractions_1d[k][k_vals_idx] * \
                                                                          fractions_1d[m][m_vals_idx] * \
                                                                          fractions_1d[n][n_vals_idx] * \
                                                                          fractions_1d[p][p_vals_idx]
                                                rare_value_flag = (current_fraction < (expected_under_uniform * divisor)) and \
                                                                  (current_fraction < (expected_given_marginal * divisor)) and \
                                                                  (current_fraction < 0.01)
                                                if rare_value_flag:
                                                    row_nums = six_d_row_nums  # todo: can remove some variables here
                                                    assert len(row_nums) == round(current_fraction * num_rows), \
                                                        f"len of matching rows: {len(row_nums)}, fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                                                    for r in row_nums:
                                                        # todo: i doubt this is threadsafe
                                                        outliers_6d_arr_for_i[r] += 1
                                                        # todo: use the actual values, not their index
                                                        outliers_explanation_arr_for_i[r] += f" [[[Columns: {i} {j} {k} {m} {n} {p} Values: {i_vals_idx} {j_vals_idx} {k_vals_idx} {m_vals_idx} {n_vals_idx} {p_vals_idx} Fraction: {current_fraction}]]]"
                                                local_rare_arr[i_vals_idx][j_vals_idx][k_vals_idx][m_vals_idx][n_vals_idx] = rare_value_flag
                        rare_arr_for_i[j][k][m][n][p] = local_rare_arr
    return rare_arr_for_i, outliers_6d_arr_for_i, outliers_explanation_arr_for_i, column_combos_checked_for_i


def flatten(arr):
    """
    Flatten a python array of any dimensionality into a 1d python array.
    """
    flatten_1d = lambda x: [i for row in x for i in row]
    if len(arr) == 0:
        return arr
    try:
        while True:
            arr = flatten_1d(arr)
            if len(arr) == 0:
                return arr
    except:
        pass  # todo: can this be removed?
    return arr


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


