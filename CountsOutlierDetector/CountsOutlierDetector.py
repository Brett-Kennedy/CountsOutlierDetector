import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from os import mkdir
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
import concurrent
# import cProfile
from statistics import mean
# from math import factorial
# from tqdm import tqdm
# from scipy.stats import poisson
# import matplotlib.pyplot as plt
# from numba import jit

'''
This file evaluates the presence of outliers in 3+ dimensions in the openml.org dataset collection
'''

np.random.seed(0)

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.width = 10000
pd.set_option("max_colwidth", None)

DIVISOR = 0.25  # todo: loop through different values of this to see how it affects the results.


def flatten(arr):
    flatten_1d = lambda x: [i for row in x for i in row]
    if len(arr) == 0:
        return arr
    try:
        while True:
            arr = flatten_1d(arr)
            if len(arr) == 0:
                return arr
    except:
        pass
    return arr


def is_float(v):
    if str(v).isdigit():
        return False
    try:
        float(v)
        return True
    except ValueError:
        return False


class CountsOutlierDetector:

    def __init__(self, n_bins=7, max_dimensions=5, results_folder="", results_name="", run_parallel=False):
        self.n_bins = n_bins
        self.max_dimensions = max_dimensions
        self.results_folder = results_folder
        self.results_name = results_name
        self.run_parallel = run_parallel

        self.col_types_arr = []
        self.ordinal_encoders_arr = []

    def get_col_types_arr(self, X):
        col_types_arr = ['N'] * len(X.columns)
        for c in range(len(X.columns)):
            num_unique = X[X.columns[c]].nunique()
            if not is_numeric_dtype(X[X.columns[c]]):
                col_types_arr[c] = 'C'
            # Even if the values are numeric, if there are few of them, consider them categorical, though if the values
            # are all float, the column will be cast to 'N' when collecting the unique values.
            elif is_numeric_dtype(X[X.columns[c]]) and num_unique <= 25:
                col_types_arr[c] = 'C'

        # If there are a large number of categorical columns, re-determine the types with a more strict cutoff
        if col_types_arr.count('C') > 50:
            col_types_arr = ['N'] * len(X.columns)
            for c in range(len(X.columns)):
                num_unique = X[X.columns[c]].nunique()
                if not is_numeric_dtype(X[X.columns[c]]):
                    col_types_arr[c] = 'C'
                elif is_numeric_dtype(X[X.columns[c]]) and num_unique <= 5:
                    col_types_arr[c] = 'C'

        return col_types_arr

    def ordinal_encode(self, X):
        # Numpy deals with numeric values much more efficiently than text
        self.ordinal_encoders_arr = [None]*len(X.columns)
        for i in range(len(X.columns)):
            if self.col_types_arr[i] == 'C':
                enc = OrdinalEncoder()
                self.ordinal_encoders_arr[i] = enc
                col_vals = X[X.columns[i]].values.reshape(-1, 1)
                X_np = enc.fit_transform(col_vals).astype(int)
                X[X.columns[i]] = X_np
        return X

    def get_col_value(self, col_idx, value_idx):
        if self.col_types_arr[col_idx] == "C":
            return self.ordinal_encoders_arr[col_idx].inverse_transform([[value_idx]])[0][0]
        else:
            return f"Bin {value_idx}"

    # Using numba appears to give similar performance results
    # @jit(nopython=True)
    # def get_cond(X_vals, col_idx, val):
    #     cond = (X_vals[:, col_idx] == val)
    #     return cond

    def predict(self, X):

        # todo: rename this -- it doesn't display
        def format_outlier_counts(msg, arr):
            nonlocal output_msg
            unique_counts = sorted(list(set(arr)))
            for uc in unique_counts:
                output_msg += f"\n{msg}: {uc}: {arr.count(uc):5}"

        # Given two columns i and j, gets, for each pair of values, the fraction of the dataset and the row numbers.
        def get_2d_fractions(i, j):
            two_d_fractions = []
            two_d_row_nums = []
            for i_val in unique_vals[i]:
                i_vals_fractions = []
                i_vals_row_nums = []
                cond1 = (X_vals[:, i] == i_val)
                for j_val in unique_vals[j]:
                    #rows_both = np.where((X_vals[:, i] == i_val) & (X_vals[:, j] == j_val))
                    cond2 = (X_vals[:, j] == j_val)
                    rows_both = np.where(cond1 & cond2)
                    i_vals_fractions.append(len(rows_both[0]) / num_rows)
                    i_vals_row_nums.append(rows_both[0])
                two_d_fractions.append(i_vals_fractions)
                two_d_row_nums.append(i_vals_row_nums)
            return two_d_fractions, two_d_row_nums

        def get_unique_vals():
            nonlocal output_msg

            # An element for each column. For categorical columns, lists the unique values. Used to maintain a
            # consistent order.
            unique_vals = [[]] * num_cols

            num_unique_vals = [0] * num_cols

            for i in range(num_cols):
                uv = X.iloc[:, i].unique()
                # If there are many unique values, remove the float values
                # todo: set this threshold as a parameter
                # todo: need to save the pre-ordinal encoded values to do this.
                # todo: or could not do it: then don't need to save the unique values, just the count and can assume it's
                #  0 up to that.
                #if len(uv) > 25:
                #    uv = [v for v in uv if not is_float(v)]
                col_threshold = (1.0 / len(uv)) * DIVISOR
                unique_vals[i] = uv
                num_unique_vals[i] = len(uv)
            return unique_vals, num_unique_vals

        def get_1d_stats():
            nonlocal output_msg

            # Parallel 2d array to unique_vals. Indicates the fraction of the total dataset for this value in this column.
            fractions_1d = [[]] * num_cols

            # Parallel 2d array to unique_vals. Boolean value indicates if the value is considered a 1d outlier.
            rare_1d_values = [[]] * num_cols

            # Integer value for each row in the dataset indicating how many individual columns have values considered
            # outliers.
            outliers_1d_arr = [0] * num_rows

            # Text explanation for each row explaining each 1d outlier.
            outliers_explanation_arr = [""] * num_rows

            for i in range(num_cols):

                col_threshold = (1.0 / num_unique_vals[i]) * DIVISOR

                # Array with an element for each unique value in column i. Indicates the fraction of the total dataset held
                # by that value.
                col_fractions_1d = []

                # Array with an element for each unique value in column i. Indicates if that value is considered rare.
                col_rare_1d_values = []

                for v in unique_vals[i]:  # loop through each unique value in the current column.
                    frac = X.iloc[:, i].tolist().count(v) / num_rows
                    col_fractions_1d.append(frac)
                    rare_values_flag = (frac < col_threshold) and (frac < 0.01)
                    if rare_values_flag:
                        rows_matching = np.where(X_vals[:, i] == v)
                        for r in rows_matching[0]:
                            outliers_1d_arr[r] += 1
                            outliers_explanation_arr[r] += f'[Column: {X.columns[i]}, ' + \
                                                           f'Value: {self.get_col_value(i,v)}, fraction: {frac}]'
                    col_rare_1d_values.append(rare_values_flag)
                fractions_1d[i] = col_fractions_1d
                rare_1d_values[i] = col_rare_1d_values

            output_msg += f"\n\n1d: num common values: {flatten(rare_1d_values).count(False)}"
            output_msg += f"\n1d: num rare values: {flatten(rare_1d_values).count(True)}"
            format_outlier_counts("1d: Outlier Counts by score", outliers_1d_arr)
            return fractions_1d, rare_1d_values, outliers_1d_arr, outliers_explanation_arr

        def get_2d_stats():
            nonlocal output_msg

            # This returns 2 parallel 4d arrays, fractions_2d and rare_2d_values, with the dimensions: i column,
            # j column, value in i column, value in j column

            # Each element stores the fraction of the total dataset with this combination of values.
            fractions_2d = [] * num_cols

            # Each element stores a boolean indicating if this combination of values is considered rare in the 2d sense.
            rare_2d_values = [] * num_cols

            # Integer value for each row in the dataset indicating how many pairs of columns have combinations considered
            # outliers.
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

                    # Determine which of these fraction would be considered rare in the 2d sense
                    i_rare_arr = []
                    expected_under_uniform = 1.0 / (len(unique_vals[i]) * len(unique_vals[j]))
                    for i_vals_idx in range(len(fractions_2d[i][j])):
                        j_rare_arr = []
                        for j_vals_idx in range(len(fractions_2d[i][j][i_vals_idx])):
                            current_fraction = fractions_2d[i][j][i_vals_idx][j_vals_idx]
                            expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx]
                            rare_value_flag = (rare_1d_values[i][i_vals_idx] == False) and \
                                              (rare_1d_values[j][j_vals_idx] == False) and \
                                              (current_fraction < (expected_under_uniform * DIVISOR)) and \
                                              (current_fraction < (expected_given_marginal * DIVISOR)) and \
                                              (current_fraction < 0.01)
                            if rare_value_flag:
                                row_nums = two_d_row_nums[i_vals_idx][j_vals_idx]
                                assert len(row_nums) == round(
                                    current_fraction * num_rows), f"len of matching rows: {len(row_nums)}, fraction*num_rows: current_fraction*num_rows: {current_fraction * num_rows}"
                                for r in row_nums:
                                    outliers_2d_arr[r] += 1
                                    # todo: format this for 3, 4, 5d too
                                    outliers_explanation_arr[r] += f" [[Columns: {X.columns[i]} and " +\
                                        f"{X.columns[j]} Values: {self.get_col_value(i, i_vals_idx)} and " +\
                                        f"{self.get_col_value(j, j_vals_idx)}, Fraction: {current_fraction}]]"
                            j_rare_arr.append(rare_value_flag)
                        i_rare_arr.append(j_rare_arr)
                    rare_2d_values[i][j] = i_rare_arr

            out = flatten(rare_2d_values)
            output_msg += f"\n\n2d: num common combinations: {out.count(False)}"
            output_msg += f"\n2d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            format_outlier_counts("2d: Outlier Counts by score", outliers_2d_arr)
            return fractions_2d, rare_2d_values, outliers_2d_arr, outliers_explanation_arr

        def get_3d_stats(num_combinations):
            nonlocal output_msg

            # This returns 2 parallel 6d arrays: fractions_3d and rare_3d_values (with the dimensions: i column, j column,
            # k column, value in i column, value in j column, value in the k column), as well as outliers_3d_arr and
            # outliers_explanation_arr.
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
                                            X_vals,
                                            num_cols,
                                            num_rows,
                                            unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values)
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
                        X_vals,
                        num_cols,
                        num_rows,
                        unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values
                    )
                    rare_3d_values[i] = rare_arr_for_i
                    outliers_3d_arr = [x + y for x, y in zip(outliers_3d_arr, outliers_3d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_3d_values)
            output_msg += f"\n\n3d: num common combinations: {out.count(False)}"
            output_msg += f"\n3d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            format_outlier_counts("3d: Outlier Counts by score", outliers_3d_arr)
            return fractions_3d, rare_3d_values, outliers_3d_arr, outliers_explanation_arr, column_combos_checked

        def get_4d_stats(num_combinations):
            nonlocal output_msg

            # This returns 2 parallel 8d arrays: fractions_4d and rare_4d_values (with the dimensions: i column, j column,
            # k column, m column, value in i column, value in j column, value in the k column, value in the m column),
            # as well as outliers_43d_arr and  outliers_explanation_arr.
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
                                            X_vals,
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
                        X_vals,
                        num_cols,
                        num_rows,
                        unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        rare_3d_values
                    )
                    rare_4d_values[i] = rare_arr_for_i
                    outliers_4d_arr = [x + y for x, y in zip(outliers_4d_arr, outliers_4d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_4d_values)
            output_msg += f"\n\n4d: num common combinations: {out.count(False)}"
            output_msg += f"\n4d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            format_outlier_counts("4d: Outlier Counts by score", outliers_4d_arr)
            return fractions_4d, rare_4d_values, outliers_4d_arr, outliers_explanation_arr, column_combos_checked

        def get_5d_stats(num_combinations):
            nonlocal output_msg

            # todo: update this comment. Make more general, so don't repeat it
            # This returns 2 parallel 8d arrays: fractions_5d and rare_5d_values (with the dimensions: i column, j column,
            # k column, m column, value in i column, value in j column, value in the k column, value in the m column),
            # as well as outliers_5d_arr and  outliers_explanation_arr.
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
                                            X_vals,
                                            num_cols,
                                            num_rows,
                                            unique_vals,
                                            fractions_1d,
                                            rare_1d_values,
                                            rare_2d_values,
                                            rare_3d_values,
                                            rare_4d_values)
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
                        X_vals,
                        num_cols,
                        num_rows,
                        unique_vals,
                        fractions_1d,
                        rare_1d_values,
                        rare_2d_values,
                        rare_3d_values,
                        rare_4d_values
                    )
                    rare_5d_values[i] = rare_arr_for_i
                    outliers_5d_arr = [x + y for x, y in zip(outliers_5d_arr, outliers_5d_arr_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, outliers_explanation_arr_for_i)]
                    column_combos_checked += column_combos_checked_for_i

            out = flatten(rare_5d_values)
            output_msg += f"\n\n5d: num common combinations: {out.count(False)}"
            output_msg += f"\n5d: num rare combinations: {out.count(True)} (Typically most with zero rows)"
            format_outlier_counts("5d: Outlier Counts by score", outliers_5d_arr)
            return fractions_5d, rare_5d_values, outliers_5d_arr, outliers_explanation_arr, column_combos_checked

        def create_output_csv(outliers_1d_arr,
                              outliers_2d_arr,
                              outliers_3d_arr,
                              outliers_4d_arr,
                              outliers_5d_arr,
                              explanations_1d_arr,
                              explanations_2d_arr,
                              explanations_3d_arr,
                              explanations_4d_arr,
                              explanations_5d_arr):
            if self.results_folder != "":
                try:
                    mkdir(self.results_folder)
                except FileExistsError as e:
                    pass
                except Exception as e:
                    print(f"Error creating results folder: {e}")

            # todo: de-encode from the ordinal values in teh explanations
            df = pd.DataFrame({"1d Counts": outliers_1d_arr,
                               "2d Counts": outliers_2d_arr,
                               "3d Counts": outliers_3d_arr,
                               "4d Counts": outliers_4d_arr,
                               "5d Counts": outliers_5d_arr,
                               "1d Explanations": explanations_1d_arr,
                               "2d Explanations": explanations_2d_arr,
                               "3d Explanations": explanations_3d_arr,
                               "4d Explanations": explanations_4d_arr,
                               "5d Explanations": explanations_5d_arr
                               })
            df['Any at 1d'] = df['1d Counts'] > 0
            df['Any at 2d'] = df['2d Counts'] > 0
            df['Any at 3d'] = df['3d Counts'] > 0
            df['Any at 4d'] = df['4d Counts'] > 0
            df['Any at 5d'] = df['5d Counts'] > 0

            df['Any up to 1d'] = df['1d Counts'] > 0
            df['Any up to 2d'] = df['Any up to 1d'] | df['2d Counts'] > 0
            df['Any up to 3d'] = df['Any up to 2d'] | df['3d Counts'] > 0
            df['Any up to 4d'] = df['Any up to 3d'] | df['4d Counts'] > 0
            df['Any up to 5d'] = df['Any up to 4d'] | df['5d Counts'] > 0

            df['Any Scored'] = (df['1d Counts'] + df['2d Counts'] + df['3d Counts'] + df['4d Counts'] + df['5d Counts']) > 0

            if self.results_folder != "":
                n = datetime.now()
                dt_string = n.strftime("%d_%m_%Y_%H_%M_%S")
                file_name = self.results_folder + "\\" + self.results_name + "_results_" + dt_string + ".csv"
                df.to_csv(file_name)

            return df

        ################################
        # Start of code
        ################################

        # Bin any numeric columns
        self.col_types_arr = self.get_col_types_arr(X)
        numeric_col_names = []
        for c in range(len(self.col_types_arr)):
            if self.col_types_arr[c] == 'N':
                numeric_col_names.append(X.columns[c])
        # todo: test with k-means as the strategy
        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        if len(numeric_col_names):
            X_num = X[numeric_col_names]
            Xt = est.fit_transform(X_num)
            for num_idx, col_name in enumerate(numeric_col_names):
                X[col_name] = Xt[:, num_idx].astype(int)

        # Remove any columns with 1 unique value or a very large number of unique values
        # todo: make these limits parameters
        col_names_arr = []
        for c in range(len(X.columns)):
            if X[X.columns[c]].nunique() < 2 or X[X.columns[c]].nunique() > 50:
                col_names_arr.append(X.columns[c])
        X = X.drop(columns=col_names_arr)
        num_cols = len(X.columns)
        num_rows = len(X)

        #output_msg = print_header(dataset_index, dataset_name)
        output_msg = f"\nNumber of rows: {num_rows}"
        output_msg += f"\nNumber of columns: {num_cols}"

        # Create a summary of this run, giving statistics about the outliers found
        run_summary_df = pd.DataFrame(columns=[
            'Percent Flagged as 1d',
            'Percent Flagged as 2d',
            'Percent Flagged as 3d',
            'Percent Flagged as 4d',
            'Percent Flagged as 5d',

            'Percent Flagged up to 1d',
            'Percent Flagged up to 2d',
            'Percent Flagged up to 3d',
            'Percent Flagged up to 4d',
            'Percent Flagged up to 5d',

            'Checked_3d',  # False if too many combinations to even check
            'Checked_4d',
            'Checked_5d',

            '3d column combos checked',  # Skip column combinations where expected count based on marginal probs is too low.
            '4d column combos checked',
            '5d column combos checked',

            'Percent Flagged'])

        if num_cols < 2:
            output_msg += "\nLess than two categorical columns found. Cannot determine outliers"
            return output_msg, run_summary_df

        X = self.ordinal_encode(X)
        X_vals = X.values
        unique_vals, num_unique_vals, = get_unique_vals()
        output_msg += f"\nCardinality of the columns: {num_unique_vals}"

        # Determine the 1d stats
        fractions_1d, rare_1d_values, outliers_1d_arr, explanations_1d_arr = get_1d_stats()

        # Determine the 2d stats
        fractions_2d, rare_2d_values, outliers_2d_arr, explanations_2d_arr = get_2d_stats()

        # Determine the 3d stats unless there are too many columns and unique values to do so efficiently
        checked_3d = False
        column_combos_checked_3d = -1
        avg_num_unique_vals = mean([len(x) for x in unique_vals])
        num_combinations = (num_cols*(num_cols-1)*(num_cols-2)) * pow(avg_num_unique_vals, 3)
        if num_combinations > 100_000_000:  # todo: set this as a parameter
            output_msg += (f"\n\nCannot determine 3d outliers given the number of categorical columns ({num_cols}) and" +
                  "number of unique values in each.")
            outliers_3d_arr = [0] * num_rows
            explanations_3d_arr = [""] * num_rows
        else:
            fractions_3d, rare_3d_values, outliers_3d_arr, explanations_3d_arr, column_combos_checked_3d = \
                get_3d_stats(num_combinations=num_combinations)
            checked_3d = True

        # Determine the 4d stats unless there are too many columns and unique values to do so efficiently
        # todo here and above just use pow method
        checked_4d = False
        column_combos_checked_4d = -1
        num_combinations = (num_cols*(num_cols-1)*(num_cols-2)*(num_cols-3)) * pow(avg_num_unique_vals, 4)
        outliers_4d_arr = [0] * num_rows
        explanations_4d_arr = [""] * num_rows
        if num_cols < 4:
            output_msg += f"\n\nCannot determine 4d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
        elif num_combinations > 100_000_000:  # todo: set this as a parameter
            output_msg += f"\n\nCannot determine 4d outliers given the number of categorical columns ({num_cols}) and number of unique values in each."
        else:
            fractions_4d, rare_4d_values, outliers_4d_arr, explanations_4d_arr, column_combos_checked_4d = \
                get_4d_stats(num_combinations=num_combinations)
            checked_4d = True

        # Determine the 5d stats unless there are too many columns and unique values to do so efficiently
        checked_5d = False
        column_combos_checked_5d = -1
        num_combinations = (num_cols*(num_cols-1)*(num_cols-2)*(num_cols-3)*(num_cols-4)) * pow(avg_num_unique_vals, 5)
        outliers_5d_arr = [0] * num_rows
        explanations_5d_arr = [""] * num_rows
        if num_cols < 5:
            output_msg += f"\n\nCannot determine 5d outliers. Too few columns: {num_cols}."  # todo: these are printing before the output for 1d, 2d, 3d
        elif num_combinations > 100_000_000:  # todo: set this as a parameter
            output_msg += f"\n\nCannot determine 5d outliers given the number of categorical columns ({num_cols}) and number of unique values in each."
        else:
            fractions_5d, rare_5d_values, outliers_5d_arr, explanations_5d_arr, column_combos_checked_5d = \
                get_5d_stats(num_combinations=num_combinations)
            checked_5d = True

        flagged_rows_df = create_output_csv(
            outliers_1d_arr, outliers_2d_arr, outliers_3d_arr, outliers_4d_arr, outliers_5d_arr,
            explanations_1d_arr, explanations_2d_arr, explanations_3d_arr, explanations_4d_arr, explanations_5d_arr)

        num_rows_scored = list(flagged_rows_df['Any at 1d'] > 0).count(True)
        output_msg += f"\n\nNumber of rows flagged as outliers examining 1d: {num_rows_scored}" +\
                      f" ({round(num_rows_scored*100.0/num_rows,1)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 2d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 2d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,1)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 3d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 3d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,1)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 4d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 4d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,1)}%)"
        num_rows_scored = list(flagged_rows_df['Any at 5d'] > 0).count(True)
        output_msg += f"\nNumber of rows flagged as outliers examining 5d: {num_rows_scored} " +\
                      f"({round(num_rows_scored*100.0/num_rows,1)}%)"

        # Update run_summary_df
        run_summary_df = run_summary_df.append(pd.DataFrame(np.array([[
                flagged_rows_df['Any at 1d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 2d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 3d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 4d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any at 5d'].sum() * 100.0 / num_rows,

                flagged_rows_df['Any up to 1d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 2d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 3d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 4d'].sum() * 100.0 / num_rows,
                flagged_rows_df['Any up to 5d'].sum() * 100.0 / num_rows,

                checked_3d,
                checked_4d,
                checked_5d,
                column_combos_checked_3d,
                column_combos_checked_4d,
                column_combos_checked_5d,
                flagged_rows_df['Any Scored'].sum() * 100.0 / num_rows]]),
            columns=run_summary_df.columns))

        row_explanations = self.output_explanations(flagged_rows_df)

        return flagged_rows_df, row_explanations, output_msg, run_summary_df

    def output_explanations(self, flagged_rows_df):
        df_subset = flagged_rows_df[flagged_rows_df['Any Scored']]
        expl_arr = []
        index_arr = list(df_subset.index)
        for i in range(len(df_subset)):
            row = df_subset.iloc[i]
            row_expl=[index_arr[i], "", "", "", "", ""]
            for i in range(1, self.max_dimensions+1):
                col_name = f"{i}d Explanations"
                row_expl[i] = row[col_name]
            expl_arr.append(row_expl)
        expl_df = pd.DataFrame(expl_arr, columns=['Row Index', '1d Explanations', '2d Explanations', '3d Explanations',
                                                  '4d Explanations', '5d Explanations'])
        return expl_df

# These methods are outside the class so can be called as concurrent processes
def process_inner_loop_3d(
        i,
        X_vals,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values):

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
                cond1 = (X_vals[:, i] == i_val)
                for j_vals_idx in range(num_unique_vals_j):
                    if rare_1d_values[j][j_vals_idx]:
                        continue
                    if rare_2d_values[i][j][i_vals_idx][j_vals_idx]:
                        continue
                    j_val = unique_vals[j][j_vals_idx]
                    cond2 = (X_vals[:, j] == j_val)
                    for k_vals_idx in range(num_unique_vals_k):
                        if rare_1d_values[k][k_vals_idx]:
                            continue
                        if rare_2d_values[i][k][i_vals_idx][k_vals_idx]:
                            continue
                        if rare_2d_values[j][k][j_vals_idx][k_vals_idx]:
                            continue
                        k_val = unique_vals[k][k_vals_idx]
                        cond3 = (X_vals[:, k] == k_val)
                        rows_all = np.where(cond1 & cond2 & cond3)
                        current_fraction = len(rows_all[0]) / num_rows
                        three_d_row_nums = rows_all[0]

                        expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx] * \
                                                  fractions_1d[k][k_vals_idx]
                        rare_value_flag = (current_fraction < (expected_under_uniform * DIVISOR)) and \
                                          (current_fraction < (expected_given_marginal * DIVISOR)) and \
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
        X_vals,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values,
        rare_3d_values
):
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
                    cond1 = (X_vals[:, i] == i_val)
                    for j_vals_idx in range(num_unique_vals_j):
                        if rare_1d_values[j][j_vals_idx]:
                            continue
                        j_val = unique_vals[j][j_vals_idx]
                        cond2 = (X_vals[:, j] == j_val)
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
                            cond3 = (X_vals[:, k] == k_val)
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
                                cond4 = (X_vals[:, m] == m_val)
                                rows_all = np.where(cond1 & cond2 & cond3 & cond4)
                                current_fraction = len(rows_all[0]) / num_rows
                                four_d_row_nums = rows_all[0] # todo: use less variables

                                expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx] * \
                                                          fractions_1d[k][k_vals_idx] * fractions_1d[m][m_vals_idx]
                                rare_value_flag = (current_fraction < (expected_under_uniform * DIVISOR)) and \
                                                  (current_fraction < (expected_given_marginal * DIVISOR)) and \
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
        X_vals,
        num_cols,
        num_rows,
        unique_vals,
        fractions_1d,
        rare_1d_values,
        rare_2d_values,
        rare_3d_values,
        rare_4d_values
):
    num_unique_vals_i = len(unique_vals[i])
    outliers_5d_arr_for_i = [0] * num_rows
    outliers_explanation_arr_for_i = [""] * num_rows
    rare_arr_for_i = [[[[[]]*num_cols]*num_cols]*num_cols]*num_cols
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
                    local_rare_arr = [[[[[False]*num_unique_vals_n]*num_unique_vals_m]*num_unique_vals_k]*num_unique_vals_j]*num_unique_vals_i
                    for i_vals_idx in range(num_unique_vals_i):
                        if rare_1d_values[i][i_vals_idx]:
                            continue
                        i_val = unique_vals[i][i_vals_idx]
                        cond1 = (X_vals[:, i] == i_val)
                        for j_vals_idx in range(num_unique_vals_j):
                            if rare_1d_values[j][j_vals_idx]:
                                continue
                            j_val = unique_vals[j][j_vals_idx]
                            cond2 = (X_vals[:, j] == j_val)
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
                                cond3 = (X_vals[:, k] == k_val)
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
                                    cond4 = (X_vals[:, m] == m_val)
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
                                        cond5 = (X_vals[:, n] == n_val)

                                        rows_all = np.where(cond1 & cond2 & cond3 & cond4 & cond5)
                                        current_fraction = len(rows_all[0]) / num_rows
                                        five_d_row_nums = rows_all[0] # todo: use less variables

                                        expected_given_marginal = fractions_1d[i][i_vals_idx] * fractions_1d[j][j_vals_idx] * \
                                                                  fractions_1d[k][k_vals_idx] * fractions_1d[m][m_vals_idx] * fractions_1d[n][n_vals_idx]
                                        rare_value_flag = (current_fraction < (expected_under_uniform * DIVISOR)) and \
                                                          (current_fraction < (expected_given_marginal * DIVISOR)) and \
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


