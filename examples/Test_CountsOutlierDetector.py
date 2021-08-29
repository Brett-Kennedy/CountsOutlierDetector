import pandas as pd
import numpy as np
from datetime import datetime
import cProfile
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, 'C:\\python_projects\\DatasetsEvaluator_project\\DatasetsEvaluator')
import DatasetsEvaluator as de

sys.path.insert(0, 'C:\\python_projects\\CountsOutlierDetection_project\\CountsOutlierDetector')
from CountsOutlierDetector import CountsOutlierDetector

'''
This file evaluates the presence of outliers in 3+ dimensions in the openml.org dataset collection
'''

np.random.seed(0)

pd.options.display.max_columns = 1000
pd.options.display.max_rows = 1000
pd.options.display.width = 10000

PROFILE_RUN = False
RUN_PARALLEL = True
NUM_DATASETS_EVALUATED = 3  #todo: seems to stop at 124 when set to 200
TEST_SINGLE_DATASET = ""
#TEST_SINGLE_DATASET = "mushroom"

results_folder = "c:\\outlier_results"


def print_header(dataset_index, dataset_name):
    stars = "********************************************"
    msg = f"\n\n{stars}\n{dataset_index}: {dataset_name}\n{stars}"
    return msg


# todo: add a 2nd method to get the data for numeric columns
def load_categorcial_datasets():
    cache_folder = "c:\\dataset_cache"

    if TEST_SINGLE_DATASET == "synth_1":
        return generate_synth_1()

    datasets_tester = de.DatasetsTester(problem_type="both", path_local_cache=cache_folder)
    if TEST_SINGLE_DATASET:
        matching_datasets = datasets_tester.find_by_name([TEST_SINGLE_DATASET])
    else:
        matching_datasets = datasets_tester.find_datasets(
            use_cache=True,
            min_num_features=0,
            max_num_features=np.inf,
            min_num_instances=500,
            max_num_instances=50_000,
            min_num_numeric_features=0,
            max_num_numeric_features=np.inf,  
            min_num_categorical_features=3,
            max_num_categorical_features=np.inf)

    print("Number matching datasets found: ", len(matching_datasets))

    # Note: some datasets may have errors loading or testing.
    datasets_tester.collect_data(max_num_datasets_used=NUM_DATASETS_EVALUATED,
                                 # Files known to not load properly, excluding to reduce time
                                 # todo: I'm using the automatic exclude list feature instead, should be able to remove this
                                 exclude_list=['PCam16k', 'realAdExchange2', 'FOREX_cadchf-day-High', 'lotto',
                                               'titanic_1', 'dataset_time_3', 'FOREX_eurgbp-day-Close',
                                               'FOREX_eurdkk-day-Close'],
                                 use_automatic_exclude_list=True,
                                 max_cat_unique_vals=np.inf,
                                 check_local_cache=True,
                                 check_online=True,   # Don't generally set to False, but sometimes to be faster
                                 save_local_cache=True,
                                 one_hot_encode=False)
    return datasets_tester.get_dataset_collection()


def run_tests(datasets):
    # todo: call print_header in here, not in the detector. i've moved the func to this file.
    '''
    run_summary_df = pd.DataFrame(columns=[
        'Dataset Index',
        'Dataset Name',

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
    '''

    run_summary_df = None
    for dataset_dict in datasets:
        dataset_index = dataset_dict['Index']
        dataset_name = dataset_dict['Dataset_name']
        X = dataset_dict['X']
        det = CountsOutlierDetector(results_folder=results_folder, results_name=dataset_name, run_parallel=RUN_PARALLEL)
        flagged_rows_df, row_explanations, output_msg, dataset_run_summary_df = det.predict(X)
        if run_summary_df is None:
            run_summary_df = dataset_run_summary_df
            run_summary_df.insert(0, 'Dataset Index', dataset_index)
            run_summary_df.insert(1, 'Dataset Name', dataset_name)
        else:
            run_summary_df = run_summary_df.append(dataset_run_summary_df, ignore_index=True)
            run_summary_df.loc[run_summary_df.index[-1], 'Dataset Index'] = dataset_index
            run_summary_df.loc[run_summary_df.index[-1], 'Dataset Name'] = dataset_name
        print(output_msg)
    return run_summary_df


def test_data(datasets):
    global RUN_PARALLEL

    # Profiling is not possible with multiprocessing, as the code executed by processes other than the main
    # are not included.
    if PROFILE_RUN:
        RUN_PARALLEL = False
        with cProfile.Profile() as pr:
            run_summary_df = run_tests(datasets)
        pr.print_stats()
    else:
        t1 = datetime.now()
        run_summary_df = run_tests(datasets)
        t2 = datetime.now()
        print("\n\n\nTime for total run: ", t2-t1)

    # Save run_summary_df
    n = datetime.now()
    dt_string = n.strftime("%d_%m_%Y_%H_%M_%S")
    file_name = results_folder + "\\run_summary_" + dt_string + ".csv"
    run_summary_df.to_csv(file_name)

    return run_summary_df


# todo: set max_dimensions as a parameter

def output_run_summary(run_summary_df):
    sns.set()

    print(run_summary_df)
    run_summary_df = run_summary_df.dropna()
    if len(run_summary_df) == 0:
        return

    print("avg Percent Flagged as 1d: ", run_summary_df['Percent Flagged as 1d'].astype(float).mean())
    print("avg Percent Flagged as 2d: ", run_summary_df['Percent Flagged as 2d'].astype(float).mean())
    print("avg Percent Flagged as 3d: ", run_summary_df['Percent Flagged as 3d'].astype(float).mean())
    print("avg Percent Flagged as 4d: ", run_summary_df['Percent Flagged as 4d'].astype(float).mean())
    print("avg Percent Flagged as 5d: ", run_summary_df['Percent Flagged as 5d'].astype(float).mean())

    ############################################################################
    # Plot the average number of rows flagged at each dimension as a bar graph
    heights_arr = [run_summary_df['Percent Flagged as 1d'].astype(float).mean(),
                   run_summary_df['Percent Flagged as 2d'].astype(float).mean(),
                   run_summary_df['Percent Flagged as 3d'].astype(float).mean(),
                   run_summary_df['Percent Flagged as 4d'].astype(float).mean(),
                   run_summary_df['Percent Flagged as 5d'].astype(float).mean()]
    fig, ax = plt.subplots()
    ax.bar(
        x=range(1, 6),
        height=heights_arr,
        tick_label=[f"avg % Flagged examining {x}d: " for x in range(1, 6)]
    )
    plt.xticks(rotation=45, ha='right')
    ax.set_yticks(range(0, 110, 10))
    ax.set_title("Average percent of rows flagged at each dimension")
    results_plot_filename = results_folder + "\\counts_by_dim_bar.png"
    fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

    ############################################################################
    # Plot the average number of rows flagged examining up to each dimension as a bar graph
    heights_arr = [run_summary_df['Percent Flagged up to 1d'].astype(float).mean(),
                   run_summary_df['Percent Flagged up to 2d'].astype(float).mean(),
                   run_summary_df['Percent Flagged up to 3d'].astype(float).mean(),
                   run_summary_df['Percent Flagged up to 4d'].astype(float).mean(),
                   run_summary_df['Percent Flagged up to 5d'].astype(float).mean()]
    fig, ax = plt.subplots()
    ax.bar(
        x=range(1, 6),
        height=heights_arr,
        tick_label=[f"avg % Flagged examining up to {x}d: " for x in range(1, 6)]
    )
    plt.xticks(rotation=45, ha='right')
    ax.set_yticks(range(0, 110, 10))
    ax.set_title("Average percent of rows flagged up to and including each dimension")
    results_plot_filename = results_folder + "\\counts_up_to_dim_bar.png"
    fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

    ############################################################################
    # Plot the percent of rows flagged in some way as the # of dimensions examined is increased
    # for each dataset, drawing one line per dataset.
    fig, ax = plt.subplots()
    for r in range(len(run_summary_df)):
        row = run_summary_df.iloc[r]
        x_coords = list(range(1, 6))
        y_coords = [row['Percent Flagged up to 1d'],
                    row['Percent Flagged up to 2d'],
                    row['Percent Flagged up to 3d'],
                    row['Percent Flagged up to 4d'],
                    row['Percent Flagged up to 5d']]
        y_coords = [float(x) for x in y_coords]
        ax.plot(x_coords, y_coords, label=row['Dataset Name'])
    fig.tight_layout()
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([str(x) for x in range(1, 6)])
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.set_xlabel("Number dimensions")
    ax.set_ylabel("% of Rows Flagged")
    ax.set_title("Percent of Rows Flagged Given # Dimensions Examined")
    results_plot_filename = results_folder + "\\count_up_to_dim_line.png"
    fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

    ############################################################################
    # Plot the number of cases where don't check because too many combinations
    heights_arr = [100.0,
                   100.0,
                   run_summary_df['Checked_3d'].map({'True': True, 'False': False}).sum() * 100.0 / len(run_summary_df),
                   run_summary_df['Checked_4d'].map({'True': True, 'False': False}).sum() * 100.0 / len(run_summary_df),
                   run_summary_df['Checked_5d'].map({'True': True, 'False': False}).sum() * 100.0 / len(run_summary_df)]
    fig, ax = plt.subplots()
    ax.bar(
        x=range(1, 6),
        height=heights_arr,
        tick_label=[f"% Datasets examining up to {x}d: " for x in range(1, 6)]
    )
    plt.xticks(rotation=45, ha='right')
    ax.set_yticks(range(0, 110, 10))
    ax.set_title("Percent of Datasets Examining Each Dimension (not skipped due to size limits)")
    results_plot_filename = results_folder + "\\percent_datasets_checked_by_dim_bar.png"
    fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)

    # todo: plot the number of cases where don't check because the expected under uniform is too low

    # todo: plot the number of cases where don't check because the expected given marginal distribs is too low


def main():
    datasets = load_categorcial_datasets()
    run_summary_df = test_data(datasets)
    output_run_summary(run_summary_df)

if __name__ == "__main__":
    main()

