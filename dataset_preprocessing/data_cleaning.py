import os
import numpy as np
import pandas as pd
from config import MISSING_MARKER
from utils import DELTA  # assuming DELTA is defined there
from utils import convert_to_minutes

def apply_ignore_conditions(data_mat, ign_cond, missing_marker=MISSING_MARKER, delta=DELTA):
    """
    Parameters:
    data_mat : numpy.ndarray
        The array to process. This function modifies the array in-place.
    ign_cond : str
        A string that defines the ignore condition. Supported values are:
            - ''           : Do not apply any range-based filtering; only convert missing_marker to NaN.
            - '<=0'        : Mark as NaN values that are effectively non-positive (x < delta).
            - '<0&>7'      : Mark as NaN values that are either less than -delta or greater than 7 + delta.
            - '<6&>60'     : Mark as NaN values that are either less than 6 - delta or greater than 60 + delta.
            - '<25&>220'   : Mark as NaN values that are either less than 25 - delta or greater than 220 + delta.
            - '<0&>300'    : Mark as NaN values that are either less than -delta or greater than 300 + delta.
            - '<=0&>100'   : Mark as NaN values that are either effectively non-positive (x < delta)
                            or greater than 100 + delta.
    missing_marker : float, optional
        A default missing marker in the data (default is -200).
    delta : float, optional
        A tolerance value to deal with floating-point precision issues.
    """
    # Map each condition string to a corresponding lambda function that returns a boolean mask.
    condition_map = {
        '<=0':       lambda x: x < delta,
        # '<=25':      lambda x: x < 25 + delta,
        '<0&>7':     lambda x: np.logical_or(x < -delta, x > 7 + delta),
        '<6&>60':     lambda x: np.logical_or(x < 6 - delta, x > 60 + delta),
        '<25&>220':     lambda x: np.logical_or(x < 25 -  delta, x > 220 + delta),
        '<0&>300':     lambda x: np.logical_or(x < -delta, x > 300 + delta),
        '<=0&>100':  lambda x: np.logical_or(x < delta, x > 100 + delta),
    }
    # If the condition is one of the defined ones, apply it.
    if ign_cond in condition_map:
        mask = condition_map[ign_cond](data_mat)
        data_mat[mask] = np.nan
    elif ign_cond == '':
        pass
    else:
        # You might want to decide how to handle unsupported conditions.
        raise ValueError(f"Unsupported ignore condition: {ign_cond}")
    # Convert any remaining default missing markers to NaN.
    mask = np.isclose(data_mat, missing_marker, atol=delta)
    data_mat[mask] = np.nan
    return data_mat

def load_feature_data(chosen_pid_folder, date_str, feature_dict_list, num_time_samples,
                      log_file_path=None):
    """
    Load and process the feature data.
    Returns a matrix of shape (num_time_samples, number_of_features).
    """
    num_features = len(feature_dict_list)
    feature_mat = np.full((num_time_samples, num_features), np.nan)
    for feature_idx, feature_dict in enumerate(feature_dict_list):
        num_options = len(feature_dict['csv_list'])
        options_mat = np.full((num_time_samples, num_options), np.nan)
        for option_idx in range(num_options):
            csv_name = feature_dict['csv_list'][option_idx]
            source_name = feature_dict['source_list'][option_idx]
            full_path = os.path.join(chosen_pid_folder, csv_name + '.csv')
            df = pd.read_csv(full_path)
            if df.empty:
                assert csv_name == 'step-log', f"DataFrame is of type {csv_name}."
                # raise ValueError(f"DataFrame from {full_path} is empty.")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"DataFrame from {full_path} is empty.\n")
                continue
            # Filter dataframe by timestamp date_str
            chosen_df = df[df['Start Time (Local)'].str.contains(date_str)]
            if chosen_df.empty:
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{chosen_pid_folder}: No data for date {date_str} in file {full_path}\n")
                continue
            # Filter by subject and source (if provided)
            if source_name:
                chosen_df = chosen_df[(chosen_df["Source"] == source_name)]
                # chosen_df = df[(df["User First Name"] == chosen_pid) & (df["Source"] == source_name)]
            # else:
                # chosen_df = df[df["User First Name"] == chosen_pid]
            timestamps = chosen_df['Start Time (Local)']
            minutes_list = np.array([convert_to_minutes(t) for t in timestamps])
            if minutes_list.size:
                assert np.all(minutes_list < num_time_samples), "One or more timestamp indices are out-of-range!"
                feature_data = np.array(chosen_df[feature_dict['feature']], dtype=float)
                options_mat[minutes_list, option_idx] = feature_data
        options_mat = apply_ignore_conditions(options_mat, feature_dict['ign_cond'])
        # Choose the option (i.e. CSV file) that has the most valid (non-NaN) entries.
        option_valid_counts = np.sum(~np.isnan(options_mat), axis=0)
        best_option = np.argmax(option_valid_counts)
        feature_mat[:, feature_idx] = options_mat[:, best_option]
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Feature: {feature_dict['feature']}, Chosen CSV: {feature_dict['csv_list'][best_option]}, "
                           f"Chosen Source: {feature_dict['source_list'][best_option]}\n")
    return feature_mat

def load_hrv_data(hrv_folder, chosen_pid, chosen_day, hrv_name_list,
                  num_time_samples, date_str, hrv_fname='hrv_indices.csv',
                  log_file_path=None):  
    """
    Load and process HRV data.
    Returns a matrix of shape (num_time_samples, number_of_hrv_features).
    """
    # Initialize the HRV matrix with NaNs.
    hrv_data = np.full((num_time_samples, len(hrv_name_list)), np.nan)
    hrv_file_path = os.path.join(hrv_folder, chosen_pid, chosen_day, hrv_fname) # modified!!
    hrv_df = pd.read_csv(hrv_file_path)
    if hrv_df.empty:
        raise ValueError(f"HRV DataFrame from {hrv_file_path} is empty.")
    # Filter HRV data to the chosen subject and day.
    # day_str_format = compute_day_str(chosen_day, first_date_str)
    hrv_pid_df = hrv_df[(hrv_df['Subject'] == chosen_pid) & (hrv_df['Start Time'].str.contains(date_str))]
    # assert not hrv_pid_df.empty, f"No HRV data for pid={chosen_pid} on day={chosen_day} in file {hrv_file_path}"
    if hrv_pid_df.empty:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"No HRV data for pid={chosen_pid} on day={chosen_day} in file {hrv_file_path}\n")
        return hrv_data
    timestamps = hrv_pid_df['Start Time']
    minutes_list = np.array([convert_to_minutes(t, fmt='YYYY/MM/DD HH:MM:SS') for t in timestamps])
    if minutes_list.size:
        assert np.all(minutes_list < num_time_samples), "One or more timestamp indices are out-of-range!"
        # Place HRV measurements in the corresponding time slots.
        hrv_data[minutes_list, :] = hrv_pid_df[hrv_name_list] 
    # convert all with MISSING_MARKER values to NaN
    hrv_data = apply_ignore_conditions(hrv_data, '', MISSING_MARKER)
    return hrv_data

def load_sleep_feature_data(read_path, start_time_index, end_time_index, num_time_samples):
    feature_df = pd.read_csv(read_path)
    assert len(feature_df) == num_time_samples
    subset_df = feature_df.iloc[start_time_index:end_time_index+1]
    return subset_df