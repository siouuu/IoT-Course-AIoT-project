from os import listdir
from os.path import isfile, join
import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score



def sliding_window_pd(
        df,
        ws=500,
        overlap=250,
        w_type="hann",
        w_center=True,
        print_stats=False
) -> list:
    """Applies the sliding window algorithm to the DataFrame rows.

    Args:
        df: The DataFrame with all the values that will be inserted to the
            sliding window algorithm.
        ws: The window size in number of samples.
        overlap: The hop length in number of samples.
        w_type: The windowing function.
        w_center: If False, set the window labels as the right edge of the
            window index. If True, set the window labels as the center of the
            window index.
        print_stats: Print statistical inferences from the process. Defaults
            to False.

    Returns:
        A list of DataFrames each one corresponding to a produced window.
    """
    counter = 0
    windows_list = list()
    # min_periods: Minimum number of observations in window required to have
    # a value;
    # For a window that is specified by an integer, min_periods will default
    # to the size of the window.
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))

    return windows_list


def apply_filter(
        arr,
        order=5,
        wn=0.1,
        filter_type="lowpass"
) -> np.ndarray:
    """Applies filter to the multi-axis signal.

    Args:
        arr: The initial NumPy signal array values.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:
        NumPy Array with the filtered signal.
    """
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type,
                                     output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)

    return filtered_signal


def filter_instances(
        instances_list,
        order,
        wn,
        filter_type
) -> list:
    """Applies filter to a list of windows (each window is a DataFrame).

    Args:
        instances_list: List of DataFrames.
        order: The order of the filter.
        wn: The critical frequency or frequencies.
        filter_type: The type of filter. {‘lowpass’, ‘highpass’, ‘bandpass’,
            ‘bandstop’}

    Returns:

    """
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.apply(apply_filter,
                                       args=(order, wn, filter_type)
                                       )
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:",
          len(filtered_instances_list)
          )

    return filtered_instances_list


def flatten_instances_df(instances_list: list) -> pd.DataFrame:
    """Flattens each instance and create a DataFrame with the whole flattened
        instances.

    Args:
        instances_list: The list of DataFrames to be flattened

    Returns:
        A DataFrame that includes the whole flattened DataFrames
    """
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)

    return df


def df_rebase(
        df: pd.DataFrame,
        target_list: list,
        ref_list: list
) -> pd.DataFrame:
    """Changes the order and name of DataFrame columns to the project's needs
        for readability.

    Args:
        df: The pandas DataFrame.
        order_list: List object that contains the proper order of the default
             column names.
        ref_list: List object that contains the renaming list based
            on the project needs.

    Returns:
        A DataFrame with the new columns order and names.
    """
    print("Initial columns:", list(df.columns))

    if are_lists_equal(list(df.columns), ref_list):
        pass

    else:
        if len(target_list) == len(ref_list): 
            # keep and re-order only the necessary columns of the initial DataFrame
            df = df[target_list]
            rename_dict = dict(zip(target_list, ref_list))
            df = df.rename(columns=rename_dict)  # rename the columns
        else:
            print("The length of the target list and the reference list is not equal.")

    print("Processed columns:", list(df.columns))

    return df


def rename_df_column_values(
    np_array: np.ndarray, 
    y: list, 
    columns_names: tuple = ("acc_x", "acc_y", "acc_z")
):
    """Creates a DataFrame with a "y" label column and replaces the values of the y with the index
    of the unique values of y.

    Args:
        np_array: 2D NumPy array.
        y: List with the y labels
        columns_names: List with the DF columns names.

    Returns:
        DataFrame with the multi-axes values and the target labels column.
    """
    arr_y = np.array(y)  # list to numpy array
    unique_values_list = np.unique(arr_y)  # unique list of values

    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y

    # replace the row item value in the y column of the df, with its index in the unique list
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])

    return df


def are_lists_equal(
    list1: list, 
    list2: list
) -> bool:
    return set(list1) == set(list2)


def encode_labels(instances_list) -> np.ndarray:
    """Encodes target labels.

    Args:
        instances_list: List of instances to be encoded.

    Returns:
        The encoded array.
    """
    le = preprocessing.LabelEncoder()
    le.fit(instances_list)
    instances_arr = le.transform(instances_list)

    return instances_arr


def list_files_in_folder(folder_path) -> list:
    """Returns a list of all CSV files within the specified folder.

    Args:
        folder_path (str): The directory path to search for files.

    Returns:
        list: A list containing the filenames (strings) of all files
              in the directory that end with the '.csv' extension.
    """
    files_list = list()
    for f in listdir(folder_path):
        if isfile(join(folder_path, f)):
            if f.endswith(".csv"):
                files_list.append(f)

    return files_list

def process_sensor_data(df, trim_size=100, z_threshold=3):
    df_trimmed = df.iloc[:-trim_size].copy()
    for col_name in df_trimmed.columns:
        mean_val = df_trimmed[col_name].mean()
        std_val = df_trimmed[col_name].std()
        z_scores = (df_trimmed[col_name] - mean_val) / std_val
        df_trimmed.loc[np.abs(z_scores) > z_threshold, col_name] = np.nan
        df_trimmed[col_name] = df_trimmed[col_name].interpolate(method='linear')
        df_trimmed[col_name] = df_trimmed[col_name].ffill().bfill()
    return df_trimmed

def extract_features(window_df):
    features = []
    
    acc_cols = [c for c in window_df.columns if 'acc' in c]
    if len(acc_cols) == 3:
        acc_vals = window_df[acc_cols].values
        vm = np.sqrt(np.sum(acc_vals**2, axis=1))
        sma = np.mean(np.sum(np.abs(acc_vals), axis=1))
        features.extend([np.mean(vm), np.std(vm), sma])
        
    for col in window_df.columns:
        values = window_df[col].values
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        median_val = np.median(values)
        range_val = max_val - min_val
        zcr_val = np.mean(np.diff(values > mean_val) != 0)
        
        features.extend([mean_val, std_val, min_val, max_val, median_val, range_val, zcr_val])
        
        fft_vals = np.abs(np.fft.rfft(values))
        fft_energy = np.sum(fft_vals**2)
        dom_freq_idx = np.argmax(fft_vals)
        
        psd = (fft_vals**2) / (fft_energy + 1e-9)
        spectral_entropy = -np.sum(psd * np.log2(psd + 1e-9))
        
        features.extend([fft_energy, dom_freq_idx, spectral_entropy])
        
    return features

def evaluate_models(X_tr, y_tr, X_te, y_te, le):
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier()
    }
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        print(f"=================== {name} Classification Report ===================")
        print(classification_report(y_te, y_pred, target_names=le.classes_, digits=4))
        print("\n")
