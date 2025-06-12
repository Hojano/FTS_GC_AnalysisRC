import pandas as pd
from scipy import integrate
import os
from typing import Literal
import matplotlib.pyplot as plt

experiment_path=None
def collect_chromatogram_files(experiment_path):
    """
    Collects file lists for FID, TCD_AuxLeft, and TCD_AuxRight chromatograms and loads the first file from each list.
    Args:
        experiment_path (str): Path to the experiment directory.
    Returns:
        tuple: (FIDList, AuxLeftList, AuxRightList, FID_0, AuxLeft_0, AuxRight_0)
    """
    # Collect chromatogram files
    DataDict = os.path.join(experiment_path, 'chromatograms')
    FIDList = []
    AuxRightList = []
    AuxLeftList = []
    for root, dirs, files in os.walk(DataDict, topdown=True):
        for name in files:
            if 'FID_' in name:
                FIDList.append(os.path.join(root, name))
            if 'TCD_AuxLeft' in name:
                AuxLeftList.append(os.path.join(root, name))
            if 'TCD_AuxRight' in name:
                AuxRightList.append(os.path.join(root, name))
    FID_0 = pd.read_csv(FIDList[0], names=['Time', 'Step', 'Value'], sep='\t', skiprows=43) if FIDList else None
    AuxLeft_0 = pd.read_csv(AuxLeftList[0], names=['Time', 'Step', 'Value'], sep='\t', skiprows=43) if AuxLeftList else None
    AuxRight_0 = pd.read_csv(AuxRightList[0], names=['Time', 'Step', 'Value'], sep='\t', skiprows=43) if AuxRightList else None
    return FIDList, AuxLeftList, AuxRightList, FID_0, AuxLeft_0, AuxRight_0

def read_logfile(experiment_path, gases_to_plot=None, datetime_start=None, plot_against='TOS'):
    """
    Reads and processes reactor logfile(s) from experiment_path.

    Parameters:
        experiment_path (str): Path to the experiment folder containing logfiles.
        gases_to_plot (list of str): Gases to plot (e.g. ['CO', 'H2']). Default: all common.
        datetime_start (datetime or str): Reference start time for TOS calculation.
        plot_against (str): 'TOS' (default) or 'datetime' for x-axis reference.

    Returns:
        pd.DataFrame: Processed logfile with DateTime index and 'TOS' column (in minutes).
    """
    # Step 1: Collect .txt logfiles
    logfile_files = sorted([f for f in os.listdir(experiment_path) if f.endswith('.txt')])
    if not logfile_files:
        raise ValueError('No logfile found in the specified path.')
    
    # Step 2: Read header from first logfile
    df1 = pd.read_csv(os.path.join(experiment_path, logfile_files[0]), header=None, sep='\t', skiprows=1, nrows=1)
    header_row = df1.iloc[0].tolist()

    # Step 3: Read and combine logfiles
    if len(logfile_files) > 1:
        print('Multiple logfiles found! Combining them...')
        logfile = pd.concat([
            pd.read_csv(os.path.join(experiment_path, f), sep='\t', skiprows=2, names=header_row)
            for f in logfile_files
        ])
    else:
        logfile = pd.read_csv(os.path.join(experiment_path, logfile_files[0]), sep='\t', skiprows=2, names=header_row)
    
    # Step 4: Parse datetime and filter for Valve 9 ON (reactor line)
    logfile.index = pd.to_datetime(logfile['Date/Time'], format='%d-%b-%Y %H:%M:%S', errors='coerce')
    logfile = logfile.drop(columns='Date/Time')
    logfile = logfile[logfile['Valve 9'] == 1]

    # Step 5: Select relevant columns
    mfc_columns = [col for col in logfile.columns if 'MFC' in col and 'pv' in col]
    static_columns = ['Valve 9', 'Oven PV', 'Pressure R1', 'Pressure R2', 'BPC A-SP', 'MFM']
    selected_columns = static_columns + mfc_columns
    logfile = logfile[selected_columns]
    logfile['Total Flow'] = logfile[mfc_columns].sum(axis=1)  # Calculate total flow

    # Handle datetime_start and compute TOS
    if datetime_start is None:
        datetime_start = logfile.index[0]  # Default to first on-stream timestamp
        print(f"datetime_start not provided, using: {datetime_start}")
    else:
        datetime_start = pd.to_datetime(datetime_start)

    logfile['TOS'] = (logfile.index - datetime_start).total_seconds() / 60  # minutes

    # Choose x-axis for plotting
    x_axis = logfile['TOS'] if plot_against.lower() == 'tos' else logfile.index

    gas_map = { # Step 7: Define which gas flows to plot
        'CO': 'MFC CO pv',
        'H2': 'MFC H2 pv',
        'Ar': 'MFC Ar pv',
        'N2': 'MFC N2 pv',
        'CO2': 'MFC CO2 pv',
        'O2': 'MFC O2 pv',
        'He': 'MFC He pv'
    }
    if gases_to_plot is None:
        gases_to_plot = list(gas_map.keys())
    gas_columns = [gas_map[g] for g in gases_to_plot if gas_map.get(g) in logfile.columns]
    
    return logfile, x_axis, gas_columns, plot_against

FIDList=[]
def chromatogram(file_list, file_type:str=Literal['FID', 'AuxLeft', 'AuxRight'], fid_reference_list=FIDList, output_path=experiment_path, output_name:str=Literal['FID_total1.csv', 'AuxLeft_total1.csv', 'AuxRight_total1.csv']):    
    """
    Processes chromatogram files, aligns them by minutes from FID start time,
    and optionally saves to CSV unless the output already exists.

    Parameters:
        file_list (list of str): List of chromatogram file paths to process.
        file_type (str): The file type identifier in filenames (e.g., 'FID', 'AuxLeft').
        fid_reference_list (list of str): List of FID filenames to establish start time.
        output_path (str): Directory to save output CSV.
        output_name (str): Name of the CSV file to write.

    Returns:
        pd.DataFrame: Combined chromatogram DataFrame indexed by time (if computed), otherwise None.
    """
    output_file = os.path.join(output_path, output_name)

    #Check if output file already exists
    if os.path.isfile(output_file):
        print(f"[INFO] Data is already loaded: {output_name} exists in {output_path}.")
        return None

    #Step 1: Determine the earliest start time from FID list
    start_times = []
    for fid in fid_reference_list:
        fid_time_str = fid.split('FID_')[-1].split('.txt')[0]
        fid_datetime = pd.to_datetime(fid_time_str, format='%d-%b-%Y %H_%M', errors='coerce')
        start_times.append(fid_datetime)
    datetime_start = min(start_times) # Use earliest time as reference point

    chromatogram_dict = {}

    #Step 2: Process each file in the chromatogram file list
    for file_path in file_list: #Read the data: skip metadata rows and load columns as Time, Step, and Value
        df = pd.read_csv(file_path, names=['Time', 'Step', 'Value'], sep='\t', skiprows=43)
        df = df.replace(',', '', regex=True) # Remove commas if present

        filename = os.path.basename(file_path) # Extract file name from full path
        if file_type not in filename:
            raise ValueError(f"Expected '{file_type}' in filename but got: {filename}")

        #Extract timestamp portion from filename
        time_raw = filename.split('.txt')[0].split(file_type)[-1]
        parts = time_raw.split('_')
        if len(parts) < 3:
            raise ValueError(f"Filename format not recognized for timestamp extraction in: {filename}")
        time_str = parts[1] + '_' + parts[2] #Build timestamp string like '27-Feb-2025_14_30'

        # Convert extracted string into datetime object
        file_datetime = pd.to_datetime(time_str, format='%d-%b-%Y %H_%M', errors='coerce')
        delta_minutes = round((file_datetime - datetime_start).total_seconds() / 60)

        # Add to dict using minutes-from-start as column key
        chromatogram_dict[delta_minutes] = df['Value']
        chromatogram_dict['Time'] = df['Time']

    # Step 3: Construct final DataFrame
    df_combined = pd.DataFrame.from_dict(chromatogram_dict)
    df_combined.index = df['Time']
    df_combined = df_combined.drop(columns='Time')

    # Step 4: Save to CSV
    df_combined.to_csv(output_file, index=True)
    print(f"[INFO] Chromatogram saved to: {output_file}")

    return df_combined, datetime_start


def baseline_correct_column(col, time_index, start, end):
    # Select the baseline window values for this column based on the provided time index.
    baseline_values = col[ (time_index >= start) & (time_index <= end) ]
    # Compute the mean over the baseline period.
    baseline = baseline_values.mean()
    # Return the column with the baseline subtracted.
    return col - baseline 

def integrate_named_peaks(DF, named_peak_windows):
    """
    Integrate multiple named peaks from a chromatogram DataFrame.

    Parameters:
    - DF: DataFrame with chromatograms (columns = time points, index = retention time)
    - named_peak_windows: List of [peak_name, [StartTime, EndTime], UseTwoPointBaseLine]

    Returns:
    - DataFrame with integrated areas:
        rows = time points
        columns = peak names (e.g., C1, C2, C3)
    """
    result = {}

    for name, (StartTime, EndTime), UseTwoPointBaseLine in named_peak_windows:
        # Find nearest indices to the time window
        minutes = DF.index
        StartIndex = min(range(len(minutes)), key=lambda i: abs(minutes[i] - StartTime))
        EndIndex = min(range(len(minutes)), key=lambda i: abs(minutes[i] - EndTime))
        PeakDF = DF.iloc[StartIndex:EndIndex]

        peak_areas = []
        time_points = []

        for column in PeakDF.columns:
            time_points.append(float(column))

            if UseTwoPointBaseLine:
                Orginal_Chromatogram = PeakDF[column]
                x_values = PeakDF.index
                y1 = Orginal_Chromatogram.iloc[0]
                y2 = Orginal_Chromatogram.iloc[-1]
                x1 = x_values[0]
                x2 = x_values[-1]
                slope = (y2 - y1) / (x2 - x1)
                ZeroY = y1 - (x1 * slope)
                SlopeLine = x_values * slope + ZeroY
                Subtracted_Chromatogram = Orginal_Chromatogram - SlopeLine
                area = integrate.trapezoid(y=Subtracted_Chromatogram, x=x_values)
            else:
                area = integrate.trapezoid(y=PeakDF[column], x=PeakDF.index)

            peak_areas.append(area)

        result[name] = pd.Series(peak_areas, index=time_points)

    # Combine all peak area series into a DataFrame
    result_df = pd.DataFrame(result)
    result_df.index.name = "Time_Point"
    return result_df
