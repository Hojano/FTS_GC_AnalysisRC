import pandas as pd
from scipy import integrate
import os
from typing import Literal

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

FIDList=[]
def chromatogram(file_list, file_type:str=Literal['FID', 'AuxLeft', 'AuxRight'], fid_reference_list=FIDList, output_path=experiment_path, output_name:str=Literal['FID_total1.csv', 'AuxLeft_total1.csv', 'AuxRight_total1.csv']):
    """
    Processes a group of chromatogram files (e.g., FID, AuxLeft, AuxRight), aligns them by minutes from FID start time,
    and optionally saves to CSV.

    Parameters:
        file_list (list of str): List of chromatogram file paths to process.
        file_type (str): The file type identifier in filenames (e.g., 'FID', 'AuxLeft').
        fid_reference_list (list of str): List of FID filenames to establish start time.
        output_path (str): Directory to save output CSV.
        output_name (str): Name of the CSV file to write.

    Returns:
        pd.DataFrame: Combined chromatogram DataFrame indexed by time.
    """
    # Step 1: Extract datetime from each FID file to establish a reference (earliest) start time
    start_times = []
    for fid in fid_reference_list:
        fid_time_str = fid.split('FID_')[-1].split('.txt')[0]
        fid_datetime = pd.to_datetime(fid_time_str, format='%d-%b-%Y %H_%M', errors='coerce')
        start_times.append(fid_datetime)
    datetime_start = min(start_times) # Use earliest time as reference point


    chromatogram_dict = {}
    # Step 2: Process each chromatogram file in the list
    for file_path in file_list:
        # Read the data: skip metadata rows and load columns as Time, Step, and Value
        df = pd.read_csv(file_path, names=['Time', 'Step', 'Value'], sep='\t', skiprows=43)
        df = df.replace(',', '', regex=True) # Remove commas if present

        filename = file_path.split('\\')[-1] # Extract file name from full path
        if file_type not in filename:
            raise ValueError(f"Expected '{file_type}' in filename but got: {filename}")

        # Extract timestamp portion from filename
        time_raw = filename.split('.txt')[0].split(file_type)[-1]
        parts = time_raw.split('_')
        if len(parts) < 3:
            raise ValueError(f"Filename format not recognized for timestamp extraction in: {filename}")
        time_str = parts[1] + '_' + parts[2] # Build timestamp string like '27-Feb-2025_14_30'
        
        # Convert extracted string into datetime object
        file_datetime = pd.to_datetime(time_str, format='%d-%b-%Y %H_%M', errors='coerce')
        delta_minutes = round((file_datetime - datetime_start).total_seconds() / 60)
        
        # Add to dict using minutes-from-start as column key
        chromatogram_dict[delta_minutes] = df['Value']
        chromatogram_dict['Time'] = df['Time']

    # Step 3: Convert the dictionary to a DataFrame
    df_combined = pd.DataFrame.from_dict(chromatogram_dict)
    df_combined.index = df['Time']
    df_combined = df_combined.drop(columns='Time')
    
    # Step 4: Optionally save the combined DataFrame to a CSV file
    if output_path and output_name:
        df_combined.to_csv(f"{output_path}/{output_name}", index=True)

    return df_combined

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
