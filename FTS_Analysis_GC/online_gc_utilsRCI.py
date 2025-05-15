import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy import integrate

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

def integrate_peak1(DF,StartTime,EndTime):
    minutes = DF.index
    StartIndex = min(range(len(minutes)), key=lambda i: abs(minutes[i]-StartTime))
    EndIndex = min(range(len(minutes)), key=lambda i: abs(minutes[i] - EndTime))
    DF = DF.iloc[StartIndex:EndIndex]
    x = []
    y = []
    for column in DF.columns:
        x.append(float(column))
        Orginal_Chromatogram = DF[column]
        x_values = DF.index
        y1 = DF[column].iloc[0]
        y2 = DF[column].iloc[-1]
        x1 = DF.index[0]
        x2 = DF.index[-1]
        slope = (y2 - y1) / (x2 - x1)
        ZeroY = y1 - (x1 * slope)
        SlopeLine = x_values * slope + ZeroY
        Subtracted_Chromatogram = Orginal_Chromatogram - SlopeLine
        MethaneIntegration = integrate.trapezoid(y=Subtracted_Chromatogram,x=x_values)
        y.append(MethaneIntegration)
    return pd.Series(data=y,index=x)


def chromatogram(filename, file_type, FIDList):
    # Compute the reference start time from FIDList.
    DateTimeList = []
    for fid in FIDList:
        # Extract timestamp portion from each FID filename.
        TempStartTimeRaw_ = fid.split('FID_')[-1]
        TempStartTimeRaw = TempStartTimeRaw_.split('.txt')[0]
        TempDateTime_Start = pd.to_datetime(TempStartTimeRaw, format='%d-%b-%Y %H_%M', errors='coerce')
        DateTimeList.append(TempDateTime_Start)
    DateTime_Start = min(DateTimeList)
    #print("DateTime_Start:", DateTime_Start)
    
    # Read the chromatogram data from the file, skipping the first 43 rows.
    DF = pd.read_csv(filename, names=['Time', 'Step', 'Value'], sep='\t', skiprows=43)
    DF = DF.replace(',', '', regex=True)
    # Extract the file name from the full path.
    Name = filename.split('\\')[-1]
    # Remove the file extension.
    TimeRaw_ = Name.split('.txt')[0]
    # Check that the expected file_type is in the filename.
    if file_type not in Name:
        raise ValueError(f"Expected '{file_type}' in filename but got: {Name}")
    # Isolate the timestamp portion by splitting using file_type.
    TimeRaw_ = TimeRaw_.split(file_type)[-1]
    parts = TimeRaw_.split('_')
    if len(parts) < 3:
        raise ValueError("Filename format not recognized for timestamp extraction.")
    TimeRaw = parts[1] + '_' + parts[2]  # Expect the timestamp to be the second and third parts.

    DateTime = pd.to_datetime(TimeRaw, format='%d-%b-%Y %H_%M')
    DateTimeFromStart = DateTime - DateTime_Start
    MinutesFromStart = round(DateTimeFromStart.total_seconds() / 60)
    return {
        'DF': DF,
        'Name': Name,
        'TimeRaw': TimeRaw,
        'DateTime': DateTime,
        'DateTimeFromStart': DateTimeFromStart,
        'MinutesFromStart': MinutesFromStart
    }



def baseline_correct(frame):
    #for each channel, substract the minimum value
    for i in range(0,3):
        #substract the mean of the values between 288 and 294 seconds
        frame.iloc[:,0] = frame.iloc[:,0] - frame.iloc[288*50:294*50,0].mean()
        frame.iloc[:,1] = frame.iloc[:,1] - frame.iloc[288*50:294*50,1].mean()
        frame.iloc[:,2] = frame.iloc[:,2] - frame.iloc[10*50:15*50,2].mean()
        # frame.iloc[:,i] = frame.iloc[:,i] - frame.iloc[:,i].min()
    return frame

