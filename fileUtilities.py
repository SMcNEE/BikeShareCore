from os import listdir
from time import perf_counter
from tqdm import tqdm
import pandas as pd
import zipfile
from stations import Stations, create_stations_from_json
import io
import numpy as np
from multiprocessing import Pool

# Dictionary with the new col headings from the raw data
col_translation = {'Trip Id': 'trip_id',
                    'Subscription Id': 'subscription_id',
                    'Trip  Duration': 'trip_duration_seconds', 
                    'Start Station Id': 'from_station_id',
                    'Start Time': 'trip_start_time', 
                    'Start Station Name': 'from_station_name', 
                    'End Time': 'trip_stop_time',
                    'End Station Id': 'to_station_id', 
                    'End Station Name': 'to_station_name',
                    'Bike Id': 'bike_id', 
                    'User Type': 'user_type'}

# Function which returns a list of the files with a given file extention
# in a directory. By default return .csv files
def getCsvFilenames(directory, suffix=".csv"):
    filenames = listdir(directory)
    listOfFilenames = [name for name in filenames if name.endswith( suffix )]
    return listOfFilenames

# Read the raw data from CSV file into a pandas dataframe
# colums are renamed and start and end times are converted from string to datetime
def readRawTripData(filename):
    df = pd.read_csv(filename)
    df = df.rename(columns = col_translation)
    
    df["trip_start_time"] = pd.to_datetime(df["trip_start_time"], errors="coerce")
    df["trip_stop_time"] = pd.to_datetime(df["trip_stop_time"], errors="coerce")
    return df

def ingestRawTripData(rawDataDirectory):
    startTime = perf_counter()
    print(f"Reading data in: {rawDataDirectory}")
    data = []
    csvFiles = getCsvFilenames(rawDataDirectory)
    combinedDataframe = pd.DataFrame()
    could_not_ingest = []
    if csvFiles is not None:
        for f in tqdm(csvFiles):
            filepath = f"{rawDataDirectory}/{f}"
            try:
                newData = readRawTripData(filepath)
                data.append(newData)
            except:
                could_not_ingest.append(filepath)
        combinedDataframe = pd.concat(data)
        combinedDataframe = combinedDataframe.reset_index()
    endTime = perf_counter()
    for f in could_not_ingest:
        print(f"Could not ingest {f}")
    print(f"Data import complete in {endTime-startTime}s")
    return combinedDataframe

def multi_ingest_raw_trip_data(raw_data_dir: str) -> pd.DataFrame:
    start = perf_counter()
    print(f"Reading data in: {raw_data_dir}")
    csv_files = getCsvFilenames(raw_data_dir)
    csv_files_w_dir = []
    for f in csv_files:
        csv_files_w_dir.append(f'{raw_data_dir}/{f}')
    # Return an empty dataframe if no records found
    if csv_files_w_dir is None:
        return pd.DataFrame()
    with Pool() as p:
        data = p.map(readRawTripData, csv_files_w_dir)
    combined_df = pd.concat(data, ignore_index=True)
    cols = [
        'trip_duration_seconds',
        'from_station_id',
        'trip_start_time',
        'from_station_name',
        'trip_stop_time',
        'to_station_id',
        'to_station_name'
    ]
    combined_df = combined_df[cols]
    col_types = {
        'trip_duration_seconds': 'float64',
        'from_station_id': 'float64',
        'from_station_name': 'category',
        'to_station_id': 'float64',
        'to_station_name': 'category'
    }
    combined_df = combined_df.astype(col_types)
    print(f'Import complete in {perf_counter() - start}s')
    return combined_df

def saveDataToCsv(data, outputFilename):
    data.to_csv(outputFilename, index=False, index_label=False)

def load_csv_version(input_file):
    # This reads the data back from the create CSV (much faster for subsequent use)
    all_bike_data = pd.read_csv(input_file, parse_dates=[5,8], infer_datetime_format=True)
    return all_bike_data

def ingestRawTripDataAndSaveToCsv(rawDataDirectory, outputFile, preIngested=None):
    start_time = perf_counter()
    data = preIngested
    if data is None:
        print(f"Ingesting data from {rawDataDirectory}")
        data = ingestRawTripData(rawDataDirectory)
    data.to_csv(outputFile, index=False, index_label=False)
    end_time = perf_counter()
    print(f"Saved to {outputFile} in {end_time-start_time}s")

def ingestRawTripAndSaveToFeather(rawDataDirectory, outputFile):
    data = ingestRawTripData(rawDataDirectory)
    data.to_feather(outputFile)
    print("Save complete")

def ingestRawTripDataAndSaveToHDF5(rawDataDirectory, outputFile, preIngested=None):
    start_time = perf_counter()
    data = preIngested
    if data is None:
        print(f"Ingesting data from {rawDataDirectory}")
        data = ingestRawTripData(rawDataDirectory)
    data.to_hdf(outputFile, key='trips', mode='w', format='table', complevel=9, data_columns=True, nan_rep='NaN')
    end_time = perf_counter()
    print(f"Saved to {outputFile} in {end_time-start_time}s")

def getListOfStations(filename):
    df = pd.read_csv(filename)
    return df["station_id"].tolist()

def readJsonStationData(jsonFilename):
    df = pd.read_json(jsonFilename)
    return df 

def map_zip_file(zip_filepath):
    """
    This method opens a zipfile and gets the start and end point
    of the timestamped bike system snapshots
    """
    # TODO make a function to unzip folders
    filenames = []
    starts = []
    ends = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip:
        namelist = sorted(zip.namelist())
        for name in namelist:
            filenames.append(name)
            with io.TextIOWrapper(zip.open(name), encoding="utf-8") as f:
                stns, timestamps = create_stations_from_json(f)
                starts.append(timestamps[0])
                ends.append(timestamps[-1])
    repeated_zipfilenames = [zip_filepath for f in filenames]
    output = pd.DataFrame({'zipfile': repeated_zipfilenames, 
                           'filename': filenames, 'start': starts, 
                           'end': ends})
    return output

def map_zip_files_in_dir(dir_path):
    list_of_zipfiles = getCsvFilenames(dir_path, suffix=".zip")
    output_list = []
    for z in list_of_zipfiles:
        output_list.append(map_zip_file(f"{dir_path}/{z}"))
    output = pd.concat(output_list)
    return output

def station_snapshot_to_df(file):
    output = None
    timestamps_series = None
    raw_data = file.read()
    list_of_snapshots = Stations.split_station_snapshots(raw_data, [])
    list_of_timestamps = []
    list_of_dfs = []
    for snap in list_of_snapshots:
        try:
            timestamp, data = Stations.parse_single_snapshot(snap)
            df = pd.read_json(data, orient="records")
            list_of_timestamps.append(int(timestamp))
            list_of_dfs.append(df)
        except:
            print(f'Failed to parse snapshot at {timestamp} in {file.name}')
    # Check that the outputs aren't empty before making the output
    if list_of_dfs and list_of_timestamps:
        output = pd.concat(list_of_dfs, 
                        keys=list_of_timestamps, 
                        names=['timestamp',None])
        timestamps_series = pd.Series(list_of_timestamps)
    return output, timestamps_series

def zipped_snapshots_to_df(zip_filepath):
    all_dfs = []
    all_timestamps = []
    with zipfile.ZipFile(zip_filepath, 'r') as zip:
        namelist = sorted(zip.namelist())
        exclude1 = 'Bike Share Data/Bike Share Data Pull'
        exclude2 = 'Bike Share Data/bike_share_data_working.txt'
        for name in namelist:
            if name != exclude1 and name != exclude2:
                with io.TextIOWrapper(zip.open(name), encoding='utf-8') as f:
                    df, timestamps = station_snapshot_to_df(f)
                    # Check that the values aren't empty
                    if df is not None and timestamps is not None:
                        all_dfs.append(df)
                        all_timestamps.append(timestamps)
    combined_dfs = pd.concat(all_dfs)
    combined_timestamps = pd.concat(all_timestamps, ignore_index=True)
    return combined_dfs, combined_timestamps

def zipped_snapshots_dir_to_hdf(input_dir, output_filepath):
    zip_filenames = getCsvFilenames(input_dir, suffix='.zip')
    m = 'w' # to make a new file in the first instance
    for zip_filename in tqdm(zip_filenames):
        df, timestamps = zipped_snapshots_to_df(input_dir + zip_filename)
        # del df['num_bikes_available_types']
        df = df.drop(columns=['num_bikes_available_types', 'traffic'],
                     errors='ignore')
        try:
            df.to_hdf(output_filepath, 
                    key='snapshots', 
                    mode=m, 
                    format='table', 
                    complevel=9, 
                    append=True)
            m = 'r+' # Append to the new file afterwords
            timestamps.to_hdf(output_filepath,
                            key='timestamps',
                            mode=m,
                            format='table',
                            complevel=9,
                            append=True)
        except:
            print(f'Issue saving {zip_filename} to hdf')

def make_super_tt_matrix(trips_hdf_file, tt_csv_file, tt_if_none, output_filepath):
    """
    This method will make a new travel time matrix that will have all possible
    combinations of to/from stations based on the full trip list.
    0 is given for to/from the same station
    tt_if_none is used for values not in the given travel time matrix
    """
    all_trips = pd.read_hdf(trips_hdf_file, key='trips')
    truck_tt = pd.read_csv(tt_csv_file, na_values='#VALUE!')
    unique_start_stns = all_trips['from_station_id'].unique()
    unique_end_stns = all_trips['to_station_id'].unique()
    combo = np.concatenate((unique_start_stns, unique_end_stns))
    unique_stns = np.unique(combo)
    unique_stns = unique_stns[~np.isnan(unique_stns)]
    all_pairs = np.array([[i, j] for j in unique_stns for i in unique_stns], int)
    def get_travel_time(i, j, tt_df, tt_if_none_found):
        tt = tt_if_none_found
        if i == j:
            tt = 0
        else:
            try:
                tt = tt_df.loc[(tt_df['Start Station Id'] == i) & (tt_df['End Station Id'] == j)].iloc[0]['travelTime']
            except:
                tt = tt_if_none_found
        return tt

    super_truck_tt = pd.DataFrame(
        all_pairs, 
        index=[x for x in range(len(all_pairs))],
        columns=['Start Station Id', 'End Station Id'])
    super_truck_tt['travelTime'] = super_truck_tt.apply(lambda x: get_travel_time(x['Start Station Id'], x['End Station Id'], truck_tt, 120), axis=1)
    super_truck_tt['travelTime'] = super_truck_tt['travelTime'].fillna(tt_if_none)
    super_truck_tt.to_csv(output_filepath, index=False)


def main():
    # Combine the trip data into single HDF File
    rawTripDataDir = "data/Bike Share Trips"
    combinedCsvFilename = "data/Combined Bike Trips/combined_bike_trip_data_201701_202206.csv"
    tripDataHDF = "data/Combined Bike Trips/combined_bike_trip_data_201701_202206.h5"
    data = multi_ingest_raw_trip_data(rawTripDataDir)
    ingestRawTripDataAndSaveToCsv(rawTripDataDir, combinedCsvFilename, data)
    ingestRawTripDataAndSaveToHDF5(rawTripDataDir, tripDataHDF, data)
    print('Complete')

if __name__ == '__main__':
    main()