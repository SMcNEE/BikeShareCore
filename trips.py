import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# trip_list should be a list of tuples with the from and to of the trip
def trip_list_to_array(trip_list, trip_lengths, stations):
    # find the longest trip length
    max_len = np.amax(trip_lengths)
    num_stns = len(stations.station_ids)
    arrivals = np.zeros((num_stns, max_len+1))
    # from, to
    departures = np.zeros_like(stations.current_occupancy)
    keys = stations.station_keys
    for trip, length in zip(trip_list, trip_lengths):
        departures[keys[trip[0]]] -= 1 # remove one from the from station
        # add one to the arrival position
        arrivals[keys[trip[1]], length] += 1 # add one to the to station
    return departures, arrivals

def pd_trip_list_to_array(pd_trip_list, stations):
    df = pd_trip_list
    max_len = df["rounded_duration"].max()
    num_stns = len(stations.station_ids)
    arrivals = np.zeros((num_stns, max_len+1))
    # from, to
    departures = np.zeros_like(stations.current_occupancy)
    keys = stations.station_keys
    impossible_trips = []
    for start, end, length in zip(df["from_station_id"], 
                                  df["to_station_id"], 
                                  df["rounded_duration"]):
        check = False
        # Try to find the index for the start and end stations
        try:
            index_start = keys[start]
            index_end = keys[end]
            check = True
        # If cannot find one of the stations, print error
        except:
            impossible_trips.append((start,end))
            # print(f'Could not complete the trip from {start} to {end}')
        # Only add the trip to the matrix if both stations were found
        if check:
            departures[index_start] -= 1
            arrivals[index_end, length] += 1
    # print(f'Failed to complete {len(impossible_trips)} trips')
    return departures, arrivals, impossible_trips

def timestep_from(reference_time, input_time):
    timestep = 0
    delta = input_time - reference_time
    timestep = delta.total_seconds() // 60.0 # int division to round down
    return timestep

def read_df_trip_list(df_trip_list, reference_time, max_length):
    df = df_trip_list.copy()
    df["timestep"] = df.apply(lambda row: timestep_from(reference_time, row["trip_start_time"]), axis=1)
    df["rounded_duration"] = df.apply(lambda row: round(row["trip_duration_seconds"] / 60.0), axis=1)
    df = df.loc[df['rounded_duration'] < max_length]
    return df

def read_csv_trip_list(filename, reference_time, max_length):
    """
    This method reads a trip list from csv
    """
    df = pd.read_csv(filename, parse_dates=["trip_start_time", "trip_stop_time"])
    output = read_df_trip_list(df, reference_time, max_length)
    return output

def get_trips_for_timestep(trip_list, timestep):
    return trip_list.loc[trip_list["timestep"] == timestep]

def main():
    print("Stating Test Run")

if __name__ == '__main__':
    main()