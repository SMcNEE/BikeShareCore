import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from trips import pd_trip_list_to_array, get_trips_for_timestep
from stations import create_stations_from_zip, Stations
import zipfile
import io

class World:
    def __init__(self, timestamp, stations, od_matrix_file = None, walk_od_matrix_file = None):
        self.reference_time = datetime.fromtimestamp(timestamp)
        self.current_time = datetime.fromtimestamp(timestamp)
        self.stations = stations
        self.expected_arrivals = np.zeros((len(stations.station_ids),120))
        self.bike_od_matrix = None
        self.walk_od_matrix = None
        if od_matrix_file is not None:
            self.bike_od_matrix = pd.read_csv(od_matrix_file)
        if walk_od_matrix_file is not None:
            self.walk_od_matrix = pd.read_csv(walk_od_matrix_file)
        # self.given_trips = 0
        self.completed_trips = 0
        self.trips_delayed_by_full = 0
        self.trips_delayed_by_empty = 0
        self.time_from_full_delay = 0
        self.time_from_empty_delay = 0
        self.empty_failed_to_reroute = 0
        self.full_delays = []
        self.empty_delays = []

    def change_reference_time(self, timestamp):
        self.reference_time = datetime.fromtimestamp(timestamp)

    def change_time(self, new_timestamp):
        self.current_time = datetime.fromtimestamp(new_timestamp)
        self.stations.current_time = self.current_time

    def increment_time(self, days, hours, minutes, seconds=0, milliseconds=0, microseconds=0):
        delta = timedelta(days, seconds, microseconds, milliseconds, minutes, hours)
        self.current_time = self.current_time + delta
        self.stations.current_time = self.current_time

    def set_stations(self, new_stations):
        self.stations = new_stations
    
    def reset_results_counters(self):
        # self.given_trips = 0
        self.completed_trips = 0
        self.trips_delayed_by_full = 0
        self.trips_delayed_by_empty = 0
        self.time_from_full_delay = 0
        self.time_from_empty_delay = 0
        self.empty_failed_to_reroute = 0

    def num_bikes_in_world(self) -> float:
        bikes_in_stns = np.sum(self.stations.current_occupancy)
        bikes_in_world = np.sum(self.expected_arrivals)
        num_bikes = bikes_in_stns + bikes_in_world
        return num_bikes

    def apply_trips(self, departures, arrivals):
        """
        This method updates the expected arrivals array with a given trip list
        Arrivals are positive and departures are negative
        """
        self.expected_arrivals[:,0] = self.expected_arrivals[:,0] + departures
        arrivals_timesteps = arrivals.shape[1]
        self.expected_arrivals[:,:arrivals_timesteps] = self.expected_arrivals[:,:arrivals_timesteps] + arrivals

    def increment_expected_arrivals(self, timesteps, enforce_capacity=True):
        """
        This method increments the array tracking the expected arrivals in the world
        timesteps is the number of time increments (e.g. minutes being incremented)
        The arrivals (positive and negative) within the timesteps are applied to the stations
        and the expected arrivals array is re-baselined to the new time
        """
        self.increment_time(0,0,timesteps)
        trips_to_apply = self.expected_arrivals[:,:timesteps]
        old_arrivals_array = self.expected_arrivals
        self.expected_arrivals = np.zeros_like(old_arrivals_array)
        self.expected_arrivals[:,:-1*timesteps] = old_arrivals_array[:,timesteps:]
        cap = self.stations.increment_occupancy(np.sum(trips_to_apply, axis=1), 
                                                enforce_capacity)
        return cap

    def reroute_trips(self, cap, trips):
        empty_stn_ids = cap[0]
        under_amounts = cap[1]
        full_stn_ids = cap[2]
        over_amounts = cap[3]
        bike_od = self.bike_od_matrix
        walk_od = self.walk_od_matrix
        stations = self.stations
        keys = stations.station_keys
        # reroute trips to a full station
        # Get a subset of the bike od matrix with only acceptable new
        # destinations (i.e. below capacity)
        with_cap = stations.current_occupancy < stations.station_capacities
        stn_ids_with_cap = stations.station_ids[with_cap]
        valid_od = bike_od.loc[bike_od['End Station Id'].isin(stn_ids_with_cap)]
        for stn, amount in zip(full_stn_ids, over_amounts):
            # find the nearest station by bike
            od = valid_od.loc[valid_od['Start Station Id'] == stn]
            od = od.loc[od['travelTime'] == od['travelTime'].min()]
            if not od.empty:
                destination = od.iloc[0]['End Station Id']
                travel_time = od.iloc[0]['travelTime']
                # get the row in the expected arrivals matrix to use
                i = keys[destination]
                j = int(np.ceil(travel_time))
                # update the expected arrivals matrix with the rerouted trip
                self.expected_arrivals[i,j] += amount
                # add the number of trips and extra travel time to the results
                self.trips_delayed_by_full += amount
                self.time_from_full_delay += amount * travel_time
                for x in range(int(amount)):
                    (self.full_delays).append(travel_time)
            else:
                self.empty_failed_to_reroute += amount
                self.completed_trips -= amount
        # reroute trips from an empty station
        # Get a subset of the walk od matrix with only non empty destinations
        non_empty = stations.current_occupancy > 0
        stn_ids_with_bikes = stations.station_ids[non_empty]
        valid_od = walk_od.loc[walk_od['End Station Id'].isin(stn_ids_with_bikes)]
        for stn, amount in zip(empty_stn_ids, under_amounts):
            # Find the nearest station by walking
            od = valid_od.loc[valid_od['Start Station Id'] == stn]
            od = od.loc[od['travelTime'] == od['travelTime'].min()]
            if not od.empty:
                new_trip_start_id = od.iloc[0]['End Station Id']
                new_trip_start_time = od.iloc[0]['travelTime']
                # get the row in the expected arrivals matrix to use
                i = keys[new_trip_start_id]
                j = int(np.ceil(new_trip_start_time))
                # update the expected arrivals matrix with the new departure
                self.expected_arrivals[i,j] -= amount
                # Get the travel time by bike from the new start to the trip end
                rerouted_trips = trips.loc[trips['from_station_id'] == stn]
                num_possible_trips = rerouted_trips.shape[0]
                num_trips_to_reroute = int(np.around(amount))
                if num_possible_trips >= num_trips_to_reroute:
                    for t in range(num_trips_to_reroute):
                        destination = rerouted_trips.iloc[t]['to_station_id']
                        try:
                            new_trave_time = bike_od.loc[(bike_od['Start Station Id'] == new_trip_start_id) & 
                                                        (bike_od['End Station Id'] == destination)]
                            tt = new_trave_time.iloc[0]['travelTime']
                            # get the row and col for the expected arrivals matrix
                            i = keys[destination]
                            j_dest = j + int(np.ceil(tt)) # j is the start point of the trip
                            self.expected_arrivals[i,j_dest] += 1
                            # add 1 to the rerouted empty trips counter
                            self.trips_delayed_by_empty += 1
                            # compute the delay caused by the reroute
                            empty_delay = j_dest - rerouted_trips.iloc[t]['rounded_duration']
                            # only count as a delay if the resulting travel time is greater
                            if empty_delay < 0:
                                empty_delay = 0
                            self.time_from_empty_delay += empty_delay
                            (self.empty_delays).append(empty_delay)
                        except:
                            # print(f'Issue finding new route for trip # {t+1} from {stn} from {new_trip_start_id} to {destination}')
                            self.empty_failed_to_reroute += 1
                            self.completed_trips -= 1
                else:
                    # print(f'Could not find {num_trips_to_reroute} trips to reroute from {stn} at {stations.current_time}')
                    self.empty_failed_to_reroute += num_trips_to_reroute
                    self.completed_trips -= num_trips_to_reroute
            else:
                # print(f'Issue rerouting {amount} trips from empty station {stn} at {stations.current_time} no valid walking destination')
                self.empty_failed_to_reroute += amount
                self.completed_trips -= amount



    def step_through_time(self, trip_list, num_steps, output_interval, enforce_capacity=True, operator=None):
        output_check = 0
        output_count = 0
        station_ids = self.stations.station_ids
        num_output_times = num_steps // output_interval + 1
        output_shape = (num_output_times, len(station_ids))
        output = np.zeros(output_shape)
        timestamps = np.zeros(num_output_times)
        failed_trips = []
        cap_details = []
        for step in range(num_steps):
            if step == output_check:
                output[output_count] = self.stations.current_occupancy
                timestamps[output_count] = datetime.timestamp(self.current_time)
                output_count += 1
                output_check += output_interval
            trips = get_trips_for_timestep(trip_list, step)
            # self.given_trips += trips.shape[0]
            if not trips.empty:
                stn = self.stations
                departures, arrivals, trip_fails = pd_trip_list_to_array(trips, 
                                                                         stn)
                # log the departures as served trips
                self.completed_trips -= departures.sum()
                self.apply_trips(departures, arrivals)
                failed_trips = failed_trips + trip_fails
            # Before incrementing the expected arrivals, check if the operator
            # made any changes
            if operator is not None:
                operator.adjust_superstations()
                operator.dispatch_manager()
            cap = self.increment_expected_arrivals(1, enforce_capacity)
            # Use the capacity results from cap to address rerouting of trips
            # from and empty station or to a full station
            if enforce_capacity is True:
                # log the capacity issues in a dataframe for results
                if len(cap[0]) > 0:
                    cap_details_empty = pd.DataFrame(
                        {
                            'time': self.current_time,
                            'full_empty': 'empty',
                            'stn_id': cap[0],
                            'amount': cap[1]
                        }, index=cap[0]
                    )
                    cap_details.append(cap_details_empty)
                if len(cap[2]) > 0:
                    cap_details_full = pd.DataFrame(
                        {
                            'time': self.current_time,
                            'full_empty': 'full',
                            'stn_id': cap[2],
                            'amount': cap[3]
                        }, index=cap[2]
                    )
                    cap_details.append(cap_details_full)
                # reroute the trips
                self.reroute_trips(cap, trips)
        # add the output if the last step should have an output
        if num_steps == output_check:
            output[output_count] = self.stations.current_occupancy
            timestamps[output_count] = datetime.timestamp(self.current_time)
        if len(cap_details) > 0:
            cap_details = pd.concat(cap_details, ignore_index=True)
        else:
            cap_details = None
        return output, station_ids, timestamps, failed_trips, cap_details
    
    def rebase_world(self, rebase_world):
        rebase_stn = rebase_world.stations
        stn = self.stations
        stn_ids = stn.station_ids
        changed = rebase_stn.match_station_ids(stn_ids)
        # Check if the station objects have matching stations
        # if changed:
        #     print('Rebase does not match original, updated to match')
        # else:
        #     print('Rebase matches original')
        self.set_stations(rebase_stn)
        self.reference_time = rebase_world.reference_time

def build_actuals_array_from_zip(station_ids, timestamps, tol, actuals_map_df):
    output = np.zeros((len(timestamps), len(station_ids)))
    start = timestamps[0]
    end = timestamps[-1]
    current_pos = 0
    max_pos = len(timestamps)
    current_timestamp = timestamps[current_pos]
    prev_delta = 100000000
    # Cut down list to only include files within the time range
    df = actuals_map_df.loc[(actuals_map_df['start'] <= end) & (actuals_map_df['end'] >= start)]
    # make sure the list is sorted by start time
    df = df.sort_values(by=['start'])
    # Counters to check how often the size of the stations are mismatched
    mismatched_station_objects = 0
    matched_station_objects = 0
    # Initialize list_of_stns and list_of_timestamps to None and trackers for prev row
    # This check avoids repeating the read if from same file
    list_of_stns = None
    list_of_timestamps = None
    prev_zipfile = None
    prev_filename = None
    for index, row in df.iterrows():
        zip_path = row['zipfile']
        filename = row['filename']
        # only read in the stations from zip if is in a new file
        # if the file is the same as previous row, use the same list_of_stns and list_of_timestamps
        if prev_zipfile != zip_path or prev_filename != filename:
            list_of_stns, list_of_timestamps = create_stations_from_zip(zip_path, filename)
        for t, stn in zip(list_of_timestamps, list_of_stns):
            # Check if the current position is complete
            if current_pos < max_pos:
                # Check that the stations objects match
                # Make the stations objects match
                changed = stn.match_station_ids(station_ids)
                if changed:
                    mismatched_station_objects += 1
                else:
                    matched_station_objects += 1
                current_timestamp = timestamps[current_pos]
                delta = abs(current_timestamp - t)
                if delta < tol:
                    # found match
                    output[current_pos] = stn.current_occupancy
                    # update the current position and timestamp
                    current_pos += 1
                    # current_timestamp = timestamps[current_pos]
                    prev_delta = 100000000
                elif prev_delta < delta:
                    # getting farther away
                    # increment to the next timestamp and set a value of NaN
                    output[current_pos] = np.nan
                    current_pos += 1
                    # current_timestamp = timestamps[current_pos]
                    prev_delta = 100000000
                else:
                    # update the previous delta value
                    prev_delta = delta
    return output

def build_actuals_array_from_hdf(station_ids, 
                                 target_timestamps, 
                                 tol, 
                                 timestamp_series, 
                                 snapshots_hdf):
    output = np.zeros((len(target_timestamps), len(station_ids)))
    # loop through each of the target timestamps
    current_pos = 0
    for t in target_timestamps:
        # Check if there is a timestamp within tolerance of the target
        # if nearby timestamp found, create stations object and add to output
        try:
            closest_t = find_closest_timestamp(t, tol, timestamp_series)
            query = f'timestamp={closest_t}'
            df = pd.read_hdf(snapshots_hdf, key='snapshots', where=query)
            stn = Stations.new_stations_from_df(t, df)
            output[current_pos] = stn.current_occupancy
        except:
            output[current_pos] = np.nan
        current_pos += 1
    return output

def find_file_containing_timestamp(timestamp, actuals_map_df):
    try:
        df = actuals_map_df.loc[(actuals_map_df['start'] <= timestamp) & (actuals_map_df['end'] >= timestamp)]
        zip_filepath = df.iloc[0]['zipfile']
        filename = df.iloc[0]['filename']
    except:
        zip_filepath = None
        filename = None
        # print(f"No file found where timestamp: {timestamp} is within range")
    return zip_filepath, filename

def world_at_timestamp(timestamp, tol, list_of_stns, list_of_timestamps, od_matrix_file = None, walk_od_matrix_file = None):
    output = None
    prev_delta = 1000000
    closest_stn = None
    closest_stn_delta = 1000000
    # Loop through the station objects to find the closest to the
    # target timestamp
    for t, stn in zip(list_of_timestamps, list_of_stns):
        delta = abs(timestamp - t)
        if delta < closest_stn_delta:
            closest_stn = stn
            closest_stn_delta = delta
    # Return the World object if have a time delta within tolerance
    # otherwise return none
    if closest_stn_delta <= tol:
        output = World(timestamp, closest_stn, od_matrix_file, walk_od_matrix_file)
    return output

def world_from_json(timestamp, tol, actuals_map_df, 
                    od_matrix_file = None, walk_od_matrix_file = None):
    """
    This method reads through the actuals file list and finds the value closest
    to the given timestamp, building the world object with the actual state as 
    listed in the json file.
    If no timestamp is found within the tolerance None is returned
    """
    output = None
    zip_filepath, filename = find_file_containing_timestamp(timestamp, 
                                                            actuals_map_df)
    if filename is not None:
        list_of_stns, list_of_timestamps = create_stations_from_zip(zip_filepath, filename)
        output = world_at_timestamp(timestamp, tol, list_of_stns, list_of_timestamps, od_matrix_file, walk_od_matrix_file)
    return output

def find_closest_timestamp(target_timestamp, tol, timestamp_series):
    """
    This method finds the closest timestamp to a target in a series of 
    timestamps. If no timestamp is found within the tolerance, None is 
    returned.
    """
    output = None
    upper = target_timestamp + tol
    lower = target_timestamp - tol
    nearby = timestamp_series[timestamp_series.between(lower, 
                                                       upper, 
                                                       inclusive='both')]
    closest_delta = tol*100
    for t in nearby:
        delta = abs(target_timestamp - t)
        if delta < closest_delta:
            closest_delta = delta
            output = t
    return output

def find_first_timestamp_before(target_timestamp, timestamp_series: pd.Series):
    """
    This method finds the first timestamp before a target in a series of
    timestamps. If there are no values before the target, None is returned
    """
    output = None
    before = timestamp_series[timestamp_series.le(target_timestamp)]
    if len(before.index) > 0:
        output = before.max()
    return output

def world_from_hdf_at_t(
    timestamp, 
    snapshot_timestamp,
    snapshots_hdf,
    od_matrix_file = None,
    walk_od_matrix_file = None
):
    output = None
    if snapshot_timestamp is not None:
        query = f'timestamp={snapshot_timestamp}'
        try:
            df = pd.read_hdf(snapshots_hdf, key='snapshots', where=query)
            closest_stn = Stations.new_stations_from_df(timestamp, df)
            output = World(timestamp, closest_stn, 
                        od_matrix_file, walk_od_matrix_file)
        except:
            print(f'No snapshot found at timestamp {snapshot_timestamp}')
    return output

def world_from_hdf(timestamp, tol, timestamp_series, snapshots_hdf, 
                   od_matrix_file = None, walk_od_matrix_file = None):
    output = None
    closest_t = find_closest_timestamp(timestamp, tol, timestamp_series)
    output = world_from_hdf_at_t(
        timestamp,
        closest_t,
        snapshots_hdf,
        od_matrix_file,
        walk_od_matrix_file
    )
    if output is None:
        print(f'No timestamp found within {tol}s of {timestamp}')
    return output

def main():
    print("Stating Test Run")

if __name__ == '__main__':
    main()