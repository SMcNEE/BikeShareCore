from os import times
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import zipfile
import io

class Stations:
    def __init__(self, timestamp, ids, capacities, occupancy, theoretical_capcities=None):
        self.current_time = datetime.fromtimestamp(timestamp)
        self.station_ids = ids # int array
        keys = {}
        # Build the dictionary to find the index of the ids
        for id, i in zip(self.station_ids, range(len(self.station_ids))):
            keys[id] = i
        self.station_keys = keys
        self.station_capacities = capacities
        self.current_occupancy = occupancy
        # track the bikes and docks that are out of service
        if theoretical_capcities is None:
            self.theoretical_capacities = capacities
        else:
            self.theoretical_capacities = theoretical_capcities
    
    def make_station_detail(list_of_properties):
        station_details = {
            "id": list_of_properties[0],
            "name": list_of_properties[1],
            "physical_configuration": list_of_properties[2],
            "lat": list_of_properties[3],
            "lon": list_of_properties[4],
            "altitude": list_of_properties[5],
            "address": list_of_properties[6],
            "capacity": list_of_properties[7],
            "rental_method": list_of_properties[8],
            "groups": list_of_properties[9],
            "obcn": list_of_properties[10],
            "nearby_distance": list_of_properties[11],
            "post_code": list_of_properties[12],
            "cross_street": list_of_properties[13]
        }
        return station_details

    def set_station_details(self, station_details):
        id = station_details["id"]
        self.station_keys[id] = station_details

    def set_system_properties(self, ids, capacities):
        self.station_ids = ids
        self.station_capacities = capacities
    
    def change_time(self, new_timestamp):
        self.current_time = datetime.fromtimestamp(new_timestamp)

    def increment_time(self, days, hours, minutes, seconds=0, milliseconds=0, microseconds=0):
        delta = timedelta(days, seconds, microseconds, milliseconds, minutes, hours)
        self.current_time = self.current_time + delta

    def set_occupancy(self, occupancy):
        self.current_occupancy = occupancy
    
    def increment_occupancy(self, occupancy_changes, enforce_capacity=True):
        capacity = self.station_capacities
        stn_ids = self.station_ids
        new_occupancy = self.current_occupancy + occupancy_changes
        # Initiallize return values which will track the capacity issues
        full_stn_ids = None
        over_amounts = None
        empty_stn_ids = None
        under_amounts = None
        # if have capacity limits, check upper and lower
        if enforce_capacity:
            # check that no values are negative
            under = new_occupancy < 0
            empty_stn_ids = stn_ids[under]
            under_amounts = new_occupancy[under] * -1
            new_occupancy[under] = 0
            # check that no values exceed the capacity
            over = new_occupancy > capacity
            full_stn_ids = stn_ids[over]
            over_amounts = new_occupancy[over] - capacity[over]
            new_occupancy[over] = capacity[over]
        # new_occupancy[new_occupancy > capacity] = capacity
        self.current_occupancy = new_occupancy
        return empty_stn_ids, under_amounts, full_stn_ids, over_amounts

    def parse_single_snapshot(snapshot):
        end_index = len(snapshot)-1
        x=snapshot.find('"last_updated":', 0, end_index)
        x = x + 15 # change the start index to the end of last_updated
        y = snapshot.find(",", x, end_index)
        i=snapshot.find("[", y, end_index)
        j=snapshot.find("]", i, end_index)
        return snapshot[x:y], snapshot[i:j+1]
    
    def split_station_snapshots(snapshots, list_of_snapshots):
        length = len(snapshots)
        i = snapshots.find("{")
        j = snapshots.find("}}")
        # if don't find the pattern, return the current list
        if i == -1 or j == -1:
            return list_of_snapshots
        snap = snapshots[i:j+1]
        list_of_snapshots.append(snap)
        snap_length = len(snap)
        remainder = length - snap_length - i
        # if there is still more text in the remainder search recursively
        if remainder >= 1:
            rest = snapshots[j+1:]
            return Stations.split_station_snapshots(rest, list_of_snapshots)
        # if no remainder return the list
        else:
            return list_of_snapshots

    def new_stations_from_df(timestamp, df):
        # ensure station ids are in order
        df.sort_values(by=["station_id"])
        # make sure the values are integers
        df['station_id'] = df['station_id'].astype('int64')
        df['num_bikes_available'] = df['num_bikes_available'].astype('int64')
        df['num_docks_available'] = df['num_docks_available'].astype('int64')
        df['num_bikes_disabled'] = df['num_bikes_disabled'].astype('int64')
        # get numpy arrays for making staiton object
        ids = df["station_id"].to_numpy()
        cap = df["num_bikes_available"] + df["num_docks_available"]
        capacities = cap.to_numpy()
        occupancy = df["num_bikes_available"].to_numpy()
        # Get the theoretical capacity using the mikes disabled
        cap_theo = cap + df['num_bikes_disabled']
        cap_theo = cap_theo.to_numpy()
        # make new station object
        new_stations = Stations(timestamp, ids, capacities, occupancy, cap_theo)
        return new_stations

    def compare_occupancy(self, stn_to_compare):
        # TODO add check for same time
        # TODO add check for same size
        delta = stn_to_compare.current_occupancy - self.current_occupancy
        return delta
    
    def match_station_ids(self, new_station_ids):
        """
        This method updates the stations object to match the station ids
        given as argument.
        The order of each of the arrays is updated as well as the keys
        Returns False if object is unchanged, True if was changed
        """
        old_station_ids = self.station_ids
        # if station ids already match given, do nothing
        same = False
        changed = False
        if old_station_ids.shape == new_station_ids.shape:
            if np.equal(old_station_ids, new_station_ids).all():
                same = True
        if same is False:
            old_capacities = self.station_capacities
            old_occupancies = self.current_occupancy
            old_theo_capacities = self.theoretical_capacities
            old_keys = self.station_keys
            # Build the dictionary to find the index of the ids
            # Make the capacities and occupancy arrays
            new_capacities = np.zeros_like(new_station_ids, int)
            new_occupancies = np.zeros_like(new_station_ids, int)
            new_theo_capacities = np.zeros_like(new_station_ids, int)
            new_keys = {}
            # Loop through the new station ids and get the values for the ids
            # from the old arrangement
            # If the id was not in the old ids list fill with 0
            for id, i in zip(new_station_ids, range(len(new_station_ids))):
                new_keys[id] = i
                try:
                    old_index = old_keys[id]
                    new_capacities[i] = old_capacities[old_index]
                    new_occupancies[i] = old_occupancies[old_index]
                    new_theo_capacities[i] = old_theo_capacities[old_index]
                except:
                    new_capacities[i] = 0
                    new_occupancies[i] = 0
                    new_theo_capacities[i] = 0
            # Update the values in the stations object
            self.station_ids = new_station_ids
            self.station_keys = new_keys
            self.current_occupancy = new_occupancies
            self.station_capacities = new_capacities  
            self.theoretical_capacities = new_capacities  
            changed = True
        return changed

    def remove_dead_bikes(self, stn_id):
        """
        This method changes the station capacities to the theoretical capacities
        simulating the removal of dead bikes from the system
        """
        num_dead_bikes = 0
        try:
            key = self.station_keys[stn_id]
            num_dead_bikes = self.theoretical_capacities[key] - self.station_capacities[key]
            if num_dead_bikes > 0:
                self.station_capacities[key] = self.theoretical_capacities[key]
        except:
            print(f'Key not found for {stn_id}, could not remove dead bikes')
        return num_dead_bikes    

def create_stations_from_json(file):
    data = file.read()
    # with open(file, "r") as f:
    #     data = f.read()
    list_of_snapshots = Stations.split_station_snapshots(data, [])
    list_of_dfs = []
    list_of_timestamps = []
    for snapshot in list_of_snapshots:
        timestamp, snapshot_data = Stations.parse_single_snapshot(snapshot)
        new_stations_data = pd.read_json(snapshot_data, orient="records")
        list_of_timestamps.append(int(timestamp))
        list_of_dfs.append(new_stations_data)
    # create the station objects
    list_of_stns = []
    for timestamp, df in zip(list_of_timestamps, list_of_dfs):
        list_of_stns.append(Stations.new_stations_from_df(timestamp, df))
    return list_of_stns, list_of_timestamps

def create_stations_from_zip(zip_filepath, filename):
    list_of_stns = None
    list_of_timestamps = None
    with zipfile.ZipFile(zip_filepath, 'r') as z:
        with io.TextIOWrapper(z.open(filename), encoding='utf-8') as f:
            # Create the station object for every instance in the json
            list_of_stns, list_of_timestamps = create_stations_from_json(f)   
    return list_of_stns, list_of_timestamps 
        