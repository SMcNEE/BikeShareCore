# imports
import numpy as np
import pandas as pd
from trips import timestep_from
from world import World
from datetime import datetime, timedelta
from numba import jit

class Operator:
    def __init__(self, world_to_operate: World, num_super_stations: int, num_trucks: int):
        self.op_world = world_to_operate
        self.num_super_stations = num_super_stations
        self.num_trucks = num_trucks
        # make numpy arrays to track the trucks
        self.truck_available = np.array([True for i in range(num_trucks)], bool)
        self.truck_distance = np.array(0.0 for i in range(num_trucks))
        # make numpy array to track the ids of superstations
        self.super_stations = np.empty(num_super_stations, dtype=int)
        self.super_station_capacities = np.empty(num_super_stations, dtype=int)
        self.super_station_positions = np.empty(num_super_stations, dtype=int)
        self.planned_truck_trips = []
        self.planned_dead_bike_removals = []
        self.bike_removal_log = []

    def set_superstation(self, position, id):
        # TODO make this update the superstations
        self.super_stations[position] = id
        stns = self.op_world.stations
        row_num = stns.station_keys[id]
        self.super_station_positions[position] = row_num
        self.super_station_capacities[position] = stns.station_capacities[row_num]
    
    def adjust_superstations(self):
        """
        This method adjusts the current occupancy of the super stations so that
        they will have at least 1 free bike or 1 empty station
        """
        timestep_changes = self.op_world.expected_arrivals[:,0]
        for i in range(self.num_super_stations):
            row = self.super_station_positions[i]
            stn_cap = self.super_station_capacities[i]
            timestep_change = timestep_changes[row]
            occupancy = self.op_world.stations.current_occupancy[row] + timestep_change
            # If the station would be empty, adjust to have at least one bike
            if occupancy <= 0:
                self.op_world.stations.current_occupancy[row] = 1 - timestep_change
            # If the station would be full, adjust to have at least one open
            # dock
            elif occupancy >= stn_cap:
                self.op_world.stations.current_occupancy[row] = (stn_cap - 1) - timestep_change

    def truck_trip(self, duration, num_bike_from, num_bike_to, from_id, to_id):
        """
        This method excecutes a truck trip to move bikes from one station to
        another
        """
        w = self.op_world
        stns = w.stations
        if abs(num_bike_from) > 0:
            from_row = stns.station_keys[from_id]
            self.op_world.expected_arrivals[from_row, 0] -= num_bike_from
        if abs(num_bike_to) > 0:
            to_row = stns.station_keys[to_id]
            self.op_world.expected_arrivals[to_row, int(duration)] += num_bike_to
    
    def plan_truck_trip(
        self, 
        start, 
        duration, 
        num_bike_from, 
        num_bike_to, 
        from_id, 
        to_id):
        # plan the truck trip
        self.planned_truck_trips.append((start, duration, num_bike_from, num_bike_to, from_id, to_id))
        # plan a removal of dead bikes at the arrival of the truck trip
        self.planned_dead_bike_removals.append((start+duration, to_id))

    def plan_truck_trips_from_df(
        self,
        trip_df: pd.DataFrame
    ):
        cols = ['d_early', 'duration', 'num_bikes_from', 'num_bikes_to', 'from_station', 'to_station']
        for row in trip_df[cols].to_records(index=False):
            self.plan_truck_trip(row[0], row[1], row[2], row[3], row[4], row[5])

    def dispatch_manager(self):
        w = self.op_world
        current_timestep = timestep_from(w.reference_time, w.current_time)
        for t in self.planned_truck_trips:
            if t[0] == current_timestep:
                self.truck_trip(t[1], t[2], t[3], t[4], t[5])
        for t in self.planned_dead_bike_removals:
            if t[0] == current_timestep:
                stn_id = t[1]
                num_removed = w.stations.remove_dead_bikes(stn_id)
                if num_removed > 0:
                    self.bike_removal_log.append((t[0], t[1], num_removed))
    
    def dead_bike_removal_log_to_df(self):
        cols = ['timestep', 'station_id', 'num_dead_bikes_removed']
        df = pd.DataFrame(
            self.bike_removal_log,
            columns=cols,
            index=[x for x in range(len(self.bike_removal_log))]
        )
        return df


def plan_truck_routing(capacity_details_df):
    df = capacity_details_df.reset_index()
    df = df.groupby(['stn_id', 'full_empty']).agg(
                first_reroute=('time', 'min'), 
                total_reroutes=('amount', 'sum'))
    df = df.reset_index()
    df = df.sort_values('total_reroutes', ascending=False)

def combine_route_check(i, i_tour, j, j_tour) -> bool:
    # First check if the stns are in each other's tours
    # if they are, tours cannot be combined
    if (i in j_tour) or (j in i_tour):
        return False
    i_pos = get_tour_position(i, i_tour)
    j_pos = get_tour_position(j, j_tour)
    # 0: No route assigned
    # 1: First node in route
    # 2: Node is interior to route
    # 3: Last node in route
    combo = False
    # if neither are in a route
    if (i_pos == 0) and (j_pos == 0):
        combo = True
    # if i is not in route and j is at start of route
    elif (i_pos == 0) and (j_pos == 1):
        combo = True
    # if i is at end of route and j is not in route
    elif (i_pos == 3) and (j_pos == 0):
        combo = True
    # if i is at the end of route and j is at begining of route
    elif (i_pos == 3) and (j_pos == 1):
        combo = True
    return combo

def get_new_route(i_tour, j_tour):
    new_tour = i_tour[:-1] + j_tour[1:]
    return new_tour

def compute_route_length(route, tt_df):
    l = 0
    for x in range(len(route)-1):
        i = route[x]
        j = route[x+1]
        l += tt_df.loc[(tt_df['Start Station Id'] == i) & (tt_df['End Station Id'] == j)].iloc[0]['travelTime']
    return l

def get_tour_position(i, i_tour):
    if len(i_tour) <= 3:
        position = 0
    elif i_tour[1] == i:
        position = 1
    elif i_tour[-2] == i:
        position = 3
    else:
        position = 2
    return position

def capacity_check(tour_reqs_df: pd.DataFrame, truck_size):
    check = False
    cumsum = tour_reqs_df['amount'].cumsum()
    max_load = cumsum.max()
    min_load = cumsum.min()
    spread = max_load - min_load
    if spread <= truck_size and max_load <= truck_size and abs(min_load) <= truck_size:
        check = True
    return check

def build_tour_timeline(
    tour_reqs_df: pd.DataFrame, 
    tt_df: pd.DataFrame, 
    earliest_departure: int,
    unload_time: int,
    max_wait_time: int,
    depot_location: int
):
    route_departures = []
    starts = tour_reqs_df['start_time'].to_numpy()
    ends = tour_reqs_df['end_time'].to_numpy()
    stns = tour_reqs_df.index.to_numpy()
    # compute the window from the first step from depot to first stn
    # ealiest and latest valid arrival times from depot to first stn
    earliest_a = starts[0] - max_wait_time
    latest_a = ends[0] - unload_time
    # window could arrive in
    tt_depot = tt_df.loc[(tt_df['Start Station Id'] == depot_location) & (tt_df['End Station Id'] == stns[0])].iloc[0]['travelTime']
    a_1 = earliest_departure + tt_depot
    # check if the window is feasible
    if (a_1 > latest_a):
        return None
    else:
        # the window is feasible, need to narrow the window for the next steop
        d_early = max(earliest_a, a_1) + unload_time
        d_late = ends[0]
        # Departure from the depot
        route_departures.append((depot_location, d_early-unload_time-tt_depot, d_late-unload_time-tt_depot, 0))
        # Departure from the first station
        route_departures.append((stns[0], d_early, d_late, tt_depot))
    for j in range(1, len(starts)):
        i = j-1
        tt = tt_df.loc[(tt_df['Start Station Id'] == stns[i]) & (tt_df['End Station Id'] == stns[j])].iloc[0]['travelTime']
        # constraints on arrival and service at node j
        # earliest valid arrival
        earliest_a = starts[j] - max_wait_time
        # latest valid arrival
        latest_a = ends[j] - unload_time
        # window could arrive in
        a_1 = d_early + tt
        a_2 = d_late + tt
        # check if the window is feasible
        if (a_1 > latest_a) or (a_2 < earliest_a):
            return None
        else:
            # the window is feasible, need to narrow the window for the next steop
            d_early = max(earliest_a, a_1) + unload_time
            d_late = min(latest_a, a_2) + unload_time
            route_departures.append((stns[j], d_early, d_late, tt))
    return route_departures

def time_check(
    tour_reqs_df: pd.DataFrame, 
    tt_df: pd.DataFrame, 
    earliest_departure: int,
    unload_time: int,
    max_wait_time: int,
    depot_location: int):
    # Check if the route can feasibly meet the time constraints
    check = True
    timeline = build_tour_timeline(
        tour_reqs_df, 
        tt_df, 
        earliest_departure, 
        unload_time, 
        max_wait_time, 
        depot_location
        )
    if timeline is None:
        check = False
    return check

def combine_tours(
    i, 
    j, 
    rebalance_req_df: pd.DataFrame, 
    tour_list: list, 
    truck_travel_times_df: pd.DataFrame, 
    truck_size: int,
    earliest_departure: int,
    unload_time: int,
    max_wait_time: int
    ):
    # Get the relevant rows to have their routes combined
    i_row = rebalance_req_df.loc[rebalance_req_df['station_id'] == i].iloc[0]
    j_row = rebalance_req_df.loc[rebalance_req_df['station_id'] == j].iloc[0]
    i_tour_num = int(i_row['tour'])
    j_tour_num = int(j_row['tour'])
    i_tour = tour_list[i_tour_num]
    j_tour = tour_list[j_tour_num]
    # perform the check to see if can combine tours
    if combine_route_check(i, i_tour, j, j_tour):
        # Get the new tour and the tour length
        combo_tour = get_new_route(i_tour, j_tour)
        tour_reqs_df = rebalance_req_df.set_index('station_id').loc[combo_tour[1:-1]]
        depot_location = combo_tour[0]
        if capacity_check(tour_reqs_df, truck_size):
            if time_check(tour_reqs_df, truck_travel_times_df, earliest_departure, unload_time, max_wait_time, depot_location):
                # Update the relevant items
                tour_list[i_tour_num] = combo_tour
                tour_list[j_tour_num] = combo_tour
                mask = rebalance_req_df['tour'] == j_tour_num
                rebalance_req_df['tour'].loc[mask] = i_tour_num

def clark_wright_savings(
    rebalance_req_df: pd.DataFrame,
    truck_travel_times_df: pd.DataFrame,
    depot_location: int,
    truck_size: int,
    earliest_departure: int,
    unload_time: int,
    max_wait_time:int
    ) -> tuple[pd.DataFrame, list]:
    # First, trim down the travel time matrix to only include relevant stations
    all_stns = rebalance_req_df['station_id']
    tt_df = truck_travel_times_df.loc[
        (truck_travel_times_df['Start Station Id'].isin(all_stns)) &
        (truck_travel_times_df['End Station Id'].isin(all_stns))
    ]
    depot_from_tt = truck_travel_times_df.loc[truck_travel_times_df['Start Station Id'] == depot_location]
    depot_to_tt = truck_travel_times_df.loc[truck_travel_times_df['End Station Id'] == depot_location]
    # Step 1 Compute the Savings and sort in descending order
    pairs = [[i, j] for j in all_stns for i in all_stns if i!=j]
    savings_df = pd.DataFrame(
        pairs,
        index = [x for x in range(len(pairs))],
        columns=['from', 'to']
    )
    savings_df['D_j'] = savings_df['to'].apply(
        (lambda x: depot_from_tt.loc[depot_from_tt['End Station Id'] == x].iloc[0]['travelTime'])
        )
    savings_df['i_D'] = savings_df['from'].apply(
        (lambda x: depot_to_tt.loc[depot_to_tt['Start Station Id'] == x].iloc[0]['travelTime'])
        )
    savings_df['i_j'] = savings_df[['from', 'to']].apply(
        (lambda x: tt_df.loc[(tt_df['Start Station Id'] == x['from']) & (tt_df['End Station Id'] == x['to'])].iloc[0]['travelTime']),
        axis = 1
        )
    savings_df['savings'] = savings_df['D_j'] + savings_df['i_D'] - savings_df['i_j']
    # Step 2 Rank savings in descending order
    savings_df = savings_df.sort_values('savings', ascending=False)

    # Step 3 Check about combining routes

    rebalance_req_df['tour'] = [int(x) for x in range(len(rebalance_req_df.index))]
    tour_list = [[depot_location, x, depot_location] for x in rebalance_req_df['station_id']]

    for index, row in savings_df.iterrows():
        # Check if can be combined
        combine_tours(
            row['from'], 
            row['to'], 
            rebalance_req_df, 
            tour_list, 
            truck_travel_times_df, 
            truck_size,
            earliest_departure,
            unload_time,
            max_wait_time
            )

    # print out the results
    print(rebalance_req_df)
    for tour in tour_list:
        print(tour, compute_route_length(tour, truck_travel_times_df))
    
    return rebalance_req_df.set_index('station_id'), tour_list

@jit(nopython=True)
def build_tour_list(
    stn_ids: np.ndarray,
    tours: np.ndarray,
    tour_positions: np.ndarray,
    depot: int
) -> list:
    """
    This method takes the tour information stored in numpy arrays and converts
    it into a list of lists with the nodes from the tours starting and ending
    at the depot location
    """
    all_tours = np.unique(tours)
    tour_list = []
    for t in all_tours:
        where = (tours == t)
        stns_in_t = stn_ids[where]
        positions_in_t = tour_positions[where]
        t_list = [0 for x in range(len(stns_in_t))]
        for p, s in zip(positions_in_t, stns_in_t):
            t_list[p] = s
        t_list = [depot] + t_list + [depot]
        tour_list.append(t_list)
    return tour_list

@jit(nopython=True)
def convert_tts_to_od_matrix(
    i_array: np.ndarray,
    j_array: np.ndarray,
    tt_array: np.ndarray
) -> np.ndarray:
    all_stns = np.unique(i_array)
    num_stns = len(all_stns)
    od_matrix = np.zeros((num_stns, num_stns), 'int64')
    for i, j, tt in zip(i_array, j_array, tt_array):
        od_matrix[i][j] = tt
    return od_matrix

@jit(nopython=True)
def convert_depot_tt_to_od(
    depot_from_times: np.ndarray,
    depot_from_j_values: np.ndarray,
    depot_to_times: np.ndarray,
    depot_to_i_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    depot_from = np.zeros_like(depot_from_times)
    for tt, j in zip(depot_from_times, depot_from_j_values):
        depot_from[j] = tt
    depot_to = np.zeros_like(depot_to_times)
    for tt, i in zip(depot_to_times, depot_to_i_values):
        depot_to[i] = tt
    return depot_from, depot_to

@jit(nopython=True)
def compute_savings(
    pairs: np.ndarray,
    tt_od: np.ndarray,
    depot_from_tt: np.ndarray, 
    depot_to_tt: np.ndarray
) -> np.ndarray:
    savings = np.zeros((pairs.shape)[0], dtype='int64')
    for p, savings_i in zip(pairs, range(len(savings))):
        i = p[0]
        j = p[1]
        savings[savings_i] = depot_from_tt[j] + depot_to_tt[i] - tt_od[i][j]
    savings_argsort = np.argsort(-savings)
    pairs_sorted = pairs[savings_argsort]
    savings_sorted = savings[savings_argsort]
    return pairs_sorted, savings_sorted

def clark_wright_savings_v2(
    rebalance_req_df: pd.DataFrame,
    truck_travel_times_df: pd.DataFrame,
    depot_location: int,
    truck_size: int,
    earliest_departure: int,
    unload_time: int,
    max_wait_time:int
    ) -> tuple[pd.DataFrame, list]:
    """
    This method re-implements the Clark Wright Savings Algorithm with the same
    input and outputs but with improved performance
    """
    # Convert the values to numpy arrays
    stn_ids = rebalance_req_df['station_id'].astype('int64').to_numpy()
    start_ts = rebalance_req_df['start_time'].astype('int64').to_numpy()
    end_ts = rebalance_req_df['end_time'].astype('int64').to_numpy()
    amounts = rebalance_req_df['amount'].astype('int64').to_numpy()
    tours = np.array([x for x in range(len(amounts))], dtype='int64')
    tour_positions = np.zeros_like(tours)
    i_values = np.array([x for x in range(len(amounts))], dtype='int64')
    map_dict = {
        s: i for s, i in zip(stn_ids, i_values)
    }
    # Convert the tt matrix into square tt matrix
    tt_df = truck_travel_times_df.loc[
        (truck_travel_times_df['Start Station Id'].isin(stn_ids)) &
        (truck_travel_times_df['End Station Id'].isin(stn_ids))
    ]
    i_values = (tt_df['Start Station Id'].map(map_dict)).to_numpy()
    j_values = (tt_df['End Station Id'].map(map_dict)).to_numpy()
    tt_od = convert_tts_to_od_matrix(
        i_values,
        j_values,
        tt_df['travelTime'].astype('int64').to_numpy(),
    )
    # Get the tt values to and from the depot
    depot_from_tt_df = truck_travel_times_df.loc[
        (truck_travel_times_df['Start Station Id'] == depot_location) &
        (truck_travel_times_df['End Station Id'].isin(stn_ids))
    ]
    depot_to_tt_df = truck_travel_times_df.loc[
        (truck_travel_times_df['End Station Id'] == depot_location) &
        (truck_travel_times_df['Start Station Id'].isin(stn_ids))
    ]
    depot_from_tt, depot_to_tt = convert_depot_tt_to_od(
        depot_from_tt_df['travelTime'].astype('int64').to_numpy(),
        (depot_from_tt_df['End Station Id'].map(map_dict)).to_numpy(),
        depot_to_tt_df['travelTime'].astype('int64').to_numpy(),
        (depot_to_tt_df['Start Station Id'].map(map_dict)).to_numpy()
    )
    print(tt_od)
    print(depot_from_tt)
    print(depot_to_tt)
    # Compute savings and rank in descending order
    pairs, savings = compute_savings(
        np.array([[i, j] for j in stn_ids for i in stn_ids if i!=j], dtype='int64'),
        tt_od,
        depot_from_tt,
        depot_to_tt
    )
    
    
    # Convert to the expected output format from original clark wright
    output_df = pd.DataFrame(
        {
            'station_id': stn_ids,
            'start_time': start_ts,
            'end_time': end_ts,
            'amount': amounts,
            'tour': tours
        }
    )
    tour_list = build_tour_list(stn_ids, tours, tour_positions, depot_location)
    return output_df.set_index('station_id'), tour_list

def build_rebalance_req_df(
    capacity_output: pd.DataFrame,
    reference_time: datetime
) -> pd.DataFrame:
    df = capacity_output.reset_index()
    df = df.groupby(['stn_id', 'full_empty']).agg(
                first_reroute=('time', 'min'), 
                total_reroutes=('amount', 'sum'))
    df = df.reset_index()
    df = df.sort_values('total_reroutes', ascending=False)
    df['start_time'] = 0
    df['end_time'] = df['first_reroute'].apply(lambda x: timestep_from(reference_time, x))
    col_translation = {
        'stn_id': 'station_id'
    }
    df = df.rename(columns=col_translation)
    full_empty_translation ={
        'full': 1,
        'empty': -1
    }
    df['mult'] = df['full_empty'].map(full_empty_translation)
    df['amount'] = df['mult'] * df['total_reroutes']
    cols = ['station_id', 'start_time', 'end_time', 'amount']

    return df[cols]

def build_rebalance_req_df_optimized(
    sim_time: int,
    sim_world: pd.DataFrame,
    simulation_trip_list
) -> pd.DataFrame:
    # Columns for the format of the utimate output
    cols = ['station_id', 'start_time', 'end_time', 'amount', 'stn_capacity']
    results = []
    # Now, with the warmed up world and the simulation trip list, find the
    # optimal times for rebalancing
    sim_stations = sim_world.stations
    expected_arr = sim_world.expected_arrivals
    df = simulation_trip_list
    df['arrival_timestep'] = df['timestep'] + df['rounded_duration']
    df['arrival_timestep'] = df['arrival_timestep'].astype('int64')
    df['timestep'] = df['timestep'].astype('int64')
    # Apply this to all stations
    list_of_stns = np.unique(sim_stations.station_ids)
    for stn_id in list_of_stns:
        i = sim_stations.station_keys[stn_id]
        start_occupancy = sim_stations.current_occupancy[i]
        stn_cap = sim_stations.station_capacities[i]
        stn_theo_cap = sim_stations.theoretical_capacities[i]
        trip_log = np.zeros((sim_time))
        # Get the trips already in the world from the expected arrivals
        trip_log[:120] = expected_arr[i]
        # Make a tally of trips arriving to and departing from the station
        arr = np.zeros((sim_time))
        dep = np.zeros((sim_time))
        from_trips = df.loc[df['from_station_id'] == stn_id]
        from_trips = from_trips.loc[from_trips['timestep'] < sim_time]
        for t in from_trips['timestep']:
            dep[t] -= 1
        to_trips = df.loc[df['to_station_id'] == stn_id]
        to_trips = to_trips.loc[to_trips['arrival_timestep'] < sim_time]
        for t in to_trips['arrival_timestep']:
            arr[t] += 1
        # Add these values to the trip log
        trip_log = trip_log + arr + dep
        # Compute the cumulative sum of the trip log
        cumulative = np.cumsum(trip_log)
        # Get the occupancy at each timestep
        occ = cumulative + start_occupancy
        # Check if the occupancy will violate the capacity conditions at any
        # time
        sim_start_time = sim_world.current_time
        start_time = 0
        cap_check_full = occ > stn_cap
        cap_check_empty = occ < 0
        cap_check = np.logical_or(cap_check_full, cap_check_empty)
        max_iterations = 10
        iterations = 0
        while(cap_check.any()):
            iterations += 1
            if iterations > max_iterations:
                print(f'Stn {stn_id} exceeded {max_iterations} iterations for {sim_world.current_time}')
                break
            problem_t = np.where(cap_check == True)[0][0]
            # simulate the removal of the dead bikes
            stn_cap = stn_theo_cap
            # get the bounds of possible rebalance movements
            x_min = -1*stn_cap
            x_max = stn_cap
            # If full highest rebalance possible is 0
            if occ[problem_t] > stn_cap:
                x_max = 0
            # If empty lowers rebalance possible is 0
            elif occ[problem_t] < 0:
                x_min = 0
            x_values = np.array([x for x in range(x_min, x_max + 1)])
            t_values = np.zeros_like(x_values)
            full_len = False # check that will cut search short if reach end
            for x, ind in zip(x_values, range(len(x_values))):
                if full_len is False:
                    test_occ = occ[problem_t:]
                    test_occ = test_occ + x
                    test_check = np.logical_or(test_occ > stn_cap, test_occ < 0)
                    # Get how long can go with this x
                    test_t = np.where(test_check == True)[0]
                    if test_t.size > 0:
                        test_t = test_t[0]
                    # If np.where finds nothing can get to the end of the sim
                    else:
                        test_t = len(test_occ)
                        full_len = True
                    t_values[ind] = test_t
            # Select the x value that gives the best result
            best_x = x_values[np.argmax(t_values)]
            # Update the result row
            end_time = int(problem_t)
            row = [stn_id, start_time, end_time, best_x, stn_cap]
            results.append(row)
            # Change the start time to be the end time
            # This way if a station needs multiple rebalancings the time window
            # will be correct
            start_time = end_time
            # Update the occupancy array with the rebalance
            occ[problem_t:] += best_x
            # Redo the capacity check to see if need another rebalance for this
            # stn
            cap_check_full = occ > stn_cap
            cap_check_empty = occ < 0
            cap_check = np.logical_or(cap_check_full, cap_check_empty)
    results_df = None
    # Combine all the results rows into a dataframe
    if len(results) > 0:
        results_df = pd.DataFrame(
            results,
            columns=cols,
            index=[x for x in range(len(results))]
        )
    return results_df

def build_rebalance_req_df_from_pinpoint(
    pinpoint_rebalance_df: pd.DataFrame,
    reference_time: datetime
) -> pd.DataFrame:
    cols = ['station_id', 'start_time', 'end_time', 'amount']
    df = pinpoint_rebalance_df.reset_index()
    df['start_time'] = 0
    df['end_time'] = df['rebalance_time'].apply(
        lambda x: timestep_from(reference_time, x) - 1
        )
    return df[cols]


def tour_to_truck_trips(
    rebalance_req_df: pd.DataFrame,
    tour: list,
    tt_df: pd.DataFrame,
    earliest_departure: int,
    unload_time: int,
    max_wait_time: int,
    depot_location:int
):
    """
    This method takes a tour and converts it to a list of trips representing 
    each leg of the tour
    """
    tour_reqs = rebalance_req_df.loc[tour[1:-1]]
    test_stns = tour_reqs.index
    route_departures = build_tour_timeline(
        tour_reqs,
        tt_df,
        earliest_departure,
        unload_time,
        max_wait_time,
        depot_location
    )
    loads = np.zeros_like(tour)
    loads[1:-1] = tour_reqs['amount'].to_numpy()
    trips_df = pd.DataFrame(
        route_departures,
        index=[x for x in range(len(route_departures))],
        columns=['from_station', 'd_early', 'd_late', 'travel_time']
    )
    from_stns = trips_df['from_station'].to_numpy()
    travelTimes = trips_df['travel_time'].to_numpy()
    travelTimes[:-1] = travelTimes[1:]
    travelTimes[-1] = tt_df.loc[(tt_df['Start Station Id'] == from_stns[-1]) & (tt_df['End Station Id'] == depot_location)].iloc[0]['travelTime']
    trips_df['travel_time'] = travelTimes
    to_stns = np.empty_like(from_stns)
    to_stns[:-1] = from_stns[1:]
    to_stns[-1] = depot_location
    trips_df['to_station'] = to_stns
    trips_df['num_bikes_to'] = loads[1:]
    trips_df['num_bikes_from'] = 0
    trips_df['duration'] = trips_df['travel_time'] + unload_time
    return trips_df

def tour_scheduler(
    rebalance_df_w_savings: pd.DataFrame,
    tour_list: list,
    truck_tt_df: pd.DataFrame,
    earliest_departure: int,
    unload_time: int,
    max_wait_time: int,
    depot_location: int
):
    trip_list = []
    valid_tours = rebalance_df_w_savings['tour'].unique()
    print(valid_tours)
    for t in valid_tours:
        trips = tour_to_truck_trips(
            rebalance_df_w_savings,
            tour_list[t],
            truck_tt_df,
            earliest_departure,
            unload_time,
            max_wait_time,
            depot_location
        )
        trips['tour'] = t
        trip_list.append(trips)
    trip_df = pd.concat(trip_list, ignore_index=True)
    return trip_df

def dispatch_trucks_no_tour(
    rebalance_reqs_df: pd.DataFrame,
    depot_location: int
) -> pd.DataFrame:
    """
    This method takes the rebalancing requirements dataframe and returns a list
    of truck trips which can be sent to the plan_truck_trips_from_df method.
    All trips are dispatched on the starting timestamp and arrive just before
    the requirement is needed.
    """
    cols = [
        'd_early', 
        'duration', 
        'num_bikes_from', 
        'num_bikes_to', 
        'from_station', 
        'to_station'
    ]
    df = rebalance_reqs_df.reset_index()
    df['d_early'] = df['end_time'] # all trucks are dispatched at the needed time
    df['duration'] = 0 # arrive at the needed timestep
    df['num_bikes_from'] = 0
    df['num_bikes_to'] = df['amount']
    df['from_station'] = depot_location
    df['to_station'] = df['station_id']
    return df[cols]

def tour_len_from_rebalance_req(
    rebalance_req: pd.DataFrame,
    truck_travel_times: pd.DataFrame,
    depot_location: int,
    truck_size: int,
    earliest_departure: int,
    unload_time: int,
    max_wait_time: int
) -> tuple[int, int]:
    req, tour_list = clark_wright_savings(
        rebalance_req,
        truck_travel_times,
        depot_location,
        truck_size,
        earliest_departure,
        unload_time,
        max_wait_time
    )
    used_tours = req['tour'].unique()
    results = []
    num_tours = len(used_tours)
    total_length = 0
    for t in used_tours:
        total_length += compute_route_length(tour_list[t], truck_travel_times)
    return num_tours, total_length

def main():
    print('Starting test run')
    tt_array = np.array(
        [
            [0, 25, 43, 57, 43, 61, 29],
            [25, 0 ,29, 34, 43, 68, 49],
            [43, 29, 0, 52, 72, 96, 72],
            [57, 34, 52, 0, 45, 71, 71],
            [43, 43, 72, 45, 0, 27, 36],
            [61, 68, 96, 71, 27, 0, 40],
            [29, 49, 72, 71, 36, 40, 0]
        ]
    )
    truck_tt = np.empty((7**2, 3))
    counter = 0
    for i in range(7):
        for j in range(7):
            truck_tt[counter][0] = i
            truck_tt[counter][1] = j
            truck_tt[counter][2] = tt_array[i][j]
            counter += 1
    truck_tt_df = pd.DataFrame(
        truck_tt, 
        index=[x for x in range(49)], 
        columns=['Start Station Id', 'End Station Id', 'travelTime'])
    rebalance_req = pd.DataFrame(
        [
            [1, 10*60, 14*60, 4],
            [2, 13*60, 14*60, 6],
            [3, 8*60, 10*60, 5],
            [4, 8*60, 10*60, 4],
            [5, 11*60, 12*60, 7],
            [6, 10*60, 14*60, 3]
        ],
        index = [x for x in range(6)],
        columns=['station_id', 'start_time', 'end_time', 'amount']
    )
    
    depot_location = 0
    truck_size = 18
    earliest_departure = 8*60
    unload_time = 60,
    max_wait_time = 60
    rebalance_req_w_savings, tour_list = clark_wright_savings(
        rebalance_req, 
        truck_tt_df, 
        depot_location, 
        truck_size, 
        earliest_departure, 
        unload_time, 
        max_wait_time
    )
    trip_df = tour_scheduler(
        rebalance_req_w_savings,
        tour_list,
        truck_tt_df,
        earliest_departure,
        unload_time,
        max_wait_time,
        depot_location
    )

    # Test the new version
    rebalance_req_w_savings, tour_list = clark_wright_savings_v2(
        rebalance_req, 
        truck_tt_df, 
        depot_location, 
        truck_size, 
        earliest_departure, 
        unload_time, 
        max_wait_time
    )
    # print out the results
    print(rebalance_req_w_savings)
    for tour in tour_list:
        print(tour, compute_route_length(tour, truck_tt_df))

if __name__ == '__main__':
    main()
