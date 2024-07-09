# imports
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trips import read_df_trip_list
from world import World, find_closest_timestamp, world_from_hdf, find_first_timestamp_before, world_from_hdf_at_t
from stations import Stations
from time import perf_counter
from tqdm import tqdm
from bike_operator import Operator, build_rebalance_req_df, build_rebalance_req_df_from_pinpoint, build_rebalance_req_df_optimized, dispatch_trucks_no_tour, tour_len_from_rebalance_req
import copy
from multiprocessing import Pool
from itertools import repeat

def get_trips_between(
    start_time: datetime, 
    end_time: datetime,
    trips_hdf_path: str
) -> pd.DataFrame:
    d_form = "%Y%m%d"
    buffer_day = timedelta(1, 0, 0, 0, 0)
    start_str = (start_time - buffer_day).strftime(d_form)
    end_str = (end_time + buffer_day).strftime(d_form)
    trip_dates = f'index >= {start_str} & index <= {end_str}'
    trip_list = pd.read_hdf(trips_hdf_path, key='trips', where=[trip_dates])
    trip_list = trip_list.reset_index()
    trip_list = trip_list.loc[(trip_list['trip_start_time'] >= start_time) & 
                              (trip_list['trip_start_time'] <= end_time)]
    return trip_list

def find_warmup(
    target_world: World, 
    min_warmup: int, 
    max_warmup: int,
    timestamp_series: pd.Series,
    snapshots_hdf: str,
    bike_od_file: str = None,
    walk_od_file: str = None
) -> tuple[World, int]: 
    rebase_dt = target_world.current_time
    warmup_world = None
    warmup_steps = None
    warmup_delta = timedelta(0,0,0,0, -1*min_warmup)
    target_warmup_time = rebase_dt + warmup_delta
    warmup_start = find_first_timestamp_before(
        datetime.timestamp(target_warmup_time), 
        timestamp_series
        )
    if warmup_start is not None:
        closest_t = datetime.fromtimestamp(warmup_start)
        y = closest_t.year
        m = closest_t.month
        d = closest_t.day
        h = closest_t.hour
        minute = closest_t.minute
        rounded_t = datetime(y, m, d, h, minute, 0, 0)
        rounded_timestamp = datetime.timestamp(rounded_t)
        warmup_world = world_from_hdf_at_t(
            rounded_timestamp,
            warmup_start,
            snapshots_hdf,
            bike_od_file,
            walk_od_file
        )
        minutes = divmod((rebase_dt - rounded_t).seconds, 60)
        warmup_steps = minutes[0]
    
    return warmup_world, warmup_steps

def warmup_world(
    cold_world: World, 
    rebase_world: World, 
    trips: pd.DataFrame,
    steps: int
) -> World:
    """
    This method takes a cold world which has no trips tracked in the expected
    arrivals and applies trips to warm the system up. The station occupancy
    is then set to that of the rebase world.

    Returns: World with occupancy of rebase world and trips in expected 
    arrivals
    """
    # get the trip list into format readable to the step through time function
    max_trip_length = 120
    warmup_trips = read_df_trip_list(trips, 
                                     cold_world.current_time, 
                                     max_trip_length
                                     )
    # step through time to apply the trips
    cold_world.step_through_time(warmup_trips, steps, 60, False)
    # set the station ocucpancies to the rebase world
    cold_world.rebase_world(rebase_world)
    return cold_world

def find_world_on_hour_before(
    year: int,
    month: int,
    day: int,
    last_hour: int, 
    tol: int, 
    timestamp_serires: pd.Series,
    snapshots_hdf: str
) -> World:
    """
    This method finds a snapshot from the data on the given day.
    Only snapshots within the tolerance on the hour are considered
    """
    target_world = None
    hour = None
    # Try to find a snapshot on the hour between 0:00 and the hour given
    for h in range(last_hour + 1):
        if target_world is None:
            hour = h
            dt = datetime(year, month, day, hour, 0, 0, 0)
            timestamp = datetime.timestamp(dt)
            target_world = world_from_hdf(timestamp,
                                          tol,
                                          timestamp_serires,
                                          snapshots_hdf)
    return target_world

def get_closest_world_to_t(
    t: datetime, 
    max_delta_s: int, 
    timestamp_series: pd.Series,
    snapshots_hdf: str,
    bike_od_file: str = None,
    walk_od_file: str = None
) -> World:
    target_world = None
    target_timestamp = datetime.timestamp(t)
    closest_timestamp = find_closest_timestamp(target_timestamp,
                                               max_delta_s,
                                               timestamp_series)
    # Get a timestamp rounded down to the nearest minute
    if closest_timestamp is not None:
        closest_t = datetime.fromtimestamp(closest_timestamp)
        y = closest_t.year
        m = closest_t.month
        d = closest_t.day
        h = closest_t.hour
        minute = closest_t.minute
        rounded_t = datetime(y, m, d, h, minute, 0, 0)
        rounded_timestamp = datetime.timestamp(rounded_t)
        query = f'timestamp={closest_timestamp}'
        try:
            df = pd.read_hdf(snapshots_hdf, key='snapshots', where=query)
            stn = Stations.new_stations_from_df(rounded_timestamp, df)
            target_world = World(rounded_timestamp,
                                 stn,
                                 bike_od_file,
                                 walk_od_file)
        except:
            print(f'No snapshot found at timestamp {closest_t}')
    else:
        print(f'No timestamp found within {max_delta_s}s of {target_timestamp}')
    return target_world

def get_warmed_up_world(
    start_time: datetime,
    sim_time_min: int,
    timestamp_series: pd.Series,
    snapshots_filepath: str,
    trips_filepath: str,
    bike_od_filepath: str,
    walk_od_filepath: str
) -> tuple[World, pd.DataFrame, pd.DataFrame, int, datetime, datetime]:
    """
    This method gets returns a warmed up world along with the trips for the
    simulation and the simulation time.
    This is used by the bike operator to then build an optimal rebalancing
    strategy.
    """
    null_result = (None for x in range(6))
    # Get the warmup world and the simulation world and run the warmup
    # This is the same as the beginning of run_simulation
    sim_world = get_closest_world_to_t(
        start_time,
        3*60*60,
        timestamp_series,
        snapshots_filepath,
        bike_od_filepath,
        walk_od_filepath
    )
    if sim_world is None:
        return null_result
    warmup, steps = find_warmup(
        sim_world,
        120,
        480,
        timestamp_series,
        snapshots_filepath,
        bike_od_filepath,
        walk_od_filepath
    )
    if warmup is None:
        return null_result
    warmup_start = warmup.current_time
    warmup_trips = get_trips_between(
        warmup_start,
        sim_world.current_time,
        trips_filepath
    )
    sim_world = warmup_world(
        warmup,
        sim_world,
        warmup_trips,
        steps
    )
    sim_world.reset_results_counters()
    sim_start = sim_world.current_time
    sim_end_time = sim_start + timedelta(0, 0, 0, 0, sim_time_min)
    simulation_trip_df = get_trips_between(
        sim_start,
        sim_end_time,
        trips_filepath
    )
    max_trip_length = 120
    simulation_trip_list = read_df_trip_list(
        simulation_trip_df,
        sim_start,
        max_trip_length
    )
    results = (
        sim_world, 
        simulation_trip_df,
        simulation_trip_list, 
        sim_time_min, 
        sim_start, 
        warmup_start
    )
    return results

def run_simulation_from_warmed(
    warmup_results: tuple,
    start_time: datetime, 
    simulation_time_min: int,
    list_of_super_stations = None,
    list_of_truck_trips = None,
):
    simulation_world = warmup_results[0]
    simulation_trip_df = warmup_results[1]
    simulation_trip_list = warmup_results[2]
    sim_time_min = warmup_results[3]
    sim_start = warmup_results[4]
    warmup_start = warmup_results[5]
    # If truck trips or super stations are given, initialize the Operator
    system_operator = None
    if list_of_super_stations is not None or list_of_truck_trips is not None:
        if list_of_super_stations is not None:
            num_super_stations = len(list_of_super_stations)
        else:
            num_super_stations = 0
        system_operator = Operator(simulation_world, num_super_stations, 1)
    if list_of_super_stations is not None:
        for i in range(len(list_of_super_stations)):
            system_operator.set_superstation(i, list_of_super_stations[i])
    if list_of_truck_trips is not None:
        system_operator.plan_truck_trips_from_df(list_of_truck_trips)

    # print(f'Simulation world refence time: {simulation_world.reference_time}')
    results = simulation_world.step_through_time(simulation_trip_list,
                                       simulation_time_min,
                                       60,
                                       enforce_capacity=True,
                                       operator=system_operator)
    sim_end = simulation_world.current_time
    # print(f'Simulation end at {sim_end}')
    full_trips = simulation_world.trips_delayed_by_full
    full_delay = simulation_world.time_from_full_delay
    raw_full_delays = simulation_world.full_delays
    # print(f'{simulation_world.given_trips} trips given')
    # print(f'{simulation_world.completed_trips} trips served')
    # print(f'{full_trips} trips delayed by full stations, resulting in {full_delay} min of delay')
    empty_trips = simulation_world.trips_delayed_by_empty
    empty_delay = simulation_world.time_from_empty_delay
    raw_empty_delays = simulation_world.empty_delays
    # print(f'{empty_trips} trips delayed by empty stations, resulting in {empty_delay} min of delay')
    # print(f'{simulation_world.empty_failed_to_reroute} trips from empty stations could not be rerouted')
    # print(f'Simulation complete in {perf_counter() - run_start}s') 
    result = pd.DataFrame(
        {
            'input_start': start_time,
            'simulation_duration_min': simulation_time_min,
            'warmup_start': warmup_start,
            'sim_start': sim_start,
            'sim_end': sim_end,
            'num_trips': simulation_trip_df.shape[0],
            'num_trips_trimmed': simulation_trip_list.shape[0],
            'num_trips_served': simulation_world.completed_trips,
            'num_delayed_by_full': full_trips,
            'full_stn_delay_min': full_delay,
            'num_delayed_by_empty': empty_trips,
            'empty_stn_delay_min': empty_delay,
            'num_failed_to_reroute': simulation_world.empty_failed_to_reroute,
            'total_bikes_in_world': simulation_world.num_bikes_in_world()
        }, index=[0]
    )
    capacity_details = results[4]
    failed_trips = results[3]
    # Get the log of the dead bikes removed
    dead_bike_removal = None
    if system_operator is not None:
        dead_bike_removal = system_operator.dead_bike_removal_log_to_df()
        dead_bike_removal['sim_start'] = sim_start
    output_tuple = (
        result, 
        capacity_details, 
        failed_trips, 
        raw_full_delays, 
        raw_empty_delays,
        dead_bike_removal
    )
    return output_tuple

def run_simulation(
    start_time: datetime, 
    simulation_time_min: int, 
    snapshots_filepath: str,
    timestamp_series: pd.Series, 
    trips_filepath: str,
    bike_od_filepath: str,
    walk_od_filepath: str,
    list_of_super_stations = None,
    list_of_truck_trips = None
):
    warmup_results = get_warmed_up_world(
        start_time,
        simulation_time_min,
        timestamp_series,
        snapshots_filepath,
        trips_filepath,
        bike_od_filepath,
        walk_od_filepath
    )
    output_tuple = run_simulation_from_warmed(
        warmup_results,
        start_time,
        simulation_time_min,
        list_of_super_stations,
        list_of_truck_trips
    )
    result, capacity_details, failed_trips, raw_full_delays, raw_empty_delays, dead_bike_removal = output_tuple
    return result, capacity_details, failed_trips, raw_full_delays, raw_empty_delays, dead_bike_removal

def run_simulation_from_df(
    inputs_df: pd.DataFrame,
    snapshots_filepath: str, 
    trips_filepath: str,
    bike_od_filepath: str,
    walk_od_filepath: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp_series = pd.read_hdf(snapshots_filepath, key='timestamps')
    results_list = []
    capacity_details_list = []
    failed_trips_list = []
    full_delays = []
    empty_delays = []
    dead_bikes_removal = []
    print(f'{len(inputs_df.index)} simulations to run')
    for index, row in tqdm(inputs_df.iterrows()):
        try:
            result, capacity_details, failed_trips, raw_full_delays, raw_empty_delays, dead_bike_removal = run_simulation(
                row['start_time'],
                60*24,
                snapshots_filepath,
                timestamp_series,
                trips_filepath,
                bike_od_filepath,
                walk_od_filepath
            )
            if capacity_details is not None:
                capacity_details_list.append(capacity_details)
            if failed_trips is not None:
                failed_trips_list = failed_trips_list + failed_trips
            if len(raw_full_delays) > 0:
                full_delays = full_delays + raw_full_delays
            if len(raw_empty_delays) > 0:
                empty_delays = empty_delays + raw_empty_delays
            if dead_bike_removal is not None:
                dead_bikes_removal.append(dead_bike_removal)
        except:
            result = pd.DataFrame(
            {
                'input_start': row['start_time'],
                'simulation_duration_min': np.nan,
                'sim_start': np.nan,
                'sim_end': np.nan,
                'num_trips': np.nan,
                'num_trips_trimmed': np.nan,
                'num_trips_served': np.nan,
                'num_delayed_by_full': np.nan,
                'full_stn_delay_min': np.nan,
                'num_delayed_by_empty': np.nan,
                'empty_stn_delay_min': np.nan,
                'num_failed_to_reroute': np.nan,
                'total_bikes_in_world': np.nan
            }, index=[0]
            )
        results_list.append(result)
    results_df = pd.concat(results_list, ignore_index=True)
    capacity_details_df = None
    if len(capacity_details_list) > 0:
        capacity_details_df = pd.concat(capacity_details_list, ignore_index=True)
    failed_trips_df = pd.DataFrame(
        failed_trips_list, 
        index=[x for x in range(len(failed_trips_list))],
        columns=['start', 'end']
        )
    full_delays_df = pd.DataFrame(
        full_delays,
        index=[x for x in range(len(full_delays))],
        columns=['full_delay_mins']
    )
    empty_delays_df = pd.DataFrame(
        empty_delays,
        index=[x for x in range(len(empty_delays))],
        columns=['empty_delay_mins']
    )
    if len(dead_bikes_removal) > 0:
        dead_bikes_removal_combo = pd.concat(dead_bikes_removal, ignore_index=True)
    return results_df, capacity_details_df, failed_trips_df, full_delays_df, empty_delays_df, dead_bikes_removal_combo

def daterange(start_date, end_date):
    """
    Generator function to get dates between start and end
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def triple_simulation(
    sim_date: datetime,
    depot_location: int,
    pinpoint_path: str,
    hdf_file: str,
    timestamp_series: pd.Series,
    trips_file: str,
    bike_od_file: str,
    walk_od_file: str
): 
    # Initialize the outputs to none. If a simulation fails, none will be
    # returned for that value
    no_reb_results = None
    sim_from_cap_output_results = None
    sim_from_pinpoint_results = None
    # Get the parameters for the simulation
    y = sim_date.year
    m = sim_date.month
    d = sim_date.day
    simulation_time = 24*60 # simulate for a full day
    # Get the warmed up world to be used for the simulation
    warmup_results = get_warmed_up_world(
        sim_date,
        simulation_time,
        timestamp_series,
        hdf_file,
        trips_file,
        bike_od_file,
        walk_od_file
    )
    if (warmup_results[0] is not None) and (warmup_results[1] is not None):
        # run the simulation without any rebalancing
        no_reb_results = run_simulation_from_warmed(
            copy.deepcopy(warmup_results),
            sim_date,
            simulation_time,
            list_of_super_stations=None,
            list_of_truck_trips=None
        )
        # run the simulation with optimized rebalancing
        # Get the reference time
        ref_time = no_reb_results[0]['sim_start'].iloc[0]
        # Generate the rebalancing reqs from the output
        capacity_output = no_reb_results[1]
        # rebalance_req_from_no_reb = build_rebalance_req_df(
        #     capacity_output,
        #     ref_time
        # )
        # problem_stns = capacity_output['stn_id'].unique()
        warmed_up_world_copy = copy.deepcopy(warmup_results[0])
        simulation_trip_list_copy = copy.deepcopy(warmup_results[2])
        rebalance_req_from_optimized = build_rebalance_req_df_optimized(
            simulation_time,
            warmed_up_world_copy,
            simulation_trip_list_copy
        )
        truck_trips = dispatch_trucks_no_tour(
            rebalance_req_from_optimized,
            depot_location
        )
        sim_from_cap_output_results = run_simulation_from_warmed(
            copy.deepcopy(warmup_results),
            sim_date,
            simulation_time,
            list_of_super_stations=None,
            list_of_truck_trips=truck_trips
        )
        # run the simulation with observed rebalancings
        # Read the pinpoint file
        pinpoint_df = None
        try:
            pinpoint_df = pd.read_csv(pinpoint_path, parse_dates=['rebalance_time'])
        except:
            print(f'Failed to read pinpoint file {pinpoint_path}')
        if pinpoint_df is not None:
            pinpoint_df['rebalance_time'] = pinpoint_df['rebalance_time'].dt.tz_localize(None)
            rebalance_req_from_pinpoint = build_rebalance_req_df_from_pinpoint(
                pinpoint_df,
                ref_time
            )
            truck_trips_from_pinpoint = dispatch_trucks_no_tour(
                rebalance_req_from_pinpoint,
                depot_location
            )
            sim_from_pinpoint_results = run_simulation_from_warmed(
                copy.deepcopy(warmup_results),
                sim_date,
                simulation_time,
                list_of_super_stations=None,
                list_of_truck_trips=truck_trips_from_pinpoint
            )
    else:
        print(f'Could not find snapshots for simulation on {sim_date}')
    
    # Return the results from each simulation
    # if the given simulation failed, none is returned
    return no_reb_results, sim_from_cap_output_results, sim_from_pinpoint_results

def build_rebaace_req_df_from_results(
    sim_date: datetime,
    depot_location: int,
    hdf_file: str,
    timestamp_series: pd.Series,
    trips_file: str,
    bike_od_file: str,
    walk_od_file: str
) -> pd.DataFrame:
    simulation_time = 24*60 # simulate for a full day
    # Get the warmed up world to be used for the simulation
    warmup_results = get_warmed_up_world(
        sim_date,
        simulation_time,
        timestamp_series,
        hdf_file,
        trips_file,
        bike_od_file,
        walk_od_file
    )
    next_day = sim_date + timedelta(1, 0, 0, 0, 0)
    # capacity_output = capacity_output.loc[(capacity_output['time'] >= sim_date) &
    #                                       (capacity_output['time'] < next_day)]
    # problem_stns = capacity_output['stn_id'].unique()
    warmed_up_world = warmup_results[0]
    simulation_trip_list = warmup_results[2]
    rebalance_req_df = build_rebalance_req_df_optimized(
        simulation_time,
        warmed_up_world,
        simulation_trip_list
    )
    return rebalance_req_df

def get_truck_tour_details(
    sim_date: datetime,
    depot_location: int,
    pinpoint_path: str,
    hdf_file: str,
    timestamp_series: pd.Series,
    trips_file: str,
    bike_od_file: str,
    walk_od_file: str,
    truck_od: pd.DataFrame
) -> pd.DataFrame:
    simulation_time = 24*60 # simulate for a full day
    # Get the warmed up world to be used for the simulation
    results = []
    truck_size = 25
    earliest_departure = 0
    unload_time = 10
    max_wait_time = 60
    warmup_results = get_warmed_up_world(
        sim_date,
        simulation_time,
        timestamp_series,
        hdf_file,
        trips_file,
        bike_od_file,
        walk_od_file
    )
    if (warmup_results[0] is not None) and (warmup_results[1] is not None):
        # Get the rebalance req df
        warmed_up_world_copy = copy.deepcopy(warmup_results[0])
        simulation_trip_list_copy = copy.deepcopy(warmup_results[2])
        ref_time = warmed_up_world_copy.current_time
        rebalance_req_from_optimized = build_rebalance_req_df_optimized(
            simulation_time,
            warmed_up_world_copy,
            simulation_trip_list_copy
        )
        num_tours, total_length = tour_len_from_rebalance_req(
            rebalance_req_from_optimized,
            truck_od,
            depot_location,
            truck_size,
            earliest_departure,
            unload_time,
            max_wait_time
        )
        results.append(['optimized', sim_date, num_tours, total_length])
        # Read the pinpoint file
        pinpoint_df = None
        try:
            pinpoint_df = pd.read_csv(pinpoint_path, parse_dates=['rebalance_time'])
        except:
            print(f'Failed to read pinpoint file {pinpoint_path}')
        if pinpoint_df is not None:
            pinpoint_df['rebalance_time'] = pinpoint_df['rebalance_time'].dt.tz_localize(None)
            rebalance_req_from_pinpoint = build_rebalance_req_df_from_pinpoint(
                pinpoint_df,
                ref_time
            )
            num_tours, total_length = tour_len_from_rebalance_req(
                rebalance_req_from_pinpoint,
                truck_od,
                depot_location,
                truck_size,
                earliest_departure,
                unload_time,
                max_wait_time
            )
            results.append(['observed', sim_date, num_tours, total_length])
    df = pd.DataFrame(
        results,
        columns=['mode', 'sim_date', 'num_tours', 'total_length'],
        index = [x for x in range(len(results))]
    )
    return df

def get_tour_details_over_range() -> pd.DataFrame:
    start = perf_counter()
    start_date = datetime(2021, 1, 1, 0, 0, 0, 0)
    end_date = datetime(2022, 1, 1, 0, 0, 0, 0)
    depot_location = 7681
    uoft = 'data/'
    trips_file = f'{uoft}Combined Bike Trips/combined_bike_trip_data.h5'
    hdf_file = f'{uoft}Bike Share Station Data/stn_snapshots.h5'
    trips_file = f'{uoft}Combined Bike Trips/combined_bike_trip_data.h5'
    walk_od_file = f'{uoft}walk_on_network_dist_od.csv'
    bike_od_file = f'{uoft}bike_tt_min.csv'
    truck_tt_file = f'{uoft}truck_od_tt_super.csv'
    truck_od = pd.read_csv(truck_tt_file)
    timestamp_series = pd.read_hdf(hdf_file, key='timestamps')
    all_dates = [x for x in daterange(start_date, end_date)]
    all_pinpoints = [f'{uoft}results/rebalancing_detection/{x.year}_{x.month}_{x.day}_pinpoint_rebalancing.csv' for x in all_dates]
    args = zip(
        all_dates, 
        repeat(depot_location),
        all_pinpoints,
        repeat(hdf_file),
        repeat(timestamp_series),
        repeat(trips_file),
        repeat(bike_od_file),
        repeat(walk_od_file),
        repeat(truck_od)
        )
    all_results = []
    for arg in tqdm(args):
        try:
            all_results.append(get_truck_tour_details(*arg))
        except:
            print(f'Failed on {arg[0]}')
    combined_results = pd.concat(all_results, ignore_index=True)
    output_path = f'{uoft}results/tours/2021_tour_results.csv'
    combined_results.to_csv(output_path, index=False)
    print(f'Completed tour operation in {perf_counter() - start}s')
    return combined_results

def main():

    print('Running Simulation')
    # Define the paths to the imput and output files
    uoft = 'data/'
    trips_file = f'{uoft}Combined Bike Trips/combined_bike_trip_data.h5'
    hdf_file = f'{uoft}Bike Share Station Data/stn_snapshots.h5'
    trips_file = f'{uoft}Combined Bike Trips/combined_bike_trip_data.h5'
    walk_od_file = f'{uoft}walk_on_network_dist_od.csv'
    bike_od_file = f'{uoft}bike_tt_min.csv'
    truck_tt_file = f'{uoft}truck_od_tt.csv'

    timestamp_series = pd.read_hdf(hdf_file, key='timestamps')
    timezone = 'America/Toronto'
    start_date = datetime(2021, 1, 1, 0, 0, 0, 0)
    end_date = datetime(2022, 1, 1, 0, 0, 0, 0)
    depot_location = 7681
    # lists to store the results before combining into dataframes
    sim_types = ['no_reb', 'reb_from_cap', 'reb_observed']
    result_types = [
        'results', 
        'capacity_details', 
        'failed_trips', 
        'full_delays', 
        'empty_delays',
        'dead_bike_removal'
    ]
    results_no_reb = [[] for x in range(6)]
    results_no_cap = [[] for x in range(6)]
    results_observed = [[] for x in range(6)]
    all_results = [
        results_no_reb,
        results_no_cap,
        results_observed
    ]
    print(f'Performing analysis for {(end_date - start_date).days} days')
    for sim_date in tqdm(daterange(start_date, end_date)):
        y = sim_date.year
        m = sim_date.month
        d = sim_date.day
        pinpoint_path = f'{uoft}results/rebalancing_detection/{y}_{m}_{d}_pinpoint_rebalancing.csv'
        try:
            results = triple_simulation(
                sim_date,
                depot_location,
                pinpoint_path,
                hdf_file,
                timestamp_series,
                trips_file,
                bike_od_file,
                walk_od_file
            )
            # append the results to the appropirate lists
            for r, sim_type, res_list in zip(results, sim_types, all_results):
                if r is not None:
                    if r[0] is not None:
                        result_df = r[0]
                        result_df['sim_date'] = sim_date
                        result_df['sim_type'] = sim_type
                        res_list[0].append(result_df)
                    if r[1] is not None:
                        cap_df = r[1]
                        cap_df['sim_date'] = sim_date
                        cap_df['sim_type'] = sim_type
                        res_list[1].append(cap_df)
                    if len(r[2]) > 0:
                        failed_trips = pd.DataFrame(r[2], columns=['start', 'end'], index=[x for x in range(len(r[2]))])
                        failed_trips['sim_date'] = sim_date
                        failed_trips['sim_type'] = sim_type
                        res_list[2].append(failed_trips)
                    if len(r[3]) > 0:
                        full_details = pd.DataFrame(r[3], columns=['full_delay_mins'], index=[x for x in range(len(r[3]))])
                        full_details['sim_date'] = sim_date
                        full_details['sim_type'] = sim_type
                        res_list[3].append(full_details)
                    if len(r[4]) > 0:
                        empty_details = pd.DataFrame(r[4], columns=['empty_delay_mins'], index=[x for x in range(len(r[4]))])
                        empty_details['sim_date'] = sim_date
                        empty_details['sim_type'] = sim_type
                        res_list[4].append(empty_details)
                    if r[5] is not None:
                        dead_bike_df = r[5]
                        dead_bike_df['sim_date'] = sim_date
                        dead_bike_df['sim_type'] = sim_type
                        res_list[5].append(dead_bike_df)
        except:
            print(f'Error in triple simulation for {sim_date}')

    # Combine the results into each of the output files

    # Combine the results into dataframes and output the dataframes to csv
    d_form = "%Y%m%d"
    start_str = start_date.strftime(d_form)
    end_str = end_date.strftime(d_form)
    output_prefix = f'{uoft}results/triple_sim/{start_str}_{end_str}'
    for r, sim_type in zip(all_results, sim_types):
        for l, result_type in zip(r, result_types):
            if len(l) > 0:
                df = pd.concat(l, ignore_index=True)
                output = f'{output_prefix}_{sim_type}_{result_type}.csv'
                df.to_csv(output, index=False)
            

    print('Simulations complete')
    print('Generating Rebalance req lists')
    # Define the paths to the imput and output files
    output_dir = f'{uoft}results/rebalance_reqs/'
    d_form = "%Y%m%d"
    for sim_date in tqdm(daterange(start_date, end_date)):
        try:
            df = build_rebaace_req_df_from_results(
                sim_date,
                depot_location,
                hdf_file,
                timestamp_series,
                trips_file,
                bike_od_file,
                walk_od_file
            )
            date_str = sim_date.strftime(d_form)
            df.to_csv(f'{output_dir}rebalance_req_optimized_{date_str}.csv')
        except:
            print(f'Failed on {sim_date}')  
    print('Rebalance reqs complete') 

if __name__ == '__main__':
    # main()
    get_tour_details_over_range()