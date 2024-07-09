from world import World, build_actuals_array_from_hdf, world_from_hdf, world_from_json, build_actuals_array_from_zip
import trips
from datetime import datetime, timedelta, date
import numpy as np
import pandas as pd
import time
from simulation import get_closest_world_to_t, find_warmup, get_trips_between, warmup_world
from tqdm import tqdm

def detect_rebalancing_sim(
    year,
    month,
    day,
    tol,
    snapshots_filepath: str,
    timestamp_series: pd.Series,
    trips_filepath: str,
) -> pd.DataFrame:
    start_time = datetime(year, month, day, 0, 0, 0, 0)
    end_time = start_time + timedelta(1, 0, 0, 0, 0)
    output_interval = 60
    # Get the simulation world as close to the start of the day as possible
    simulation_world = get_closest_world_to_t(
        start_time,
        3*60*60,
        timestamp_series,
        snapshots_filepath
    )
    if simulation_world is None:
        print(f'{start_time} No simulation world found')
        return None
    # Find a rebase world at least 2 hours before the simulation world
    warmup, steps = find_warmup(
        simulation_world, 
        120,
        480,
        timestamp_series,
        snapshots_filepath
    )
    if warmup is None:
        print(f'{start_time} No warmup world found')
        return None
    trips_df = get_trips_between(
        warmup.current_time,
        simulation_world.current_time,
        trips_filepath
    )
    # Perform the warmup
    simulation_world = warmup_world(
        warmup,
        simulation_world, 
        trips_df,
        steps
    )
    # reset the results counter
    simulation_world.reset_results_counters()
    simulation_trip_df = get_trips_between(
        simulation_world.current_time,
        end_time,
        trips_filepath
    )
    max_trip_length = 120
    simulation_trip_list = trips.read_df_trip_list(
        simulation_trip_df,
        simulation_world.current_time,
        max_trip_length
    )
    mins_until_end = int((end_time - simulation_world.current_time).total_seconds() / 60)
    results = simulation_world.step_through_time(
        simulation_trip_list,
        mins_until_end,
        output_interval,
        False
    )
    simulated_results = results[0]
    station_ids = results[1]
    timestamps = results[2]
    # get the actuals array to compare to 
    actuals_array = build_actuals_array_from_hdf(
        station_ids,
        timestamps,
        tol,
        timestamp_series,
        snapshots_filepath
    )
    # Take the difference between the actuals and simulated results
    diff = actuals_array - simulated_results
    dates_col = pd.to_datetime(timestamps, unit='s',utc=True)
    dates_col = dates_col.tz_convert('America/Toronto')
    output = pd.DataFrame(
        diff,
        columns = station_ids,
        index = dates_col
    )
    return output


def pinpoint_rebalancing(
    daily_rebalancing_df: pd.DataFrame,
    threshold: int,
    min_valid_rows: int
) -> pd.DataFrame:
    """
    This method takes the output from the detect rebalancing and identifies the
    timestamps and amounts of the observed rebalancing actions
    """
    # Check that there are enough results for a valid day
    valid_entries = daily_rebalancing_df[daily_rebalancing_df.columns[0]].count()
    if valid_entries <= min_valid_rows:
        print(f'Insufficient rows in rebalancing detection on {daily_rebalancing_df.index[0]}')
        return None
    cols = ['station_id', 'rebalance_time', 'amount']
    all_values = daily_rebalancing_df.to_numpy()
    jumps = np.zeros_like(all_values)
    jumps[1:] = all_values[1:] - all_values[:-1]
    jumps_pos = np.where(np.absolute(jumps) >= threshold)
    rebalancing_index = daily_rebalancing_df.index
    rebalancing_cols = daily_rebalancing_df.columns
    list_of_detections = []
    for i, j in zip(jumps_pos[0], jumps_pos[1]):
        detection_time = rebalancing_index[i]
        stn_id = rebalancing_cols[j]
        amount = jumps[i][j]
        list_of_detections.append([stn_id, detection_time, amount])
    output = pd.DataFrame(
        list_of_detections,
        index = [x for x in range(len(list_of_detections))],
        columns = cols
    )
    return output

def detect_rebalancing(year, month, day, trip_list, tol, mode, 
                       actuals_map_df = None, 
                       timestamp_series = None, snapshots_hdf = None):
    output = None
    max_trip_length = 120
    output_interval = 60
    failed_trips = []
    print(f"Detect rebalancing for {year}/{month}/{day}")
    # Create the rebase world
    rebase_world = None
    rebase_hour = 0
    # Try to find a rebase world between the hours of 0:00 and 5:00
    for h in range(6):
        if rebase_world is None:
            rebase_hour = h
            rebase_dt = datetime(year, month, day, rebase_hour,0,0,0)
            rebase_timestamp = datetime.timestamp(rebase_dt)
            if mode == 'json':
                rebase_world = world_from_json(rebase_timestamp, 
                                               tol, 
                                               actuals_map_df)
            elif mode == 'hdf':
                rebase_world = world_from_hdf(rebase_timestamp, 
                                              tol, 
                                              timestamp_series, 
                                              snapshots_hdf)
            else:
                print('Invalid mode given')
    # Create the test world
    # Test world must be at least 2hrs before the rebase world
    test_world = None
    warmup_steps = 120
    for i in range(120, 480, 10):
        if test_world is None:
            warmup_steps = i
            warmup_delta = timedelta(0, 0, 0, 0, -1*warmup_steps)
            test_dt = rebase_dt + warmup_delta
            timestamp = datetime.timestamp(test_dt)
            if mode == 'json':
                test_world = world_from_json(timestamp, 
                                             tol, 
                                             actuals_map_df)
            elif mode == 'hdf':
                test_world = world_from_hdf(timestamp, 
                                            tol, 
                                            timestamp_series, 
                                            snapshots_hdf)
            else:
                print('Invalid mode given')
    if rebase_world is not None and test_world is not None:
        # Simulate the test world until it matches the time of the rebase world
        # This is to warm up the system with trips
        warmup_trip_list = trips.read_df_trip_list(trip_list, test_dt, max_trip_length)
        warmup_results = test_world.step_through_time(warmup_trip_list, warmup_steps, output_interval, False)
        failed_trips = failed_trips + warmup_results[3]
        # Rebase the world with the actual occupancy
        test_world.rebase_world(rebase_world)
        # Apply the remaining trips from rebase to end of day
        remaining_trip_list = trips.read_df_trip_list(trip_list, rebase_dt, max_trip_length)
        remaining_timesteps = (24-rebase_hour)*60
        sim_results = test_world.step_through_time(remaining_trip_list, 
                                                   remaining_timesteps, 
                                                   output_interval, 
                                                   False)
        simulated_results, station_ids, timestamps, trip_fails, cap_details = sim_results
        failed_trips = failed_trips + trip_fails
        # Get the array for the actuals
        if mode == 'json':
            actuals_array = build_actuals_array_from_zip(station_ids, 
                                                         timestamps, 
                                                         tol, 
                                                         actuals_map_df)
        elif mode == 'hdf':
            actuals_array = build_actuals_array_from_hdf(station_ids, 
                                                         timestamps,
                                                         tol,
                                                         timestamp_series,
                                                         snapshots_hdf)
        else:
            print('Invalid mode given')
        # Take the difference
        diff = actuals_array - simulated_results
        # Output the results to CSV
        zone = 'America/Toronto'
        # dates_col = pd.to_datetime(timestamps, unit='s',).tz_localize(tz=zone)
        dates_col = pd.to_datetime(timestamps, unit='s',)
        output = pd.DataFrame(diff, 
                              columns=station_ids, 
                              index=dates_col)
        print(f'Failed to complete {len(failed_trips)} trips')
    else:
        print(f"Could not find snapshots to check {year}/{month}/{day} for rebalancing")
    return output

def summarize_daily_rebalancing(daily_rebalancing_df, threshold):
    # get the max value from each column
    # get the min value from each column
    # if max value >=threshold positive rebalance
    # if min value <=-threshold negative rebalance
    pos_threshold = threshold
    neg_threshold = -1*threshold
    positive = daily_rebalancing_df.max(axis=0)
    positive = positive.apply(lambda x: x>=pos_threshold)
    negative = daily_rebalancing_df.min(axis=0)
    negative = negative.apply(lambda x: x<=neg_threshold)
    return positive, negative

def daterange(start_date, end_date):
    """
    Generator function to get dates between start and end
    """
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

def summarize_rebalancing_over_range(start_date, 
                                     end_date, 
                                     trips_hdf, 
                                     tol, 
                                     mode, 
                                     actuals_map_df = None, 
                                     timestamp_series = None, 
                                     snapshots_hdf = None):
    pos_rebalancings = []
    neg_rebalancings = []
    dates = []
    for test_d in daterange(start_date, end_date):
        # dates.append(test_d)
        # Get the year month and day from the date
        y = test_d.year
        m = test_d.month
        d = test_d.day
        # Load relevant trips
        d_form = "%Y%m%d"
        buffer_day = timedelta(1,0,0,0,0)
        start_str = (test_d - buffer_day).strftime(d_form)
        end_str = (test_d + buffer_day).strftime(d_form)
        trip_dates = f'index >= {start_str} & index <= {end_str}'
        trip_list = pd.read_hdf(trips_hdf, key='trips', where=[trip_dates])
        trip_list = trip_list.reset_index()
        # detect rebalancing for the date
        rebalance = detect_rebalancing(y, m, d, trip_list, tol, mode, 
                                       actuals_map_df, 
                                       timestamp_series, snapshots_hdf)
        if rebalance is not None:
            pos, neg = summarize_daily_rebalancing(rebalance, threshold=5)
            pos_rebalancings.append(pos)
            neg_rebalancings.append(neg)
            dates.append(test_d)
    output_pos = None
    output_neg = None
    # Check if the list isn't empty
    if pos_rebalancings:
        output_pos = pd.concat(pos_rebalancings, 
                               axis=1,
                               ignore_index=True)
        output_pos = output_pos.transpose()
        output_pos['dates'] = dates
        output_pos = output_pos.set_index('dates')
    # Check if the list isn't empty
    if neg_rebalancings:
        output_neg = pd.concat(neg_rebalancings, 
                               axis=1,
                               ignore_index=True)
        output_neg = output_neg.transpose()
        output_neg['dates'] = dates
        output_neg = output_neg.set_index('dates')
    return output_pos, output_neg

def main():
    uoft = 'data/'
    trips_file = f'{uoft}Combined Bike Trips/combined_bike_trip_data.h5'
    hdf_file = f'{uoft}Bike Share Station Data/stn_snapshots.h5'
    trips_file = f'{uoft}Combined Bike Trips/combined_bike_trip_data.h5'
    walk_od_file = f'{uoft}walk_on_network_dist_od.csv'
    bike_od_file = f'{uoft}bike_tt_min.csv'
    output_dir = f'{uoft}results/rebalancing_detection/'
    tol = 10*60
    mode = 'hdf'
    y = 2021
    m = 8
    d = 23

    timestamp_series = pd.read_hdf(hdf_file, key='timestamps')

    start_date = datetime(2021, 1, 1, 0, 0, 0, 0)
    end_date = datetime(2022, 1, 1, 0, 0, 0, 0)
    print(f'Detecting rebalancing for {(end_date - start_date).days} days')
    for sim_date in tqdm(daterange(start_date, end_date)):
        y = sim_date.year
        m = sim_date.month
        d = sim_date.day
        results = detect_rebalancing_sim(
            y,
            m,
            d,
            tol,
            hdf_file,
            timestamp_series,
            trips_file
        )
        if results is not None:
            # Save the result to csv
            output_path = f'{output_dir}{y}_{m}_{d}_detect_rebalancing.csv'
            results.to_csv(output_path)
            # Pinpoit the rebalancings
            threshold = 3
            min_valid_rows = 20
            pinpoint_df = pinpoint_rebalancing(results, threshold, min_valid_rows)
            if pinpoint_df is not None:
                # save the results to csv
                output_path = f'{output_dir}{y}_{m}_{d}_pinpoint_rebalancing.csv'
                pinpoint_df.to_csv(output_path, index=False)


if __name__ == '__main__':
    main()