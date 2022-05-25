import pandas as pd
import numpy as np
import datetime
import warnings
import os
import statsmodels.api as sm
from sqlalchemy import create_engine

#######################
# Utility functions   #
#######################

def get_wd_w_buffer(start_date, end_date, engine):
    new_start_date = start_date - datetime.timedelta(days = 7)
    
    try:
        new_data = pd.read_sql_query(f"SELECT * FROM sensor_water_depth WHERE date >= '{new_start_date}' AND date <= '{end_date}'", engine).sort_values(['place','date']).drop_duplicates()
    except:
        new_data = pd.DataFrame()
        warnings.warn("Connection to database failed to return data")
    
    if new_data.shape[0] == 0:
        warnings.warn("No new data to during requested time period!")
        pass
    
    return new_data

def get_surveys(engine):
    try:
        surveys = pd.read_sql_table("sensor_surveys", engine).sort_values(['place','date_surveyed']).drop_duplicates()
    except:
        surveys = pd.DataFrame()
        warnings.warn("Connection to database failed to return data")
        
    if surveys.shape[0] == 0:
        warnings.warn("- No survey data!")
        return
    
    return surveys


def qa_qc_flag(x, delta_wd_per_minute = 0.1):
    
    x["lag_sensor_water_depth"] = x["sensor_water_depth"] - x.groupby(by="sensor_ID")["sensor_water_depth"].shift(1)
    x["lag_duration_minutes"] = (x["date"] - x.groupby(by="sensor_ID")["date"].shift(1)).dt.total_seconds() / 60
    x["lag_wd_per_minute"] = x["lag_sensor_water_depth"]/x["lag_duration_minutes"]
    x["qa_qc_flag"] = np.where(np.abs(x["lag_wd_per_minute"]) > delta_wd_per_minute, True, False)
    
    x.drop(columns = ["lag_sensor_water_depth", "lag_duration_minutes", "lag_wd_per_minute"], inplace = True)
    
    return x


def match_measurements_to_survey(measurements, surveys):
    sites = measurements["sensor_ID"].unique()
    survey_sites = surveys["sensor_ID"].unique()
    
    matching_sites = list(set(sites) & set(survey_sites))
    missing_sites = list(set(sites).difference(survey_sites))
    
    if len(missing_sites) > 0:
        warnings.warn(message = str("Missing survey data for: " + ''.join(missing_sites) + ". The site(s) will not be processed."))    
    
    matched_measurements = pd.DataFrame()
    
    for selected_site in matching_sites:
        selected_measurements = measurements.query("sensor_ID == @selected_site").copy()
        
        selected_survey = surveys.query("sensor_ID == @selected_site")
        
        if selected_survey.empty:
            warnings.warn("There are no survey data for: " + selected_site)
        
        survey_dates = list(selected_survey["date_surveyed"].unique())
        number_of_surveys = len(survey_dates)
        
        if measurements["date"].min() < min(survey_dates):
            warnings.warn("Warning: There are data that precede the survey dates for: " + selected_site)
            
        if number_of_surveys == 1:
            selected_measurements["date_surveyed"] = pd.to_datetime(np.where(selected_measurements["date"] >= survey_dates[0], survey_dates[0], np.nan))
            
        if number_of_surveys > 1:
            survey_dates.append(pd.to_datetime(datetime.datetime.utcnow(), utc=True))
            selected_measurements["date_surveyed"] = pd.to_datetime(pd.cut(selected_measurements["date"], bins = survey_dates, labels = survey_dates[:-1]), utc = True)
    
        merged_measurements_and_surveys = pd.merge(selected_measurements, surveys, how = "left", on = ["place","sensor_ID","date_surveyed"])
        
        matched_measurements = pd.concat([matched_measurements, merged_measurements_and_surveys]).drop_duplicates()
        matched_measurements["notes"] = matched_measurements["notes_x"]
        matched_measurements.drop(columns = ['notes_x','notes_y'],inplace=True)
        
    return matched_measurements


def calc_baseline_wl(x, surveys):
    sensor_list = list(x["sensor_ID"].unique())
    
    smoothed_baseline_wl = pd.DataFrame()

    for selected_sensor in sensor_list:
        selected_data = x.query("sensor_ID == @selected_sensor")
        selected_survey = surveys.query("sensor_ID == @selected_sensor")
        
        if selected_data.shape[0] == 0:
            warnings.warn(f"No data for sensor for baseline calculation for: {selected_sensor}")     
        
        if selected_survey.shape[0] == 0:
            warnings.warn(f"No survey data for: {selected_sensor}")
            
        merged_data = match_measurements_to_survey(measurements = selected_data, surveys = selected_survey)
        merged_data_w_smoothed_baseline_wl = smooth_baseline_wl(merged_data)
        
        smoothed_baseline_wl = pd.concat([smoothed_baseline_wl, merged_data_w_smoothed_baseline_wl])
            
    return smoothed_baseline_wl


def smooth_baseline_wl(x):
    survey_dates = list(x["date_surveyed"].unique())
    
    smoothed_baseline_wl = pd.DataFrame()
    
    for selected_survey in survey_dates:
        selected_data = x.query("date_surveyed == @selected_survey")
    
        rolling_min = selected_data.set_index("date")["sensor_water_depth"].rolling('2d').min().reset_index()
        rolling_min.rename(columns={'sensor_water_depth':'rolling_min_wd'}, inplace = True)
        rolling_min["lag_min_wd"] = rolling_min["rolling_min_wd"] - rolling_min["rolling_min_wd"].shift(1)
        rolling_min["lag_duration_minutes"] = (rolling_min["date"] - rolling_min["date"].shift(1)).dt.total_seconds() / 60
        rolling_min["lag_min_wd_per_minute"] = rolling_min["lag_min_wd"]/rolling_min["lag_duration_minutes"]
        rolling_min["change_pt"] = np.select(condlist=[rolling_min["lag_min_wd_per_minute"] != 0, rolling_min["date"] == rolling_min["date"].max(), rolling_min["lag_min_wd_per_minute"] == 0], choicelist= [True, True, False], default=False)
        
        lower_quantile = np.quantile(rolling_min["rolling_min_wd"], 0.01)
        upper_quantile = np.quantile(rolling_min["rolling_min_wd"], 0.75)
        
        change_pts = rolling_min.query("change_pt == True & rolling_min_wd >= @lower_quantile & rolling_min_wd <= @upper_quantile ").loc[:,["date","rolling_min_wd"]]        
        
        if change_pts.empty:
            merged_data_and_change_pts = selected_data
            merged_data_and_change_pts["smooth_min_wd"] = rolling_min["rolling_min_wd"]
                
        if change_pts.shape[0] < 3:
            merged_data_and_change_pts = pd.merge(selected_data, change_pts.rename(columns = {"rolling_min_wd":"smooth_min_wd"}), how="left").set_index("date")
            merged_data_and_change_pts["smooth_min_wd"] = merged_data_and_change_pts["smooth_min_wd"].interpolate(method="pad").interpolate(method="backfill")
            
        if change_pts.shape[0] >= 3:
            x = np.array(change_pts["date"].astype('int'))
            y = np.array(change_pts["rolling_min_wd"])
            b = np.array(change_pts)
            z = sm.nonparametric.lowess(y, x)
        
            smoothed_min_wl = pd.DataFrame(z).rename(columns={0:"date",1:"smooth_min_wd"})
            smoothed_min_wl["date"] = pd.to_datetime(smoothed_min_wl["date"], utc=True)
        
            merged_data_and_change_pts = pd.merge(selected_data, smoothed_min_wl, how="left").set_index("date")
            merged_data_and_change_pts["smooth_min_wd"] = merged_data_and_change_pts["smooth_min_wd"].interpolate(method="time", limit_direction="both")
            
        smoothed_baseline_wl = pd.concat([smoothed_baseline_wl, merged_data_and_change_pts])

    return smoothed_baseline_wl

def correct_drift(x, start_date, end_date):
    data = x.copy().reset_index()
    
    data["sensor_water_level"] = data["sensor_elevation"] + data["sensor_water_depth"]
    data["road_water_level"] = data["sensor_water_level"] - data["road_elevation"]
    data["sensor_water_level_adj"] = data["sensor_water_level"] - data["smooth_min_wd"]
    data["road_water_level_adj"] = data["road_water_level"] - data["smooth_min_wd"]
    data["date"] = pd.to_datetime(data["date"])
    
    filtered_x = data[(data["date"] >= str(start_date)) & (data["date"] <= str(end_date))].copy()
    filtered_x.rename(columns={"atm_data_src_x":"atm_data_src", "atm_station_id_x":"atm_station_id","smooth_min_wd":"smoothed_min_water_depth"}, inplace=True)
    filtered_x["min_water_depth"] = np.nan; filtered_x["deriv"] = np.nan; filtered_x["change_pt"] = np.nan

    filtered_x = filtered_x.loc[:,["place", "sensor_ID", "date", "voltage", "sensor_water_depth", "qa_qc_flag", "date_surveyed", "sensor_elevation", "road_elevation", "lat", "lng", "alert_threshold", "min_water_depth", "deriv", "change_pt", "smoothed_min_water_depth", "sensor_water_level", "road_water_level", "sensor_water_level_adj", "road_water_level_adj"]]

    return filtered_x.set_index(["place", "sensor_ID", "date"])


def postgres_upsert(table, conn, keys, data_iter):
    from sqlalchemy.dialects.postgresql import insert

    data = [dict(zip(keys, row)) for row in data_iter]

    insert_statement = insert(table.table).values(data)
    upsert_statement = insert_statement.on_conflict_do_update(
        constraint=f"{table.table.name}_pkey",
        set_={c.key: c for c in insert_statement.excluded},
    )
    conn.execute(upsert_statement)
    

def main():

    ########################
    # Establish DB engine  #
    ########################

    SQLALCHEMY_DATABASE_URL = "postgresql://" + os.environ.get('POSTGRESQL_USER') + ":" + os.environ.get(
        'POSTGRESQL_PASSWORD') + "@" + os.environ.get('POSTGRESQL_HOSTNAME') + "/" + os.environ.get('POSTGRESQL_DATABASE')

    engine = create_engine(SQLALCHEMY_DATABASE_URL)

    #####################
    # Process data  #
    #####################

    end_date = pd.to_datetime(datetime.datetime.utcnow())
    start_date = end_date - datetime.timedelta(days=7)

    new_data = get_wd_w_buffer(start_date, end_date, engine)
    surveys = get_surveys(engine)

    qa_qcd_df = qa_qc_flag(new_data).query("qa_qc_flag == False")
    smoothed_min_wl_df = calc_baseline_wl(qa_qcd_df, surveys)
    drift_corrected_df = correct_drift(smoothed_min_wl_df, start_date, end_date)

    try:
        drift_corrected_df.to_sql("data_for_display", engine, if_exists = "append", method=postgres_upsert, chunksize = 3000)
        print("Drift-corrected data written to database!")
    except:
        warnings.warn("Error writing drift-corrected data to database")
    
    engine.dispose()

if __name__ == "__main__":
    main()