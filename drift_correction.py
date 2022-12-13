from select import select
import pandas as pd
import numpy as np
import datetime
import warnings
import os
import json
import gspread
import statsmodels.api as sm
from sqlalchemy import create_engine
import pytz
import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

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

def get_drift_corrected_data(start_date, end_date, engine):
    try:
        new_data = pd.read_sql_query(f"SELECT * FROM data_for_display WHERE date >= '{start_date}' AND date <= '{end_date}'", engine).sort_values(['place','date']).drop_duplicates()
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

def get_flood_status(engine):
    try:
        flood_status = pd.read_sql_table("flood_status", engine).sort_values(['place','sensor_ID'])
    except:
        flood_status = pd.DataFrame()
        warnings.warn("Connection to database failed to return data")
        
    if flood_status.shape[0] == 0:
        warnings.warn("- No flood status data!")
        return
    
    return flood_status


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
        # print(selected_sensor)
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
        # print(selected_survey)
        selected_data = x.query("date_surveyed == @selected_survey").copy()
    
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
            x1 = np.array(change_pts["date"].astype('int'))
            y = np.array(change_pts["rolling_min_wd"])
            b = np.array(change_pts)
            z = sm.nonparametric.lowess(y, x1)
        
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
    
    
def detect_flooding(x):
    latest_measurements = x.sort_values(["sensor_ID", "date"]).groupby("sensor_ID").tail(3).reset_index()
    latest_measurements["sample_interval"] = latest_measurements["date"] - latest_measurements.groupby(by="sensor_ID")["date"].shift(1)
    latest_measurements["min_interval"] = latest_measurements.groupby("sensor_ID")["sample_interval"].transform('min')
    
    current_time = pd.to_datetime(datetime.datetime.utcnow(), utc=True)
    
    
    last_measurement = latest_measurements.sort_values("date").groupby("sensor_ID").tail(1)
    last_measurement["above_alert_wl"] = last_measurement["sensor_water_level_adj"] >= last_measurement["alert_threshold"]
    last_measurement["time_since_measurement"] = current_time - last_measurement["date"]
    last_measurement["cutoff_time"] = datetime.timedelta(minutes=40)
    last_measurement["is_flooding"] = (last_measurement["time_since_measurement"] > last_measurement["cutoff_time"]) & last_measurement["above_alert_wl"]
    last_measurement["alert_sent"] = False
    last_measurement["current_time"] = current_time
    
    last_measurement.rename(columns = {"date":"latest_measurement"}, inplace = True)
    # last_measurement.set_index(["place","sensor_ID"], inplace=True)
    
    return last_measurement.loc[:,["place","sensor_ID", "latest_measurement","current_time","is_flooding","alert_sent"]]

def send_alert(place):
    
    list_id = os.environ.get("MAILCHIMP_LIST_ID")
    interest_category_id = os.environ.get("MAILCHIMP_INTEREST_ID")
    
    formatted_place = place.replace("North Carolina", "NC")
    
    # Get options of places for flood alerts
    try:
        client = MailchimpMarketing.Client()
        client.set_config({
            "api_key": os.environ.get("MAILCHIMP_KEY"),
            "server": "us20"
        })

        site_options = client.lists.list_interest_category_interests(list_id, interest_category_id)
        
    except ApiClientError as error:
        site_options = dict()
        print("Error: {}".format(error.text))
  
    site_options_df = pd.DataFrame.from_dict(site_options["interests"])
    interest_value_df = site_options_df.query("name == @formatted_place").copy()
    
    if interest_value_df.shape[0] == 0:
        return (formatted_place + " is not registered as an option for the listserv")
    
    # Get current time when flood was detected
    flood_time = datetime.datetime.now(pytz.timezone("US/Eastern")).strftime("%H:%M%p %Z on %m/%d/%Y")
    flood_date = datetime.datetime.now(pytz.timezone("US/Eastern")).strftime("%m/%d/%Y")
    
    # Create new campaign
    try:
        new_campaign = client.campaigns.create({"type": "plaintext", "recipients":{"segment_opts":{"match": "all","conditions":[{"condition_type": "Interests","field": ("interests-"+interest_category_id),"op": "interestcontains","value": list(interest_value_df.id.values)}]},"list_id": list_id},"tracking": {"opens": False,"text_clicks": False},"settings": {"subject_line": "Flood Alert - Sunny Day Flooding Project","preview_text": ("Flood alert for "+formatted_place),"title": (formatted_place +" Flood Alert - "+ flood_date),"from_name": "Sunny Day Flooding Project","reply_to": "sunnydayflood@gmail.com","use_conversation": True,"to_name": "*|FNAME|* *|LNAME|*","auto_footer": False}})                                        
    except:
        new_campaign = dict()
        warnings.warn("Failed to create campaign")
        return
    
    # Save new campaign's ID for updating/sending
    new_campaign_id = new_campaign.get("id")
    
    # Update the email content with appropriate info
    try:
        response = client.campaigns.set_content(new_campaign_id, {"plain_text":"Flood Alert for "+formatted_place+
                       "\n--------------------------------\n\nWater estimated on/near roadway at: "+ 
                       flood_time+
                       ".\n\nVisit our data viewer to see live data and pictures of the site:\nhttps://go.unc.edu/flood-data\n\nThis alert is informed by preliminary data and is for INFORMATIONAL PURPOSES ONLY. Please refer to your local National Weather Service station for actionable flooding info: https://water.weather.gov/ahps/region.php?state=nc  \n\n================================\nYou are receiving this email because you opted in via our website: https://tarheels.live/sunnydayflood\n\nUnsubscribe *|HTML:EMAIL|* from this list: *|UNSUB|*\n\nUpdate Profile: *|UPDATE_PROFILE|*\n\nOur mailing address is:\nSunny Day Flooding Project\n223 E Cameron Ave\nNew East Building, CB#3140\nChapel Hill, NC 27599-3140\nUSA"})
    except ApiClientError as error:
        print("Error: {}".format(error.text))
    
    # Send the new campaign!
    try:
        response = client.campaigns.send(new_campaign_id)
        print("Alert successfully sent for: ", formatted_place)
    except:
        warnings.warn("Error sending alert for: "+ formatted_place)
        
        
def alert_flooding(x, engine):
    # was it flooding
    flood_status_df = get_flood_status(engine).query("alerts_on == True").copy()
    
    active_alert_sites = list(flood_status_df.sensor_ID)
    
    # is it flooding now
    is_flooding_df = detect_flooding(x).query("sensor_ID in @active_alert_sites").copy()
    
    places = list(is_flooding_df["place"].unique())
    
    for selected_place in places:
        site_data = is_flooding_df.query("place == @selected_place").copy()
        flood_status_site = flood_status_df.query("place == @selected_place").copy()
        
        any_flooding = site_data["is_flooding"].sum() > 0
        
        if any_flooding:
            site_flooding_data = site_data.query("is_flooding == True").copy()
            alert_already_sent = (flood_status_site["alert_sent"].sum() > 0)
            
            if alert_already_sent:
                site_flooding_data["alert_sent"] = True
                print("Flooding detected, but alert previously sent for:" , selected_place)
                    
                try:
                    site_flooding_data.set_index(["place","sensor_ID"]).to_sql("flood_status", engine, if_exists = "append", method=postgres_upsert)
                    print("Flood status data written to database for:", selected_place)
                except:
                    warnings.warn("Error writing flood status data to database")
                    
                
            elif not alert_already_sent:
                send_alert(selected_place)
                
                site_flooding_data["alert_sent"] = True
                
                try:
                    site_flooding_data.set_index(["place","sensor_ID"]).to_sql("flood_status", engine, if_exists = "append", method=postgres_upsert)
                    print("Flood status data written to database for:", selected_place)
                except:
                    warnings.warn("Error writing flood status data to database")
            
            else:
                warnings.warn("Error determining if flood alert has been sent") 
                    
        else:
            try:
                site_data.set_index(["place","sensor_ID"]).to_sql("flood_status", engine, if_exists = "append", method=postgres_upsert)
                print("No flood alert sent for:", selected_place)
            except:
                warnings.warn("Error writing flood status data to database")
            
    return


def update_tracking_spreadsheet(data, flood_cutoff = 0):
    x=data.copy()
    
    current_time = pd.Timestamp('now', tz= "UTC") + pd.offsets.Hour(-4)
    
    flooding_measurements = x.reset_index().query("road_water_level_adj > @flood_cutoff").copy()
    
    n_flooding_measurements = flooding_measurements.shape[0]
  
    if(n_flooding_measurements == 0):
        return "No flooding to update spreadsheet"
    
    flooding_measurements = flooding_measurements.reset_index()
    flooding_measurements["min_date"] = flooding_measurements.date - datetime.timedelta(minutes = 1)
    flooding_measurements["max_date"] = flooding_measurements.date + datetime.timedelta(minutes = 1)
    
    flooding_measurements = flooding_measurements[["place", "sensor_ID", "date", "road_water_level_adj", "road_water_level", "voltage", "min_date", "max_date"]]
 
    # Download existing flood events from Google Sheets
    json_secret = json.loads(os.environ.get('GOOGLE_JSON_KEY'))
    google_sheet_id = os.environ.get('GOOGLE_SHEET_ID')
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict=json_secret, scopes=scope)

    gc = gspread.service_account_from_dict(json_secret)
    sh = gc.open_by_key(google_sheet_id)
    worksheet = sh.get_worksheet(0)
        
    sheet_data_df = pd.DataFrame(worksheet.get_all_records())

    min_dates = sheet_data_df.groupby(["place", "sensor_ID", "flood_event"])[["date"]].min() 
    max_dates = sheet_data_df.groupby(["place", "sensor_ID", "flood_event"])[["date"]].max()

    flood_start_stop = pd.merge(min_dates, max_dates, on = ["place", "sensor_ID","flood_event"])
    flood_start_stop["min_date"] = pd.to_datetime(flood_start_stop.date_x, utc=True) - datetime.timedelta(minutes = 30)
    flood_start_stop["max_date"] = pd.to_datetime(flood_start_stop.date_y, utc=True) + datetime.timedelta(minutes = 30)
    
    # Iterate through each place, compare overlap of each flood event in our new data and the existing data in the spreadsheet
    # If there is no overlap, collect the flood event data to then write to spreadsheet
    places = list(flooding_measurements["place"].unique())
    
    new_site_data_df = pd.DataFrame()
    
    for selected_place in places:
        print(selected_place)
        site_data = flooding_measurements.query("place == @selected_place").copy()
        site_existing_data = flood_start_stop.query("place == @selected_place").copy().reset_index()
        
        last_flood_number = pd.to_numeric(site_existing_data.flood_event).max()
        site_data["flood_event"] = flood_counter(site_data.date, start_number = 0, lag_hrs = 2)
        
        flood_events_occuring = site_data.groupby("flood_event").max_date.max() > current_time
        flood_events_occuring = flood_events_occuring.reset_index()
        flood_events_to_select = flood_events_occuring[flood_events_occuring.max_date == False].flood_event.tolist()
        
        site_data = site_data.query("flood_event in @flood_events_to_select")

        site_min_dates = site_data.groupby(["flood_event"])[["date"]].min() 
        site_max_dates = site_data.groupby(["flood_event"])[["date"]].max() 
        site_flood_start_stop = pd.merge(site_min_dates, site_max_dates, on = ["flood_event"])
        site_flood_start_stop["min_date"] = pd.to_datetime(site_flood_start_stop.date_x) 
        site_flood_start_stop["max_date"] = pd.to_datetime(site_flood_start_stop.date_y)
        
        site_keep_list = list()
        
        for (i, v) in site_flood_start_stop.iterrows():
            # need to collect values to keep to track when there is overlap. keep = not overlap
            internal_overlap_list = list()
            
            # create an interval using the new data (from the row being iterated on)
            new_interval = pd.Interval(v.min_date, v.max_date)
            
            # for each row in the existing data, check for overlap with our new data
            for (existing_i, existing_v) in site_existing_data.iterrows():
                
                existing_interval = pd.Interval(existing_v.min_date, existing_v.max_date)
                overlaps = new_interval.overlaps(existing_interval)
                internal_overlap_list.append(overlaps)
            
            if sum(internal_overlap_list) > 0:
                site_keep = False
            else:
                site_keep = True
            
            site_keep_list.append(site_keep)
            
        if sum(site_keep_list) == 0:
            # pass
            print("No new flood events")
            return 
        
        new_flood_events = site_flood_start_stop[site_keep_list].reset_index()
        
        new_site_data = site_data.query("flood_event in @new_flood_events.flood_event")
        new_site_data.flood_event = flood_counter(new_site_data.date, start_number = last_flood_number, lag_hrs = 2)
        new_site_data["drift"] = new_site_data.road_water_level - new_site_data.road_water_level_adj
        new_site_data = new_site_data.loc[:,['place','sensor_ID','flood_event', 'date', 'road_water_level_adj', 'road_water_level', 'drift', 'voltage']]
        new_site_data_df = pd.concat([new_site_data_df,new_site_data])
    
    # Get pictures that align
    new_site_data_df_w_pics = get_pictures_for_flooding(new_site_data_df)
    
    # Convert full df of new flood events to string so we can write them to a google spreadsheet
    new_site_data_df_w_pics = new_site_data_df_w_pics.astype('str')
    
    # Append new values the google sheet!
    try:
        write_to_sheet = worksheet.append_rows(values = new_site_data_df_w_pics.values.tolist(), value_input_option="USER_ENTERED")
        print("Wrote new flood events to spreadsheet")
    except:
        print("Whoops! An error writing flood events to spreadsheet")
        
    return

def get_pictures_for_flooding(data):
    json_secret = json.loads(os.environ.get('GOOGLE_JSON_KEY'))
    google_drive_folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER_ID')
    scope = ["https://www.googleapis.com/auth/drive"]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict=json_secret, scopes=scope)

    drive = build('drive', 'v3', credentials=credentials)
    
    images_folder_id = os.environ.get('GOOGLE_IMAGES_ID')
    
    x = data.copy()
    sensor_ids = x["sensor_ID"].unique().tolist()
    x_cols = x.columns.tolist()
    x_cols.append("pic_links")
    
    x["day"] = x.date.dt.strftime("%Y-%m-%d")
    
    rows_with_pics = pd.DataFrame()
    
    for selected_sensor_id in sensor_ids:
        selected_sensor_data = x.query("sensor_ID == @selected_sensor_id").copy()
        days_of_flooding = selected_sensor_data.day.unique().tolist()
        
        # Search for the camera's folder within
        camera_image_folder_info = drive.files().list(
            corpora="drive",
            driveId=google_drive_folder_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            q="name='CAM_" + selected_sensor_id + "' and mimeType='application/vnd.google-apps.folder' and '" + images_folder_id + "' in parents and trashed = false"
        ).execute().get('files', [])
        
        if len(camera_image_folder_info) > 0:
            camera_image_folder_id = camera_image_folder_info[0].get('id')
            
            for day in days_of_flooding:
                
                selected_day_data = selected_sensor_data.query("day == @day").copy()
                
                # Within the camera's folder, see if there is a folder for the specific date of interest (date_label)
                date_folder_info = drive.files().list(
                    corpora="drive",
                    driveId=google_drive_folder_id,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    q="'" + camera_image_folder_id + "'" + " in parents and trashed = false and name='" + day + "' and mimeType='application/vnd.google-apps.folder'"
                ).execute().get('files', [])

                # If there is a folder for the date within the camera's folder, get the ID
                if len(date_folder_info) > 0:
                    date_folder_id = date_folder_info[0].get('id')
                    
                    picture_info = pd.DataFrame(drive.files().list(
                        corpora="drive",
                        driveId=google_drive_folder_id,
                        includeItemsFromAllDrives=True,
                        supportsAllDrives=True,
                        q="'" + date_folder_id + "'" + " in parents and trashed = false"
                    ).execute().get('files', []))
                    
                    picture_info["time"] = picture_info.name.apply(lambda x: pd.Interval(pd.to_datetime(x.split("_")[-1].split(".")[0], utc=True),pd.to_datetime(x.split("_")[-1].split(".")[0], utc=True)))
                    selected_day_data["interval"] = selected_day_data.date.apply(lambda x: pd.Interval(x - datetime.timedelta(minutes=5), x + datetime.timedelta(minutes=5)))
                    
                    selected_day_data["pic_links"] = pd.NA
                    
                    for (i, v) in selected_day_data.iterrows():
                        overlaps = picture_info.time.apply(lambda x: v.interval.overlaps(x))
                        
                        if sum(overlaps > 0):
                            selected_day_data.loc[i, "pic_links"] = "https://drive.google.com/open?id=" + str(picture_info.id[overlaps].values.tolist()[0]) 
                    
                rows_with_pics = pd.concat([rows_with_pics, selected_day_data])
        else:
            print("No camera folder for this site: CAM_" + selected_sensor_id)
        
    return x.merge(rows_with_pics.loc[:,["place","sensor_ID","date","pic_links"]], on = ["place","sensor_ID","date"], how="left").loc[:,x_cols]
        
    
def flood_counter(dates, start_number = 0, lag_hrs = 8):
    dates = dates.copy().reset_index().date
    lagged_time = dates - dates.shift(1)
    lagged_time = lagged_time.fillna(pd.Timedelta('0 days'))
    
    group_change_vector = list()
    
    for i,v in enumerate(dates):
        x = 0

        if abs(lagged_time[i]) > datetime.timedelta(hours = lag_hrs):
            x = 1
    
        group_change_vector.append(x)
    
    group_vector = np.cumsum(group_change_vector) + 1 + start_number
    
    return group_vector



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
    
    
    ###################
    #  Flood alerts  #
    ###################
   
    alert_flooding(x = drift_corrected_df, engine = engine)
    
    #######################################
    #  Update flood tracking spreadsheet  #
    #######################################
    
    update_tracking_spreadsheet(data = drift_corrected_df, flood_cutoff = 0)
    
    #############################
    # Cleanup the DB connection #
    #############################
    
    engine.dispose()
    
    # Create requirements.txt using this commange on local machine - "pip list --format=freeze > requirements.txt"

if __name__ == "__main__":
    main()