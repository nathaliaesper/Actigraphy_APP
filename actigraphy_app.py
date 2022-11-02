# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_daq as daq
import plotly.express as px
import pandas as pd
import rdata
import re
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import dash_bootstrap_components as dbc
import time
import json
import csv
import os
import sys
import base64
import calendar
import math
from pathlib import Path
from os import listdir
from os.path import exists, isfile, join
from datetime import date, datetime
from plotly.subplots import make_subplots
import argparse

#print(daq.__version__)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#config = {
#    "modeBarButtonsToAdd": [
#        "drawrect",
#    ]
#}

parser=argparse.ArgumentParser(
    description='''Actigraphy APP to manually correct annotations for the sleep log diary. ''',
    epilog="""APP developed by Child Mind Institute.""")
parser.add_argument('input_folder', help='GGIR output folder')
args=parser.parse_args()

#global input_datapath
input_datapath = sys.argv[1]
files = [f for f in listdir(input_datapath+'/meta/ms4.out') if f.endswith(".RData")]
files = sorted(files)

if (exists(os.path.join(input_datapath+'/logs'))):
    log_path = os.path.join(input_datapath+'/logs')
else:
    os.mkdir(os.path.join(input_datapath+'/logs'))
    log_path = os.path.join(input_datapath+'/logs')

#print(files)

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
            {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)

pd.Series.which = which

# Function to load the ms4.out file and get some useful variables
def load_ms4_file(filename):
    
    filepath = os.path.abspath(os.path.join(input_datapath + '/meta/ms4.out', filename))

    night_summary = rdata.parser.parse_file(filepath)
    night_summary_converted = rdata.conversion.convert(night_summary)
    
    nights = night_summary_converted.get("nightsummary").night
    num_nights = np.size(nights)
    sleeponset = night_summary_converted.get("nightsummary").sleeponset
    sleepduration = night_summary_converted.get("nightsummary").SptDuration
    n_summary_data = np.array([nights, sleeponset, sleepduration])
    week_day = night_summary_converted.get("nightsummary").weekday
    sleep_dates = night_summary_converted.get("nightsummary").get("calendar_date")
    sleeponset_time_all = night_summary_converted.get("nightsummary").get("sleeponset")
    wake_time_all = night_summary_converted.get("nightsummary").get("wakeup")
    
    return num_nights, sleeponset, sleepduration, n_summary_data, week_day, sleep_dates, sleeponset_time_all, wake_time_all


# Function to load the metadata file and get some useful variables
def load_metadata(filename):
    filename = 'meta_' + filename
    filepath = os.path.abspath(os.path.join(input_datapath + '/meta/basic', filename))

    basic = rdata.parser.parse_file(filepath)
    basic_converted = rdata.conversion.convert(basic)

    ACC = basic_converted.get("M").get("metashort").get("ENMO")*1000
    nonwearscore = basic_converted.get("M").get("metalong").get("nonwearscore")
    nw_time = basic_converted.get("M").get("metalong").get("timestamp")
    anglez = basic_converted.get("M").get("metashort").get("anglez")
    date_time = basic_converted.get("M").get("metashort").get("timestamp")
    ws3_interm = basic_converted.get("M").get("windowsizes")
    ws3 = ws3_interm[0]
    ws2 = ws3_interm[1]
    axis_range = int((2*(60/ws3)*60))

    return ACC, nonwearscore, nw_time, anglez, date_time, ws3, ws2, axis_range


# Function to create the act graphs
def create_graphs(filename):

    identifier = filename[0:12]

    # First, load files (ms4.out and metadata)
    num_nights, sleeponset, sleepduration, n_summary_data, week_day, sleep_dates, sleeponset_time_all, wake_time_all = load_ms4_file(filename)
    ACC, nonwearscore, nw_time, anglez, date_time, ws3, ws2, axis_range = load_metadata(filename)

    # Index 0=year; 1=month; 2=day; 3=hour; 4=min; 5=sec; 6=timezone
    time_split = date_time.str.split(r"T|-|:", expand=True)

    sec = time_split[5]
    min_vec = time_split[4]
    hour = time_split[3]
    time_matrix = np.column_stack((sec, min_vec, hour))
    time = sec + min_vec + hour

    ddate = time_split[0] + "-" + time_split[1] + "-" + time_split[2] 

    nightsi = np.where(time == "000012")
    nightsi = nightsi[0]
 
    # Easy way to get all the dates and then plot the date correctly
    ddate_new = ddate[nightsi+1]
    ddate_new = pd.Index(ddate_new)

    # Prepare nonwear information for plotting
    nonwear = np.zeros((np.size(ACC)))

    # take instances where nonwear was detected (on ws2 time vector) and map results onto a ws3 lenght vector for plotting purposes
    if (np.sum(np.where(nonwearscore > 1))):
        nonwear_elements = np.where(nonwearscore > 1)
        nonwear_elements = nonwear_elements[0]

        for j in range(1, np.size(nonwear_elements)):
            # The next if deals with the cases in which the first point is a nowwear data
            # When this happens, the data takes a minute to load on the APP
            # TO-DO: find a better way to treat the nonwear cases in the first datapoint
            if nonwear_elements[j-1] == 0:
                nonwear_elements[j-1] = 1

            match_loc = np.where(nw_time[nonwear_elements[j-1]] == date_time)
            match_loc = match_loc[0]
            nonwear[int(match_loc):int((int(match_loc)+(ws2/ws3)-1))] = 1

    xaxislabels = ("noon", "2pm", "4pm", "6pm", "8pm", "10pm", "midnight", "2am", "4am", "6am", "8am", "10am", "noon")
    wdaynames = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
    npointsperday = int((60/ws3)*1440)

    # Creating auxiliary vectors to store the data
    vec_acc = np.zeros((len(nightsi)+1, npointsperday))
    vec_ang = np.zeros((len(nightsi)+1, npointsperday))
    vec_sleeponset = np.zeros(len(nightsi)+1)
    vec_wake = np.zeros(len(nightsi)+1)
    vec_sleep_hour = np.zeros(len(nightsi)+1)
    vec_sleep_min = np.zeros(len(nightsi)+1)
    vec_wake_hour = np.zeros(len(nightsi)+1)
    vec_wake_min = np.zeros(len(nightsi)+1)
    vec_nonwear = np.zeros((len(nightsi)+1, npointsperday))

    if (len(nightsi) > 0):

        nplots = np.size(nightsi)+1
        x = range(1, npointsperday+1, 1)

        daycount = 1

        #for g in range(1,len(sleep_dates)+1):
        for g in range(1, nplots+1):

            print("Creating graph ", g)

            skip = 0
            check_date = 1
            change_date = 0

            if daycount == 1:
                t0 = 1
                t1 = nightsi[daycount-1]
                non_wear = nonwear[range(t0, t1+1)]
            if daycount > 1 and daycount < nplots:
                t0 = nightsi[daycount-2]+1
                t1 = nightsi[daycount-1]
                non_wear = nonwear[range(t0, t1+1)]
            if daycount == nplots:
                t0 = nightsi[daycount-2]
                t1 = np.size(date_time)
                non_wear = nonwear[range(t0, t1)]

            # Day with 25 hours, just pretend that 25th hour did not happen
            if (((t1 - t0) + 1) / (60*60/ws3) == 25):
                t1 = t1 - (60*60/ws3)
                t1 = int(t1)

            # Day with 23 hours, just extend timeline with 1 hour
            if (((t1 - t0) + 1) / (60*60/ws3) == 23):
                t1 = t1 + (60*60/ws3)
                t1 = int(t1)

            # Initialize daily "what we think you did" vectors
            acc = abs(ACC[range(t0, t1+1)])
            ang = anglez[range(t0, t1+1)]
            non_wear = nonwear[range(t0, t1)]
            extension = range(0, (npointsperday-(t1-t0))-1, 1)
            extra_extension = range(0, 1)

            # check to see if there are any sleep onset or wake annotations on this day
            sleeponset_loc = 0
            wake_loc = 0
            sw_coefs = [12, 36]

            # Index 0=day; 1=month; 2=year
            sleep_dates_split = sleep_dates.str.split(r"/", expand=True)

            # Double check because some dates are like 2019-02-25 and other dates are like 2019-2-25
            # Or some dates are like 2019-02-01 and other dates are like 2019-02-1
            for i in range(1, len(sleep_dates_split)+1):
                if (len(sleep_dates_split[0][i]) == 1):
                    sleep_dates_split[0][i] = "0" + sleep_dates_split[0][i]
                if (len(sleep_dates_split[1][i]) == 1):
                    sleep_dates_split[1][i] = "0" + sleep_dates_split[1][i]

            new_sleep_date = sleep_dates_split[2] + "-" + sleep_dates_split[1] + "-" + sleep_dates_split[0]

            # check for sleeponset & wake time that is logged on this day before midnight
            curr_date = ddate[t0]

            # check to see if it is the first day that has less than 24 and starts after midnight 
            if ((t1 - t0) < ((60*60*12)/ws3)):  # if there is less than half a days worth of data
                list_temp = list(curr_date)
                temp = int(curr_date[8:]) - 1

                if (len(str(temp)) == 1):
                    temp = "0" + str(temp)
                else:
                    temp = str(temp)

                list_temp[8:] = temp
                curr_date = ''.join(list_temp)
                new_sleep_date = pd.concat([pd.Series(curr_date), new_sleep_date])

                if (daycount == 1):
                    # Updating the all days variable to include the day before (without act data) on the first position
                    ddate_new = pd.concat([pd.Series(curr_date), pd.Series(ddate_new)])
                    ddate_new = ddate_new.reset_index()
                    ddate_new = ddate_new[0]
                    change_date = 1

            # Since the first day started before midnight:
            #ddate_new = new_sleep_date

            if (curr_date in str(new_sleep_date)):
                check_date = 0
                idx = list(new_sleep_date).index(curr_date)


            if (check_date == False):
                # Get sleeponset
                sleeponset_time = sleeponset_time_all[idx+1]

                if ((sleeponset_time >= sw_coefs[0]) & (sleeponset_time < sw_coefs[1])):
                    sleeponset_hour = int(sleeponset_time)

                    if (sleeponset_hour == 24):
                        sleeponset_hour = 0
                    if (sleeponset_hour > 24):
                        sleeponset_hour = sleeponset_hour - 24

                    sleeponset_min = (sleeponset_time - int(sleeponset_time)) * 60
                    if (int(sleeponset_min) == 60):
                        sleeponset_min = 0

                    sleeponset_locations = (((pd.to_numeric(hour[t0:t1])) == sleeponset_hour) & ((pd.to_numeric(min_vec[t0:t1])) == int(sleeponset_min))).which()
                    sleeponset_locations = list(pd.to_numeric(sleeponset_locations)+2)

                    # Need to change this line to work with boolean
                    #if(sleeponset_locations[0] == True):
                    if (len(sleeponset_locations) == 0):
                        sleeponset_loc = 0
                    else:
                        sleeponset_loc = sleeponset_locations[0]

                # Get wakeup 
                wake_time = wake_time_all[idx+1]

                if((wake_time >= sw_coefs[0]) & (wake_time < sw_coefs[1])):
                    wake_hour = int(wake_time)
                    if (wake_hour == 24):
                        wake_hour = 0
                    if (wake_hour > 24):
                        wake_hour = wake_hour - 24
                    
                    wake_min = (wake_time - int(wake_time)) * 60
                    if (wake_min == 60):
                        wake_min = 0

                    wake_locations = (((pd.to_numeric(hour[t0:t1])) == wake_hour) & ((pd.to_numeric(min_vec[t0:t1])) == int(wake_min))).which()
                    wake_locations = list(pd.to_numeric(wake_locations)+2)

                    # Need to change this line to work with boolean
                    #if(wake_locations[0] == True):
                    if (len(wake_locations) == 0):
                        wake_loc = 0
                    else:
                        wake_loc = wake_locations[0]

                vec_sleep_hour[g-1] = sleeponset_hour
                vec_sleep_min[g-1] = sleeponset_min
                vec_wake_hour[g-1] = wake_hour
                vec_wake_min[g-1] = wake_min

            # add extensions if <24hr of data
            # hold adjustments amounts on first and last day plots  
            first_day_adjust = 0
            last_day_adjust = 0

            if ((((t1 - t0)+1) != npointsperday) & (t0 == 1)):
                extension = [0]*((npointsperday-(t1-t0))-1)
                first_day_adjust = len(extension)
                acc = extension + list(acc)
                ang = extension + list(ang)
                non_wear = extension + list(non_wear)
                t1 = len(acc)

                if len(non_wear) < 17280:
                    non_wear = list(extra_extension) + list(non_wear)

                if (len(acc) == (len(x)+1)):
                    extension = extension[1:(len(extension))]
                    acc = acc[1:(len(acc))]
                    ang = ang[1:(len(ang))]
                    non_wear = non_wear[1:(len(non_wear))]

                extension_mat = np.zeros([len(extension), 6])
                # adjust any sleeponset / wake annotations if they exist:
                if (sleeponset_loc != 0):
                    sleeponset_loc = sleeponset_loc + len(extension)

                if (wake_loc != 0):
                    wake_loc = wake_loc + len(extension)

            elif (((t1-t0)+1) != npointsperday & (t1 == len(time))):
                extension = [0]*((npointsperday-(t1-t0)))
                last_day_adjust = len(acc)
                acc = list(acc) + extension
                ang = list(ang) + extension
                non_wear = list(non_wear) + extension
                
                if len(non_wear) < 17280:
                    non_wear = list(non_wear) + extension

                if (len(acc) == (len(x)+1)):
                    extension = extension[1:(len(extension))]
                    acc = acc[1:(len(acc))]
                    ang = ang[1:(len(ang))]
                    non_wear = non_wear[1:(len(non_wear))] + list(extra_extension)

                extension_mat = np.zeros([len(extension), 6])

            #for i in range(len(acc)):
            #    if acc[int(i)] >= 900:
            #        acc[int(i)] = 900

            # Comment the next line if the app will create two different graphs: one for the arm movement and one for the z-angle
            acc = (np.array(acc)/14) - 210
            
            # storing important variables in vectors to be accessed later
            vec_acc[g-1] = acc
            vec_ang[g-1] = ang
            vec_sleeponset[g-1] = sleeponset_loc
            vec_wake[g-1] = wake_loc
            vec_nonwear[g-1] = non_wear

            daycount = daycount + 1

        #print(vec_wake)
        vec_line = []
        #vec_line = [0 for i in range((daycount-1)*2)]
        # Setting nnights = 70 because GGIR version 2.0-0 need a value for the nnights variable.
        vec_line = [0 for i in range((70)*2)]

        #wake = [sleeplog_file[idx] for idx in range(len(sleeplog_file)) if idx%2==1]
        #print(type(wake))
        #sleep = [sleeplog_file[idx] for idx in range(len(sleeplog_file)) if idx%2!=1]

        excl_night = [0 for i in range(daycount)]


    ddate_temp = ddate_new[0]
    new_sleep_date_temp = new_sleep_date[1]
    if ((ddate_new[0] != new_sleep_date[1]) and (change_date == 0) and (ddate_temp[8:] > new_sleep_date_temp[8:])):
        ddate_new = pd.concat([pd.Series(new_sleep_date[1]), pd.Series(ddate_new)])
        ddate_new = ddate_new.reset_index()
        ddate_new = ddate_new[0]

    if (len(new_sleep_date) != daycount-1):
        new_sleep_date = pd.concat([new_sleep_date, pd.Series(curr_date)])
        new_sleep_date = new_sleep_date.reset_index()
        new_sleep_date = new_sleep_date[0]

    return identifier, axis_range, daycount, week_day, new_sleep_date, vec_acc, vec_ang, vec_sleeponset, vec_wake, vec_sleep_hour, vec_sleep_min, vec_wake_hour, vec_wake_min, vec_line, npointsperday, excl_night, vec_nonwear, ddate_new

# File example:
# ID onset_N1 wake_N1 onset_N2 wake_N2 onset_N3 wake_N3 ...
def create_GGIR_file(identifier, number_of_days, filename):
    n_columns = number_of_days*2
    headline = []
    first_line = []
    
    headline.append("ID")

    for ii in range(1, number_of_days+1):
        onset_name = 'onset_N' + str(ii)
        wake_name = 'wakeup_N' + str(ii)
        headline.append(onset_name)
        headline.append(wake_name)

    f = open(os.path.join(log_path,filename), 'w')
    writer = csv.writer(f)
    writer.writerow(headline)
    f.close()


def save_GGIR_file(hour_vector, fig_variables, filename):
    
    identifier = fig_variables[0]
    daycount = fig_variables[2]
    vec_line = hour_vector

    #filename = 'sleep_log_' + identifier + '_' + datetime.now() + '.csv'
    filename = 'sleeplog_' + identifier + '.csv'
    data_line = []

    data_line.append(identifier)

    #print("vec_line: ", vec_line)

    for ii in range(np.size(vec_line)):
        if (vec_line[ii] != 0):
            data_line.append(vec_line[ii])
        else:
            data_line.append("NA")
        #    data_line.append(sleep)
        #    data_line.append(wake)

    if (exists(os.path.join(log_path,filename))):
        os.remove(os.path.join(log_path,filename))
        create_GGIR_file(identifier, 70, filename)
    else:
        create_GGIR_file(identifier, 70, filename)

    f = open(os.path.join(log_path,filename), 'a')
    writer = csv.writer(f)
    writer.writerow(data_line)
    f.close()

    print("GGIR file saved!")

def save_log_file(name, identifier):

    global filename
    
    todays_date_time = datetime.now()
    todays_date = todays_date_time.strftime("%Y-%m-%d")
    todays_time = todays_date_time.strftime("%H:%M:%S")

    #filename = 'sleep_log_' + identifier + '_' + str(todays_date) + '_' + str(todays_time) + '.csv'
    filename = 'sleeplog_' + identifier + '.csv'

    header = []
    log_info = []
    log_info.append(name)
    log_info.append(identifier)
    log_info.append(date.today())
    #log_info.append(datetime.now())
    log_info.append(filename)    

    if (exists(os.path.join(log_path, 'log_file.csv'))):
        f = open(os.path.join(log_path, 'log_file.csv'), 'a')
        writer = csv.writer(f)
        writer.writerow(log_info)
        f.close()
    else:
        f = open(os.path.join(log_path, 'log_file.csv'), 'w')
        writer = csv.writer(f)
        header.append("Username")
        header.append("Participant")
        header.append("Date")
        header.append("Filename")
        writer.writerow(header)
        writer.writerow(log_info)
        f.close()

    print("Log file saved!")

    return filename

def open_sleeplog_file(identifier):
    filename = 'sleeplog_' + identifier + '.csv'

    filename_path = os.path.join(log_path, filename)
    sleeplog_file = pd.read_csv(filename_path, index_col = 0)
    sleeplog_file = sleeplog_file.iloc[0]
    wake = [sleeplog_file[idx] for idx in range(len(sleeplog_file)) if idx%2==1]
    #print(type(wake))
    sleep = [sleeplog_file[idx] for idx in range(len(sleeplog_file)) if idx%2!=1]

    '''
    for i in range(0, len(sleep)):
        if sleep[i] == '3:0:00':
            sleep[i] = 10800

    for j in range(0, len(wake)):
        if wake[j] == '3:0:00':
            wake[j] = 10800
    '''

    return sleep, wake


def save_sleeplog_file(identifier, day, sleep, wake):
    filename = 'sleeplog_' + identifier + '.csv'
    filename_path = os.path.join(log_path, filename)

    df = pd.read_csv(filename_path)
    df.iloc[0,0] = identifier
    sleep_time, wake_time = point2time(sleep, wake)
    df.iloc[0,((day)*2)-1] = sleep_time
    df.iloc[0,((day)*2)] = wake_time

    df.to_csv(filename_path, index=False)


def save_multiple_sleep_log(name, identifier, multiple_log):

    todays_date_time = datetime.now()

    if (multiple_log == None or multiple_log == []):
        multiple_log = ''

    filename = 'multiple_sleep_log.csv'

    header = []
    log_info = []
    log_info.append(name)
    log_info.append(identifier)
    log_info.append(multiple_log)
    log_info.append(todays_date_time)
    
    # If file exists, append the new information on the existing file
    if (exists(os.path.join(log_path, 'multiple_sleep_log.csv'))):
        f = open(os.path.join(log_path, 'multiple_sleep_log.csv'), 'a')
        writer = csv.writer(f)
        writer.writerow(log_info)
        f.close()
    # If file does not exists, create the file and append the information to the new file
    else:
        f = open(os.path.join(log_path, 'multiple_sleep_log.csv'), 'w')
        writer = csv.writer(f)
        header.append("Username")
        header.append("Participant")
        header.append("Multiple sleep segments")
        header.append("Last modified")
        writer.writerow(header)
        writer.writerow(log_info)
        f.close()

    print("Multiple sleep log saved!")

    return filename


def save_log_analysis_completed(identifier, completed):

    todays_date_time = datetime.now()

    filename = 'participants_with_completed_analysis.csv'

    header = []
    log_info = []
    log_info.append(identifier)
    log_info.append(completed)
    log_info.append(todays_date_time)
    
    # If file exists, append the new information on the existing file
    if (exists(os.path.join(log_path, 'participants_with_completed_analysis.csv'))):
        f = open(os.path.join(log_path, 'participants_with_completed_analysis.csv'), 'a')
        writer = csv.writer(f)
        writer.writerow(log_info)
        f.close()
    # If file does not exists, create the file and append the information to the new file
    else:
        f = open(os.path.join(log_path, 'participants_with_completed_analysis.csv'), 'w')
        writer = csv.writer(f)
        header.append("Participant")
        header.append("Is the sleep log analysis completed?")
        header.append("Last modified")
        writer.writerow(header)
        writer.writerow(log_info)
        f.close()

    print("Sleep log analysis completed!")


# File format:
# ID, day_part5, relyonguider_part4, night_part4
def save_excluded_night(identifier, excl_night):

    filename = 'data_cleaning_file_' + identifier + '.csv'
    nights_excluded = 0
    header = []
    data_night = []

    # Adjusting night variable to the format accepted by GGIR
    for i in range(1, np.size(excl_night)+1):
        if excl_night[i-1] == 1:
            if nights_excluded == 0:
                nights_excluded = i
            else:
                nights_excluded = str(nights_excluded) + ' ' + str(i)

    # Saving the csv file
    # If file exists, remove older file, create a new one, and store the data
    if (exists(os.path.join(log_path, filename))):
        os.remove(os.path.join(log_path, filename))

        f = open(os.path.join(log_path, filename), 'w')
        writer = csv.writer(f)
        header.append("ID")
        header.append("day_part5")
        header.append("relyonguider_part4")
        header.append("night_part4")
        data_night.append(identifier)
        data_night.append("")
        data_night.append("")
        data_night.append(nights_excluded)
        writer.writerow(header)
        writer.writerow(data_night)
        f.close()
    # If file does not exists, create a new file, and store the data
    else:
        f = open(os.path.join(log_path, filename), 'w')
        writer = csv.writer(f)
        header.append("ID")
        header.append("day_part5")
        header.append("relyonguider_part4")
        header.append("night_part4")
        data_night.append(identifier)
        data_night.append("")
        data_night.append("")
        data_night.append(nights_excluded)
        writer.writerow(header)
        writer.writerow(data_night)
        f.close()

    print("Excluded nights formated: ", nights_excluded)


def store_sleep_diary(day, sleep, wake):

    vec_line = fig_variables[13]

    if (sleep == 0 and wake == 0):
        vec_line[(day*2)-2] = 0
        vec_line[day*2-1] = 0
    else:
        onset_point2time = point2time(sleep, wake)
        vec_line[(day*2)-2] = onset_point2time[0]
        vec_line[day*2-1] = onset_point2time[1]

    return vec_line


def store_excluded_night(day):

    excl_night = fig_variables[15]
    excl_night[day-1] = 1

    return excl_night

def point2time(sleep, wake):

    axis_range = fig_variables[1]
    npointsperday = fig_variables[14]

    # Get sleeponset
    if int(sleep) == 0:
        sleep_point2time = '3:0:00'
    else:
        if sleep > 6*axis_range:
            temp_sleep = ((sleep*24)/npointsperday)-12
        else:
            temp_sleep = (sleep*24)/npointsperday+12
        temp_sleep_hour = int(temp_sleep)

        temp_sleep_min = (temp_sleep - int(temp_sleep)) * 60
        if (int(temp_sleep_min) == 60):
            temp_sleep_min = 0

        sleep_point2time = str(temp_sleep_hour) + ':' + str(int(temp_sleep_min)) + ':00'

    # Get wakeup 
    
    if int(wake) == 0:
        wake_point2time = '3:0:00'
    else:
        if wake > 6*axis_range:
            temp_wake = ((wake*24)/npointsperday)-12
        else:
            temp_wake = (wake*24)/npointsperday+12
        temp_wake_hour = int(temp_wake)

        temp_wake_min = (temp_wake - int(temp_wake)) * 60
        if (int(temp_wake_min) == 60):
            temp_wake_min = 0

        wake_point2time = str(temp_wake_hour) + ':' + str(int(temp_wake_min)) + ':00'
    #print("type wake_point2time: ", type(wake_point2time))
    return sleep_point2time, wake_point2time


def time2point(sleep, wake):

    axis_range = fig_variables[1]
    npointsperday = fig_variables[14] 

    if sleep == 0:
        sleep2return = 0
    else:
        sleep_split = sleep.split(":")
        # Get sleep time and transform to timepoints
        sleep_time_hour = int(sleep_split[0])
        sleep_time_min = int(sleep_split[1])
        # hour
        if sleep_time_hour >= 0 and sleep_time_hour < 12:
            sleep_time_hour = (((sleep_time_hour+12)*8640)/12)
        else:
            sleep_time_hour = ((sleep_time_hour-12)*8640)/12
        # minute
        if sleep_time_min == 0:
            sleep_time_min = 0
        else:
            sleep_time_min = ((sleep_time_min*12))

        sleep2return = sleep_time_hour + sleep_time_min

    if wake == 0:
        wake2return = 0
    else:
        wake_split = wake.split(":")
        # Get wake time and transform to timepoints
        wake_time_hour = int(wake_split[0])
        wake_time_min = int(wake_split[1])
        # hour
        if wake_time_hour >= 0 and wake_time_hour < 12:
            wake_time_hour = ((wake_time_hour+12)*8640)/12
        else:
            wake_time_hour = ((wake_time_hour-12)*8640)/12
        # minute
        if wake_time_min == 0:
            wake_time_min = 0
        else:
            wake_time_min = ((wake_time_min*12))
        wake2return = wake_time_hour + wake_time_min


    return sleep2return, wake2return


colors = {
    'background': '#FFFFFF',
    'text': '#111111',
    'title_text': '#0060EE'
}


app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    
    html.Img(src='/assets/CMI_Logo_title.png', style={'height':'70%', 'width':'70%'}),

    html.Div([
        dcc.ConfirmDialog(
            id='insert-user',
            message='Insert the evaluator\'s name before continue'
        )
    ]),

    html.Div([
        dcc.Input(
            id="input_name", 
            type="text",
            placeholder="Insert evaluator's name",
            disabled=False,
            size="40"
        ),
        dcc.Dropdown(files, id='my-dropdown', placeholder="Select subject..."),
        dbc.Spinner(html.Div(id="loading"))
    ], style={'padding': 10}),

    html.Pre(id="annotations-data")
])

@app.callback(
    [Output('annotations-data', 'children'), 
     Output('loading', 'children'),
     Output('insert-user', 'displayed'),
     Output('input_name', 'disabled')],
    Input('my-dropdown', 'value'),
    Input('input_name', 'value'),
    suppress_callback_exceptions=True,
    prevent_initial_call=True,
)

def parse_contents(filename, name):
    
    global fig_variables
    

    try:
        if (name == None or name == ''):
            return '','',True,False
        else:
            if 'RData' in filename:
                print("Loading data ...")
                fig_variables = create_graphs(filename)
                identifier = fig_variables[0]
                axis_range = fig_variables[1]
                sleep = fig_variables[7]
                wake = fig_variables[8]
                daycount = fig_variables[2]
                hour_vector = []
                sleep_tmp = []
                wake_tmp = []

                for ii in range(0, len(sleep)):
                    sleep_tmp1, wake_tmp1 = point2time(sleep[ii], wake[ii])
                    sleep_tmp.append(sleep_tmp1)
                    wake_tmp.append(wake_tmp1)

                for jj in range(0, daycount-1):
                    hour_vector.append(sleep_tmp[jj])
                    hour_vector.append(wake_tmp[jj])

                save_GGIR_file(hour_vector, fig_variables, filename)
                
                save_log_file(name, fig_variables[0])

                return [html.Div([
                    html.B("* All the changes will be automatically saved\n\n", style={"color":"red"}),
                    html.B("Select day for participant " + fig_variables[0] +": "),
                    dcc.Slider(1, fig_variables[2]-1, 1, 
                        value=1,
                        id='day_slider',
                    )], style={'margin-left': '20px', 'padding': 10}),

                    dcc.Checklist([' I\'m done and I would like to proceed to the next participant. '], id='are-you-done', style={'margin-left': '50px'}),
                    html.Pre(id="check-done"),

                    dcc.Checklist([' Does this participant have nap times in any of the days?'], id='multiple_sleep', style={"margin-left": "50px"}),
                    html.Pre(id="checklist-items"),

                    daq.BooleanSwitch(id='exclude-night', on=False, label=' Does this participant have more than 2 hours of missing sleep data from 8PM to 8AM?'),
                                   
                    dcc.Graph(id='graph'),

                    html.Div([
                        dcc.RangeSlider(
                            min = 0, 
                            max = fig_variables[14],
                            step = 1,
                            marks = {
                                0: 'noon',
                                axis_range: '2pm',
                                2*axis_range: '4pm',
                                3*axis_range: '6pm',
                                4*axis_range: '8pm',
                                5*axis_range: '10pm',
                                6*axis_range: 'midnight',
                                7*axis_range: '2am',
                                8*axis_range: '4am',
                                9*axis_range: '6am',
                                10*axis_range: '8am',
                                11*axis_range: '10am',
                                12*axis_range: 'noon'
                            },
                            id = 'my-range-slider',
                            ),
                        html.Pre(id="annotations-slider")],
                        style = {"margin-left": "55px", "margin-right": "55px"}),

                    #html.Button('Refresh graph', id='btn_clear', style={"margin-left": "15px"}),
                            
                    html.Pre(id="annotations-save"),
                    html.P("\n\n     This software is licensed under the GNU Lesser General Public License v3.0\n     Permissions of this copyleft license are conditioned on making available complete source code of licensed works and modifications under the same license or\n     the GNU GPLv3. Copyright and license notices must be preserved.\n     Contributors provide an express grant of patent rights.\n     However, a larger work using the licensed work through interfaces provided by the licensed work may be distributed under different terms\n     and without source code for the larger work.", style={"color":"gray"})
                    ], '', False, True

    except Exception as e:
        print(e)
        return dash.no_update


@app.callback(
    Output('exclude-night', 'on'),
    Input('day_slider', 'value')
)

def update_exclude_switch(day):
    vec_night_to_exclude = fig_variables[15]

    if vec_night_to_exclude[day-1] == 0:
        return False
    else:
        return True


@app.callback(
    Output('graph', 'figure'),
    Output('my-range-slider', 'value'),
    Input('day_slider', 'value'),
    #Input('btn_clear', 'n_clicks'),
    Input('exclude-night', 'on'),
    Input('my-range-slider', 'value'),
    #[State('my-range-slider', 'value')],
    suppress_callback_exceptions=True,
    #prevent_initial_call=True,
)

#def update_graph(day, nclicks, exclude_button, position):
def update_graph(day, exclude_button, position):
    identifier = fig_variables[0]
    axis_range = fig_variables[1]
    daycount = fig_variables[2]
    week_day = fig_variables[3]
    new_sleep_date = fig_variables[4]
    vec_acc = fig_variables[5]
    vec_ang = fig_variables[6]
    #vec_sleeponset = fig_variables[7]
    #vec_wake = fig_variables[8]
    npointsperday = fig_variables[14]
    night_to_exclude = fig_variables[15]
    vec_nonwear = fig_variables[16]
    all_dates = fig_variables[17]

    sleeponset, wakeup = open_sleeplog_file(identifier)

    vec_sleeponset, vec_wake = time2point(sleeponset[day-1], wakeup[day-1])

    value_nonwear = []
    end_value_new = []
    end_value = []
    begin_value = []

    month_1 = calendar.month_abbr[int(all_dates[day-1][5:7])]
    day_of_week_1 = datetime.fromisoformat(all_dates[day-1])
    day_of_week_1 = day_of_week_1.strftime("%A")
    

    if (day < daycount-1):
        month_2 = calendar.month_abbr[int(all_dates[day][5:7])]
        day_of_week_2 = datetime.fromisoformat(all_dates[day])
        day_of_week_2 = day_of_week_2.strftime("%A")

        title_day = 'Day ' + str(day) + ': ' + day_of_week_1 + ' - ' + all_dates[day-1][8:] + ' ' + month_1 + ' ' + all_dates[day-1][0:4]
        title_day = title_day + ' | Day ' + str(day+1) + ': ' + day_of_week_2 + ' - ' + all_dates[day][8:] + ' ' + month_2 + ' ' + all_dates[day][0:4]
    else:
        title_day = 'Day ' + str(day) + ': ' + day_of_week_1 + ' - ' + all_dates[day-1][8:] + ' ' + month_1 + ' ' + all_dates[day-1][0:4]
    
    #global fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=vec_ang[day-1,:], mode='lines', name='Angle of sensor\'s z-axis', line_width=1, line_color="blue"))
    fig.add_trace(go.Scatter(y=vec_acc[day-1,:], mode='lines', name = 'Arm movement', line_width=1, line_color="black"))
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_layout(title=title_day)
    
    if (int(vec_sleeponset) == 10800 and int(vec_wake) == 10800):
    #if (int(vec_sleeponset) > 0 and int(vec_wake) > 0):
        fig.add_vrect(x0=int(vec_sleeponset), x1=int(vec_wake), line_width=0, fillcolor="red", opacity=0.2)
    else:
        fig.add_vrect(x0=int(vec_sleeponset), x1=int(vec_wake), line_width=0, fillcolor="red", opacity=0.2, annotation_text="sleep window", annotation_position="top left")

    # Nonwear
    vec_for_the_day = vec_nonwear[day-1]
    if (int(vec_for_the_day[0]) == 0):
        begin_value = np.where(np.diff(vec_nonwear[day-1]) == 1)
        begin_value = begin_value[0] + 180
        end_value = np.where(np.diff(vec_nonwear[day-1]) == -1)
        end_value = end_value[0] + 180
    else:
        first_value = 0
        begin_value = np.where(np.diff(vec_nonwear[day-1]) == 1)
        begin_value = np.asarray(begin_value)
        begin_value = np.insert(begin_value, 0, first_value)
        begin_value = begin_value + 180
        begin_value = np.insert(begin_value, 0, first_value)
        end_value = np.where(np.diff(vec_nonwear[day-1]) == -1)
        end_value = end_value[0] + 180
        end_value = np.insert(end_value, 0, 179)

    if len(end_value) == 1:
        fig.add_vrect(x0=int(begin_value), x1=int(end_value), line_width=0, fillcolor="green", opacity=0.5, annotation_text="nonwear", annotation_yshift=110, annotation_position="left")
    elif len(end_value) > 1:
        fig.add_vrect(x0=int(begin_value[1]), x1=int(end_value[1]), line_width=0, fillcolor="green", opacity=0.5, annotation_text="nonwear", annotation_yshift=110, annotation_position="left")
        for ii in range(2,len(end_value)-1):
            fig.add_vrect(x0=int(begin_value[ii]), x1=int(end_value[ii]), line_width=0, fillcolor="green", opacity=0.5)

    fig.update_xaxes(
        ticktext=["noon", "2pm", "4pm", "6pm", "8pm", "10pm", "midnight", "2am", "4am", "6am", "8am", "10am", "noon"],
        tickvals=[1, axis_range, 2*axis_range, 3*axis_range, 4*axis_range, 5*axis_range, 6*axis_range,
                  7*axis_range, 8*axis_range, 9*axis_range, 10*axis_range, 11*axis_range, 12*axis_range] )

    #fig.update_layout(showlegend=True, dragmode="drawrect")
    fig.update_layout(showlegend=True)
    fig.update_yaxes(visible=False, showticklabels=False)
    #fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

    print("Button to exclude night: ", exclude_button)

    if (exclude_button == False):
        night_to_exclude[day-1] = 0
        save_excluded_night(fig_variables[0], night_to_exclude)
        print(night_to_exclude)
    else:
        night_to_exclude[day-1] = 1
        save_excluded_night(fig_variables[0], night_to_exclude)
        print(night_to_exclude)
    
    #save_sleeplog_file(identifier, day, vec_sleeponset, vec_wake)

    return fig,[int(vec_sleeponset),int(vec_wake)]   

'''
def update_sleep_window(day, position):

    identifier = fig_variables[0]

    sleep, wake = open_sleeplog_file(identifier)

    if (int(sleep[day]) > 0 and int(wake[day]) > 0):
        fig.add_vrect(x0=sleep[day], x1=wake[day], line_width=0, fillcolor="red", opacity=0.2, annotation_text="sleep window", annotation_position="top left")
    else:
        fig.add_vrect(x0=sleep[day], x1=wake[day], line_width=0, fillcolor="red", opacity=0.2)

    #fig.update_layout()

    return fig
'''

@app.callback(
    Output('checklist-items', 'children'),
    Input('multiple_sleep', 'value'),
    Input('input_name', 'value'),
    suppress_callback_exceptions=True,
    prevent_initial_call=True
)

def save_multiple_sleep(multiple_value, name):
    if (multiple_value == None or multiple_value == []):
        save_multiple_sleep_log(name, fig_variables[0], "No")
        return ''
    else:
        save_multiple_sleep_log(name, fig_variables[0], "Yes")
        return ''

@app.callback(
    Output('check-done', 'children'),
    Input('are-you-done', 'value')
)

def save_log_done(value):
    if (value == None or value == []):
        print("Sleep log analysis not completed yet.")
    else:
        save_log_analysis_completed(fig_variables[0], "Yes")


@app.callback(
    Output("annotations-save", "children"),
    Input('my-range-slider', 'drag_value'),
    State("day_slider", "value")
)

def save_info(drag_value, day):

    identifier = fig_variables[0]
    #tt = store_sleep_diary(day, value[0], value[1])
    save_sleeplog_file(identifier, day, drag_value[0], drag_value[1])

    print('Saving chages into sleep log')




if __name__ == '__main__':
    #app.run_server(debug=True,dev_tools_ui=False)
    #app.run_server(debug=True, port=8050)
    app.run_server(debug=False)

