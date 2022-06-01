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
from pathlib import Path
from os import listdir
from os.path import exists, isfile, join
from datetime import date, datetime
from plotly.subplots import make_subplots
from ftplib import FTP
import argparse


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
    nonwear = list(range(len(ACC)))
    # take instances where nonwear was detected (on ws2 time vector) and map results onto a ws3 lenght vector for plotting purposes
    #if (np.sum(np.where(nonwearscore > 1))):
    #    nonwear_elements = np.where(nonwearscore > 1)

    #    for j in range(1, np.size(nonwear_elements)):
    #        nonwear_elements = np.asarray(nonwear_elements)
    #        match_loc = np.where(nw_time[nonwear_elements[0,j]] == date_time)
    #        match_loc = match_loc[0]
    #        match_loc = int(match_loc)
    #        for temp in range(match_loc, int((match_loc+(ws2/ws3)-1))):
    #            nonwear[temp] = 1

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
            if daycount > 1 and daycount < nplots:
                t0 = nightsi[daycount-2]+1
                t1 = nightsi[daycount-1]
            if daycount == nplots:
                t0 = nightsi[daycount-2]
                t1 = np.size(date_time)

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
            #nonwear_temp = nonwear[t0:t1+1]
            extension = range(0, (npointsperday-(t1-t0))-1, 1)

            # check to see if there are any sleep onset or wake annotations on this day
            sleeponset_loc = 0
            wake_loc = 0
            sw_coefs = [12, 36]

            # Index 0=day; 1=month; 2=year
            #print("sleep_dates: ", sleep_dates)
            sleep_dates_split = sleep_dates.str.split(r"/", expand=True)

            # Double check because some dates are like 2019-02-25 and other dates are like 2019-2-25
            # Or some dates are like 2019-02-1 and other dates are like 2019-02-01
            for i in range(1, len(sleep_dates_split)+1):
                if (len(sleep_dates_split[0][i]) == 1):
                    sleep_dates_split[0][i] = "0" + sleep_dates_split[0][i]
                if (len(sleep_dates_split[1][i]) == 1):
                    sleep_dates_split[1][i] = "0" + sleep_dates_split[1][i]

            new_sleep_date = sleep_dates_split[2] + "-" + sleep_dates_split[1] + "-" + sleep_dates_split[0]

            # check for sleeponset & wake time that is logged on this day before midnight
            curr_date = ddate[t0]

            # check to see if it is the first day that has less than 24 and starts after midnight 
            if ((t1 - t0) < ((60*60*12)/ws3)):
                list_temp = list(curr_date)
                temp = int(curr_date[8:]) - 1
                list_temp[8:] = str(temp)
                curr_date = ''.join(list_temp)
                new_sleep_date = pd.concat([pd.Series(curr_date), new_sleep_date])

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
                #nonwear_temp = extension + list(nonwear_temp)
                t1 = len(acc)

                if (len(acc) == (len(x)+1)):
                    extension = extension[1:(len(extension))]
                    acc = acc[1:(len(acc))]
                    ang = ang[1:(len(ang))]
                    #nonwear_temp = nonwear_temp[1:(len(nonwear_temp))]

                extension_mat = np.zeros([len(extension), 6])
                # adjust any sleeponset / wake annotations if they exist:
                if (sleeponset_loc != 0):
                    sleeponset_loc = sleeponset_loc + len(extension)

                if (wake_loc != 0):
                    wake_loc = wake_loc + len(extension)

            if (((t1-t0)+1) != npointsperday & (t1 == len(time))):
                extension = [0]*((npointsperday-(t1-t0))-1)
                last_day_adjust = len(acc)
                acc = list(acc) + extension
                ang = list(ang) + extension
                #nonwear_temp = list(nonwear_temp) + extension

                if (len(acc) == (len(x)+1)):
                    extension = extension[1:(len(extension))]
                    acc = acc[1:(len(acc))]
                    ang = ang[1:(len(ang))]
                    #nonwear_temp = nonwear_temp[1:len(nonwear_temp)]

                extension_mat = np.zeros([len(extension), 6])

            #for i in range(len(acc)):
            #    if acc[int(i)] >= 900:
            #        acc[int(i)] = 900

            acc = (np.array(acc)/14) - 210

            # storing important variables in vectors to be accessed later
            vec_acc[g-1] = acc
            vec_ang[g-1] = ang
            vec_sleeponset[g-1] = sleeponset_loc
            vec_wake[g-1] = wake_loc
            #vec_nonwear[g-1] = nonwear_temp

            daycount = daycount + 1

        vec_line = []
        vec_line = [0 for i in range((daycount-1)*2)]
        excl_night = [0 for i in range(daycount)]

    ddate_temp = ddate_new[0]
    new_sleep_date_temp = new_sleep_date[1]
    if ((ddate_new[0] != new_sleep_date[1]) and (change_date == 0) and (ddate_temp[8:] > new_sleep_date_temp[8:])):
        ddate_new = pd.concat([pd.Series(new_sleep_date[1]), pd.Series(ddate_new)])
        ddate_new = ddate_new.reset_index()
        ddate_new = ddate_new[0]

    if (len(new_sleep_date) != daycount-1):
        #print("curr_date: ", curr_date)
        new_sleep_date = pd.concat([new_sleep_date, pd.Series(curr_date)])
        new_sleep_date = new_sleep_date.reset_index()
        new_sleep_date = new_sleep_date[0]
        #print("new_sleep_date after : ", new_sleep_date)


    return identifier, axis_range, daycount, week_day, new_sleep_date, vec_acc, vec_ang, vec_sleeponset, vec_wake, vec_sleep_hour, vec_sleep_min, vec_wake_hour, vec_wake_min, vec_line, npointsperday, excl_night, vec_nonwear, ddate_new

# File example:
# ID onset_N1 wake_N1 onset_N2 wake_N2 onset_N3 wake_N3 ...
def create_GGIR_file(identifier, number_of_days, filename):
    n_columns = number_of_days*2
    headline = []
    first_line = []
    
    headline.append("ID")

    for ii in range(1, number_of_days):
        onset_name = 'onset_N' + str(ii)
        wake_name = 'wake_N' + str(ii)
        headline.append(onset_name)
        headline.append(wake_name)

    f = open(os.path.join(log_path,filename), 'w')
    writer = csv.writer(f)
    writer.writerow(headline)
    f.close()


def save_GGIR_file(tt, fig_variables, filename):
    
    identifier = fig_variables[0]
    daycount = fig_variables[2]
    vec_line = tt

    #filename = 'sleep_log_' + identifier + '_' + datetime.now() + '.csv'
    data_line = []

    data_line.append(identifier)

    for ii in range(np.size(vec_line)):
        if (vec_line[ii] != 0):
            data_line.append(vec_line[ii])
        else:
            data_line.append("NA")

    if (exists(os.path.join(log_path,filename))):
        os.remove(os.path.join(log_path,filename))
        create_GGIR_file(identifier, daycount, filename)
    else:
        create_GGIR_file(identifier, daycount, filename)

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

    filename = 'sleep_log_' + identifier + '_' + str(todays_date) + '_' + str(todays_time) + '.csv'

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
    # If file does not exists, create the file and append the information on the new file
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

    onset_point2time = point2time(sleep, wake)
    vec_line[(day*2)-2] = onset_point2time[0]
    vec_line[day*2-1] = onset_point2time[1]

    return vec_line


def store_excluded_night(day):

    excl_night = fig_variables[15]
    excl_night[day-1] = 1

    return excl_night


def point2time(x0, x1):

    axis_range = fig_variables[1]
    npointsperday = fig_variables[14]

    # Get sleeponset
    temp_sleeponset_loc = x0

    if temp_sleeponset_loc > 6*axis_range:
        temp_sleep = ((temp_sleeponset_loc*24)/npointsperday)-12
    else:
        temp_sleep = (temp_sleeponset_loc*24)/npointsperday+12
    temp_sleep_hour = int(temp_sleep)

    temp_sleep_min = (temp_sleep - int(temp_sleep)) * 60
    if (int(temp_sleep_min) == 60):
        temp_sleep_min = 0

    sleep_point2time = str(temp_sleep_hour) + ':' + str(int(temp_sleep_min)) + ':00'

    # Get wakeup 
    temp_wake_loc = x1
    if temp_wake_loc > 6*axis_range:
        temp_wake = ((temp_wake_loc*24)/npointsperday)-12
    else:
        temp_wake = (temp_wake_loc*24)/npointsperday+12
    temp_wake_hour = int(temp_wake)

    temp_wake_min = (temp_wake - int(temp_wake)) * 60
    if (int(temp_wake_min) == 60):
        temp_wake_min = 0

    wake_point2time = str(temp_wake_hour) + ':' + str(int(temp_wake_min)) + ':00'

    return sleep_point2time, wake_point2time


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
            #debounce=True,
            size="40"
        ),
        dcc.Dropdown(files, id='my-dropdown', placeholder="Select subject..."),
        dbc.Spinner(html.Div(id="loading"))
    ], style={'padding': 10}),

    #dcc.Tabs(id = 'tabs-example-1', 
    #         value = 'tab-1', 
    #         children = [
    #            dcc.Tab(label='Tab one', value = 'tab-1'),
    #            dcc.Tab(label='Tab two', value = 'tab-2')
    #         ]
    #),

    html.Pre(id="annotations-data")
])

@app.callback(
    [Output('annotations-data', 'children'), 
     Output('loading', 'children'),
     Output('insert-user', 'displayed')],
    Input('my-dropdown', 'value'),
    Input('input_name', 'value'),
    suppress_callback_exceptions=True,
    prevent_initial_call=True,
)

def parse_contents(filename, name):
    
    global fig_variables

    try:
        if (name == None or name == ''):
            return '','',True
        else:
            if 'RData' in filename:
                print("Loading data ...")
                fig_variables = create_graphs(filename)
                
                save_log_file(name, fig_variables[0])

                return [html.Div([
                    html.B("Select day for participant " + fig_variables[0] +": "),
                    dcc.Slider(1, fig_variables[2]-1, 1, 
                        value=1,
                        id='day_slider',
                    ),
                    daq.BooleanSwitch(id='exclude-night', on=False, label=' Do you want to exclude this night from sleep analysis?'),
                    ], style={'padding': 10}),
                    
                    dcc.Graph(id='graph'),
                    
                    dcc.Checklist([' Does this participant have multiple segments of sleep?'], id='multiple_sleep', style={"margin-left": "15px"}),
                    html.Pre(id="checklist-items"),
                            
                    html.Button('Clear graph', id='btn_clear', style={"margin-left": "15px"}),
                    #html.Button('Save changes', id='btn_save', style={"margin-left": "5px"}),
                    #html.Button('Next participant', id='btn_next', style={"margin-left": "1050px"}),
                            
                    html.Pre(id="annotations-save")], '', False

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
    Input('day_slider', 'value'),
    Input('btn_clear', 'n_clicks'),
    Input('exclude-night', 'on'),
    suppress_callback_exceptions=True,
    #prevent_initial_call=True,
)

def update_graph(day, nclicks, exclude_button):

    axis_range = fig_variables[1]
    daycount = fig_variables[2]
    week_day = fig_variables[3]
    new_sleep_date = fig_variables[4]
    vec_acc = fig_variables[5]
    vec_ang = fig_variables[6]
    vec_sleeponset = fig_variables[7]
    vec_wake = fig_variables[8]
    npointsperday = fig_variables[14]
    night_to_exclude = fig_variables[15]
    vec_nonwear = fig_variables[16]
    all_dates = fig_variables[17]


    #print("vec_nonwear: ", vec_nonwear)
    #print("vec_nonwear[day,:]: ", vec_nonwear[day-1,:])

    #if (day < daycount-1):
    #    title_day = 'Day ' + str(day) + ': ' + week_day[day] + ' ' + new_sleep_date[day-1]
    #    title_day = title_day + ' / ' + 'Day ' + str(day+1) + ': ' + week_day[day+1] + ' ' + new_sleep_date[day]
    #else:
    #    title_day = 'Day ' + str(day) + ': ' + week_day[day] + ' ' + new_sleep_date[day-1]

    #month = calendar.month_abbr[int(all_dates[day-1][5:7])]
    #day_of_week = datetime.fromisoformat(all_dates[day-1])
    #day_of_week = day_of_week.strftime("%A")
    #title_day = 'Day ' + str(day) + ': ' + day_of_week + ' | ' + all_dates[day-1][8:] + ' ' + month + ' ' + all_dates[day-1][0:4]

    
    fig = px.line(y=vec_acc[day-1,:])#, title = title_day)
    fig.update_traces(line_color='black')
    fig.add_trace(go.Scatter(y=vec_ang[day-1,:], mode='lines', name='Arm movement', line_width=1, line_color="blue"))
    #fig.add_vrect(x0=vec_nonwear[day-1], x1=vec_nonwear[day-1], line_width=0, fillcolor="green", opacity=0.2)
    
    fig.add_vrect(x0=vec_sleeponset[day-1], x1=vec_wake[day-1], line_width=0, fillcolor="red", opacity=0.2)

    fig.update_xaxes(
        ticktext=["noon", "2pm", "4pm", "6pm", "8pm", "10pm", "midnight", "2am", "4am", "6am", "8am", "10am", "noon"],
        tickvals=[1, axis_range, 2*axis_range, 3*axis_range, 4*axis_range, 5*axis_range, 6*axis_range,
                  7*axis_range, 8*axis_range, 9*axis_range, 10*axis_range, 11*axis_range, 12*axis_range] )

    fig.update_layout(showlegend=False, dragmode="drawrect")
    fig.update_yaxes(visible=False, showticklabels=False)

    print("Button to exclude night: ", exclude_button)

    if (exclude_button == False):
        night_to_exclude[day-1] = 0
        save_excluded_night(fig_variables[0], night_to_exclude)
        print(night_to_exclude)
        
    else:
        night_to_exclude[day-1] = 1
        save_excluded_night(fig_variables[0], night_to_exclude)
        print(night_to_exclude)
     
    return fig   


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
    Output("annotations-save", "children"),
    Input("graph", "relayoutData"),
    State("day_slider", "value"),
    #Input("btn_save", "n_clicks"),
    suppress_callback_exceptions=True,
    prevent_initial_call=True
)

def save_info(relayout_data, day): #, btn_save):

    if ("shapes" in relayout_data):
        if day:
            last_shape = relayout_data["shapes"][-1]
            x0 = int(last_shape["x0"])
            x1 = int(last_shape["x1"])
            
            # In case the user draws from left to right side (then x0 will be greater than x1, and it shouldn't be)
            if x0 > x1:
                x0_temp = x1
                x1 = x0
                x0 = x0_temp
            tt = store_sleep_diary(day, x0, x1)
        
        #if btn_save:
        #    print('Saving changes into sleep log')
        #    save_GGIR_file(tt, fig_variables, filename)
    
        print(tt)
        print('Saving changes into sleep log')
        save_GGIR_file(tt, fig_variables, filename)
        
    return ''



if __name__ == '__main__':
    #app.run_server(debug=True,dev_tools_ui=False)
    #app.run_server(debug=True, port=8050)
    app.run_server(debug=True)