#!/usr/bin/env python
# coding: utf-8
# In[296]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import random as rd
import time 
#%% Input data 
#Francesco
traffic_ldv = pd.read_excel('C:/Users/Francesco/Desktop/internship/hourlytraffic_ldv.xlsx', sheet_name = "Traffic", header = [0], index_col = 0)
#traffic.index = pd.to_datetime(traffic.index, format='%Y%m%d%H%M')
traffic_ldv.index = pd.to_datetime(traffic_ldv.index,utc = True)
traffic_ldv.fillna(0, inplace = True)

stations = pd.read_excel('C:/Users/Francesco/Desktop/internship/hourlytraffic_ldv.xlsx', sheet_name = "Stations", header = [0], index_col = 0)
stations = stations.apply(np.floor)

#s = [0,1,2,3,4]
#stations = stations.reindex(s)
stretches = pd.read_excel('C:/Users/Francesco/Desktop/internship/hourlytraffic_ldv.xlsx', sheet_name = "Stretches", header = [0], index_col = 0) 

traffic_hdv = pd.read_excel('C:/Users/Francesco/Desktop/internship/hourlytraffic_hdv.xlsx', index_col = 0)
traffic_hdv.index = pd.to_datetime(traffic_ldv.index,utc = True)
traffic_hdv.fillna(0, inplace = True)

nodes_position = pd.DataFrame(data=stretches.loc[:,'end']) #data frame of spatial position of nodes along the highway
nodes_position = nodes_position.drop([48])

#%% Alberto  
traffic_ldv = pd.read_excel('G:/Mi unidad/FuChar/Internship - Francesco/Coding/Data/hourlytraffic_ldv.xlsx', sheet_name = "Traffic", header = [0], index_col = 0)
#traffic should not take in account station n.7 since 
traffic_ldv.index = pd.to_datetime(traffic_ldv.index,utc = True)
traffic_ldv.fillna(0, inplace = True)

traffic_hdv = pd.read_excel(r'G:/Mi unidad/FuChar/Internship - Francesco/Coding/Data/hourlytraffic_hdv.xlsx', index_col = 0)
traffic_hdv.index = pd.to_datetime(traffic_ldv.index,utc = True)
traffic_hdv.fillna(0, inplace = True)

stations = pd.read_excel('G:/Mi unidad/FuChar/Internship - Francesco/Coding/Data/hourlytraffic_ldv.xlsx', sheet_name = "Stations", header = [0], index_col = 0)
stations = stations.apply(np.floor)
#We do not need this:
# s = [0,1,2,3,4]
# stations = stations.reindex(s)
# stations
stretches = pd.read_excel('G:/Mi unidad/FuChar/Internship - Francesco/Coding/Data/hourlytraffic_ldv.xlsx', sheet_name = "Stretches", header = [0], index_col = 0) 
stretches

#nodes_position = pd.DataFrame(data=stretches.iloc[0:-2,2]) 
nodes_position = pd.DataFrame(data=stretches.loc[:,'end']) #data frame of spatial position of nodes along the highway
nodes_position = nodes_position.drop([48])
#%% Input Data for Energy model
m_ldv = 5 #km/kWh, F. H. Malik and M. Lehtonen, “Analysis of power network loading due to fast charging of Electric Vehicles on highways” 
v_ldv = 90
p_ldv = 50 #kW (most of cars present on the market charge at this power, depending on the vehicle it can be higher)

m_hdv = 0.45 #km/kWh Taljegard, M.  "Large-scale implementation of electric road systems: Associated costs and the impact on CO 2 emissions"
v_hdv = 80
p_hdv = 200 #kW Taljegard, M.  "Large-scale implementation of electric road systems: Associated costs and the impact on CO 2 emissions"

SoC_low = 0.2 
# nodes_position = pd.DataFrame(data=stretches.loc[:,'end']) #data frame of spatial position of nodes along the highway
# nodes_position = nodes_position.drop([48])

#%%
def get_node_traffic(traffic):
    ''' Calculate nodes traffic from traffic flows on stretches
    Input - traffic /
    Output - nodes_in
    '''
#Ratios from excel sheet entry_exit (calculated), !!!!!!!!!!!!!!!!! explain better what do we mean by this
    entry_avg_ratio = 0.495
    exit_avg_ratio = 0.505
    
    d = [] #deltas list: difference of vehicles from one stretch to the other
    for i in range(len(traffic.columns)-1):
        d.append(abs(traffic.iloc[:,i+1] - traffic.iloc[:,i])) #deltas list filling
    
    d_array = np.array(d).T #deltas array conversion and transposition
    
    nodes_array_exit = d_array*exit_avg_ratio #exiting traffic flow at nodes 
    nodes_array_entry = d_array*entry_avg_ratio #entering traffic flow at nodes
    
    nodes_in = pd.DataFrame(data=nodes_array_entry, index=traffic.index) #entering traffic flow at nodes data frame creation
    nodes_out = pd.DataFrame(data=nodes_array_exit, index=traffic.index) #exiting traffic flow at nodes data frame creation
    nodes_in = nodes_in.apply(np.floor) #round up (defect)
    nodes_out = nodes_out.apply(np.floor)
    return nodes_in

#%%
nodes_in_hdv = get_node_traffic(traffic = traffic_hdv)
nodes_in_ldv = get_node_traffic(traffic = traffic_ldv)#%% t_recharge = pd.DataFrame(0, index=traffic.index, columns=stations.index)
#%%
stations
#%%

def get_demand_static(p_ev, nodes_in, traffic, stations, hours, ml_ev, v, p_charge):
    ''' Calculate for each hour and for each station the energy requested, plus the charging time that vehicles are taking to complete their charging cycle
    Input: p_ev, traffic, stations -> this could be changed to get different types of charging stations positioning
    Output: e_stations, t_charged
    '''
   
    nodes = 34 #nodes = len(nodes_in_ev.columns)
    
    #Initialize the dataframe for energy and time
    e_stations = pd.DataFrame(0, index=traffic.index, columns=stations.index) #data frame for the energy per station per hour
    t_charged = pd.DataFrame(list(), index=traffic.index, columns=stations.index) #time of recharge columns=stations.index
    
    #Time needs to be filled with empty lists
    for i in range(hours):
        for j in range(6):
            t_charged.iloc[i,j] = []
    t_charged
    
     # Question: How many of the total EVs are charging? EV_charging(time)/EV_in !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    nodes_in_ev = nodes_in*p_ev #traffic flow ev's data frame
    nodes_in_ev = nodes_in_ev.apply(np.floor) #round up (defect) 
        
    for i in range (nodes): #nodes


        for j in range(hours): #hours
            
            t_recharge = list() #necessary?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            for z in range(int(nodes_in_ev.iloc[j,i])):

                E_charged = 0 #energy charged to the vehicle at the beginning is equal to 0
                
                SoC_start_high = rd.uniform(0.6,0.8) #if vehicle is starting his journey in the first half of the day -> higher SoC  
                SoC_start_low = rd.uniform(0.3,0.5) #if vehicle is starting his journey in the second half of the day -> lower SoC 
                        
            # HDV distance is higher compared to normal cars

                # if j%12:
                #     if j%24:
                #         SoC_start = SoC_start_low
                #     else:
                #         SoC_start = SoC_start_high
                # else:
                #     if j%3:
                #         if j%6:
                #             SoC_start = SoC_start_low
                #         else:
                #             SoC_start_high
                #     else:    
                SoC_start = rd.choice([SoC_start_high, SoC_start_low])
                     

                    
            #random parameters from: F. H. Malik and M. Lehtonen, “Analysis of power network loading due to fast charging of Electric Vehicles on highways” 
            #     SoC_start_high = rd.uniform(0.6,0.8) #if vehicle is starting his journey in the first half of the day -> higher SoC  
            #     SoC_start_low = rd.uniform(0.3,0.5) #if vehicle is starting his journey in the second half of the day -> lower SoC 
            #     SoC_start = rd.choice([SoC_start_high, SoC_start_low])
                #print(SoC_start)
            # HDV distance is higher compared to normal cars
            #random parametform(0.2,0.9) #20-90%, F. H. Malik and M. Lehtonen, “Analysis of power network loading due to fast charging of Electric Vehicles on highways” 
                d_ev_east = rd.uniform(25,120) #km total distance of the trip on the highway on east direction (positive) - values to be reviewed  
                # SoC_start = rd.uniform(0.2,0.9) #20-90%, F. H. Malik and M. Lehtonen, “Analysis of power network loading due to fast charging of Electric 
                d_ev_west = rd.uniform(-25,-120) #km total distance of the trip on the highway on west direction (negative)
                d_ev = rd.choice([d_ev_east, d_ev_west]) #we chose a random value between the west or east direction journey
                bat_cap = rd.uniform(15,25) #kWh, F. H. Malik and M. Lehtonen, “Analysis of power network loading due to fast charging of Electric Vehicles on highways” 

                destination_ev = nodes_position.iloc[i].values + d_ev #evaluation of the destination of the EV / select nearest node to exit from the highway

                delta_ev = 0

                if (destination_ev < 0) | (destination_ev > 276): #condition when EV is out of bound of the highway
                        
                    if destination_ev < 0: #when EV destination is out of west bound

                        delta_ev = nodes_position.iloc[i].values #ev journey will be equal to his entry point in space

                    if destination_ev > 276: #when EV destination is out of east bound

                        delta_ev = 276 - nodes_position.iloc[i].values #ev journey will be equal to the west bound and his entry point in space 
                        
                    SoC_final = SoC_start - delta_ev/bat_cap*ml_ev #SoC_final is depedent on delta_ev as indicated above
                        
                        
            #else calculate the normal SoC at the end of the ev journey, when it will rech his destination

                else: 
                    SoC_final = SoC_start - abs(d_ev)/(bat_cap*ml_ev) #final SoC when the vehicle ends its journey 


                if SoC_final<SoC_low: #recharge condition

                    t_shift = 0 #flag if vehicle is going to be charged during the next hour

                    if ((((nodes_position.iloc[i].values < stations.iloc[0].values) & (np.sign(d_ev) == -1))) | ((nodes_position.iloc[i].values > stations.iloc[-1].values) & (np.sign(d_ev) == 1))): #entry point is before first station or after last one
                        if min(abs(nodes_position.iloc[i].values - stations.values)) <= 5: #distance between entry point and first/last station less than 5 km      
                            E_charged = rd.uniform((delta_ev/ml_ev), 0.8*bat_cap)

                            dist_station = abs(stations.loc[:,'km'] - nodes_position.iloc[i].values) #find absolute distance between charging required position and station position
                            numbst = dist_station.idxmin(axis=0)  #evaluate nearest station by selecting the minimum distance  

                        else:
                            E_charged = 0

                            dist_station = abs(stations.loc[:,'km'] - nodes_position.iloc[i].values) #find absolute distance between charging required position and station position
                            numbst = dist_station.idxmin(axis=0)  #evaluate nearest station by selecting the minimum distance

                    else:
                                    
                        d_charge = ml_ev*bat_cap*(SoC_start - SoC_low) #distance that the vehicle can run until it needs to be charged (km)               
                        charge_position = np.sign(d_ev)*d_charge + nodes_position.iloc[i,:] #position when it will be needed to charge
                        dist_station = abs(stations.loc[:,'km'] - charge_position.values) #find absolute distance between charging required position and station position
                        numbst = dist_station.idxmin(axis=0)  #evaluate nearest station by selecting the minimum distance        
                                        
                                        #assess the station for which there is the minimum distance between charging station and charge_position (d_charge+node)

                        if delta_ev == 0:
                                        
                            E_charged = rd.uniform(((abs(d_ev) - d_charge)/ml_ev), 0.8*bat_cap) #energy charged is a random value between the energy required to end the journey and an higher charged value of 80% 

                        else:
                            E_charged = rd.uniform((delta_ev/ml_ev), 0.8*bat_cap) #energy charged between minimum to finish journey and 80%
                                        
                        d_ev_charge = stations.iloc[numbst].values - nodes_position.iloc[i].values #distance run from entering node to the charging station [km]

                        t_ev_charge = d_ev_charge/v  #time from entering node to the charging station [h]

                        if t_ev_charge >= 1:

                            t_shift = 1 #localization of the charging in time
                
                        e_stations.iloc[j+t_shift, numbst] = e_stations.iloc[j+t_shift, numbst] + E_charged #energy charged per station and time interval
                        t_recharge = E_charged*60/p_charge

                        t_charged.iloc[j+t_shift, numbst].append(t_recharge)  #time of charging 
                            
                        #t_recharge.iloc[z, numbst] = ((E_charged*60)/p_charge)

                        #t_charged.iloc[j+t_shift, numbst] = t_recharge.iloc[j+t_shift, z] #time of charging            
                        
                        #t_charged.at[j+t_shift, numbst] = 
    return e_stations, t_charged, nodes_in_ev #pay attention cause t_charged is sometimes coming out as list, but also as [array([])]
# In[309]:
    
t1 = time.time() 
e_stations_ldv, t_charged_ldv, nodes_in_ev = get_demand_static(p_ev = 0.1, nodes_in = nodes_in_ldv, traffic = traffic_ldv, stations = stations, hours = 8760,  p_charge = p_ldv, ml_ev = 5, v = v_ldv)
# e_stations_hdv, t_charged_hdv = get_demand_static(p_ev = 0.1, traffic = traffic_hdv, stations = stations, hours = 100,  p_charge = p_hdv, ml_ev = m_hdv, v = v_hdv)
# e_stations, t_charged, E_charged = get_demand_static_ldv()
t2 = time.time()
time_dataframe = (t2-t1)
print(t2-t1)
# get_demand_static_hdv()

#%%
EV_rec=pd.Series(index=traffic_ldv.index)

def n_ev():
    count_ev = 0
    for x in range (34):
        for y in range (10): #per un'ora
            count_ev = count_ev + int(nodes_in_ev.iloc[y,x])
    return count_ev
#print(len(t_charged.iloc[y,x]))
print("Number of elements in the list: ", n_ev())

def n_ev_chg():
    count = 0
    for x in range (5):
        
        for y in range (10): #per un'ora
            count = count + len(t_charged_ldv.iloc[y,x])
            EV_rec.iloc[y] = len(t_charged_ldv.iloc[y,x])
            # pct_charging 
    return count
print("Number of elements in the list: ", n_ev_chg())

hourly_ev = np.sum(nodes_in_ev, axis=1)
hourly_ev_pd = pd.Series(data=hourly_ev, index=traffic_ldv.index)
np.floor(hourly_ev_pd)

percentage_charging_hour = EV_rec/hourly_ev_pd
#%%
t_charged_ldv
#%%
# # %% visualization section
energy_daily = e_stations_ldv.groupby(e_stations_ldv.index.strftime("%m%d")).mean() #calculate the daily average for all the stations
energy_daily

#yearly average for each station
e_stations_ldv.mean().plot(kind='bar', xticks=[0, 1, 2, 3, 4], rot=0, xlim=5) 
plt.ylabel('Yearly Average Energy [kWh]')
plt.xlabel('Stations')
plt.grid(True)
plt.show()

energy_daily.plot(use_index=True, figsize=[40,15], legend=True, fontsize=20, linewidth=3)
plt.ylabel('Daily Average Energy [kWh]', fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.legend(loc=2, prop={'size': 25})
plt.grid(True)
plt.show()

station0=energy_daily.iloc[:,0]
station1=energy_daily.iloc[:,1]
station2=energy_daily.iloc[:,2]
station3=energy_daily.iloc[:,3]
station4=energy_daily.iloc[:,4]

station0.plot(use_index=True, figsize=[15,5], legend=False, fontsize=10, linewidth=2, title='station 0')
plt.ylabel('Daily Average Energy [kWh]', fontsize=10)
plt.grid(True)
plt.show()

station1.plot(use_index=True, figsize=[15,5], legend=False, fontsize=10, linewidth=2, title='station 1', color='darkorange')
plt.ylabel('Daily Average Energy [kWh]', fontsize=10)
plt.grid(True)
plt.show()

station2.plot(use_index=True, figsize=[15,5], legend=False, fontsize=10, linewidth=2, title='station 2', color='green')
plt.ylabel('Daily Average Energy [kWh]', fontsize=10)
plt.grid(True)
plt.show()

station3.plot(use_index=True, figsize=[15,5], legend=False, fontsize=10, linewidth=2, title='station 3', color='red')
plt.ylabel('Daily Average Energy [kWh]', fontsize=10)
plt.grid(True)
plt.show()

station4.plot(use_index=True, figsize=[15,5], legend=False, fontsize=10, linewidth=2, title='station 4', color='mediumpurple')
plt.ylabel('Daily Average Energy [kWh]', fontsize=10)
plt.grid(True)
plt.show()

energy_daily.boxplot()

percentage_charging_hour.plot(use_index=True, linewidth=2, figsize=[40,15], legend=False, fontsize=20, title='percentage of vehicle charging')
plt.ylabel('Average share of vehicle charging', fontsize=20)
plt.grid(True)

# %%
percentage_charging_daily = percentage_charging_hour.groupby(percentage_charging_hour.index.strftime("%m%d")).mean() #calculate the daily average for all the stations
percentage_charging_daily

# %%
percentage_charging_daily.plot(use_index=True, figsize=[30,10], title='daily')

# %%
