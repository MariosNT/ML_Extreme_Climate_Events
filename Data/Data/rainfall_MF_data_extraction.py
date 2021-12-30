"""

"""
### Packages ###
import numpy as np
import xarray as xr
import pandas as pd
from numpy import linalg as LA
import matplotlib.pyplot as plt


### Dataset ###
dataset = "eobs_true_rainfall_197901-201907_uk.nc"
ds = xr.open_dataset(dataset)

#ds.close()

longitude = ds["rr"]["longitude"].values
latitude = ds["rr"]["latitude"].values
time = ds["rr"]["time"].values


cities_map = {
            "London": [51.5074, -0.1278],
            "Cardiff": [51.4816 + 0.15, -3.1791 -0.05],
            "Glasgow": [55.8642,  -4.2518],
            "Lancaster":[54.466, -2.8007],
            "Bradford": [53.7960, -1.7594],
            "Manchester":[53.4808, -2.2426],
            "Birmingham":[52.4862, -1.8904],
            "Liverpool":[53.4084 , -2.9916 +0.1 ],
            "Leeds":[ 53.8008, -1.5491 ],
            "Edinburgh": [55.9533, -3.1883],
            "Belfast": [54.5973, -5.9301],
            "Dublin": [53.3498, -6.2603],
            "LakeDistrict":[54.4500,-3.100],
            "Newry":[54.1751, -6.3402],
            "Preston":[53.7632, -2.7031 ],
            "Truro":[50.2632, -5.0510],
            "Bangor":[54.2274 - 0, -4.1293 - 0.3],
            "Plymouth":[50.3755 + 0.1, -4.1427],
            "Norwich": [52.6309, 1.2974],
            "StDavids":[51.8812+0.05, -5.2660+0.05] ,
            "Swansea":[51.6214+0.05,-3.9436],
            "Lisburn":[54.5162,-6.058],
            "Salford":[53.4875, -2.2901],
            "Aberdeen":[57.1497,-2.0943-0.05],
            "Stirling":[56.1165, -3.9369],
            "Hull":[53.7676+0.05, 0.3274]
            }

def city_coordinates(city_name, long_coord, lat_coord):
    long_city = cities_map[city_name][1]
    lat_city = cities_map[city_name][0]

    # We subtract the cities coordinates from the standard coordinates
    # then we find the minimum value, that corresponds to the closest distance
    dif_array_long = np.abs(long_coord[:]-long_city)
    dif_array_lat = np.abs(lat_coord[:]-lat_city)
    long_min = np.min(dif_array_long)
    lat_min = np.min(dif_array_lat)
    long_index = np.argwhere(dif_array_long==long_min)[0]
    lat_index = np.argwhere(dif_array_lat==lat_min)[0]
    
    return long_index[0], lat_index[0], long_city, lat_city


def rainfall_city_time(city_name, long_coord, lat_coord, time_start, time_end):
    long_key , lat_key, _, _ = city_coordinates(city_name, long_coord, lat_coord)
    long = longitude[long_key]
    lat = latitude[lat_key]
    data_city = ds["rr"].sel(latitude=lat, longitude=long)
    data_city = data_city.sel(time=slice("{}-01-01".format(time_start), "{}-12-31".format(time_end)))
    # data_city.plot.line(hue='lat', marker="o", color="grey",\
    #                 markerfacecolor="purple", alpha=0.2, linestyle = 'None')
    # plt.title("{}".format(city_name))
    # plt.show()
    return data_city


Rainfall_Cardiff_2000 = rainfall_city_time("Cardiff", longitude, latitude, 2000, 2000)
Rainfall_Cardiff_2006 = rainfall_city_time("Cardiff", longitude, latitude, 2006, 2006)
Rainfall_Cardiff_2012 = rainfall_city_time("Cardiff", longitude, latitude, 2012, 2012)
Rainfall_Cardiff_2018 = rainfall_city_time("Cardiff", longitude, latitude, 2018, 2018)

np.save("Rainfall_Cardiff_2000.npy", Rainfall_Cardiff_2000)
np.save("Rainfall_Cardiff_2006.npy", Rainfall_Cardiff_2006)
np.save("Rainfall_Cardiff_2012.npy", Rainfall_Cardiff_2012)
np.save("Rainfall_Cardiff_2018.npy", Rainfall_Cardiff_2018)


model_fields_dataset = "model_fields_1979-2019.nc"
model_fields = xr.open_dataset(model_fields_dataset)

# model_fields_HD_dataset = "model_fields_linearly_interpolated_1979-2019.nc"
# model_fields_HD = xr.open_dataset(model_fields_HD_dataset)


longitude = model_fields["longitude"].values
latitude = model_fields["latitude"].values

dt_phys = pd.date_range('1979-01-01', '2020-01-01', freq='6H')[:-1]

model_fields["time"] = dt_phys

def geophysical_city_time(city_name, long_coord, lat_coord, time_start, time_end):
    long_key , lat_key, _, _ = city_coordinates(city_name, long_coord, lat_coord)
    long = longitude[long_key]
    lat = latitude[lat_key]
    
    
    mf_time_slice = model_fields.sel(time=slice("{}-01-01".format(time_start), "{}-12-31".format(time_end)))
    mf_time_slice = mf_time_slice.sel(latitude=latitude[lat_key], longitude=longitude[long_key])
    time_points = len(mf_time_slice["time"])
    if time_points%4 != 0:
        print("Error in days")
    else:
        days = time_points//4 #since we have 4 time points per day. We need int for the arrays below.
    
    ### Reading the model fields in the specific time slice 
    geopotential = mf_time_slice["geopotential"].values
    y_wind = mf_time_slice["y_wind"].values
    x_wind = mf_time_slice["x_wind"].values
    water_vapour = mf_time_slice["unknown_local_param_137_128"].values
    humidity = mf_time_slice["unknown_local_param_133_128"].values
    air_temperature = mf_time_slice["air_temperature"].values
    
    ### Finding the average fields values per day
    geopotential_timeseries = np.average(geopotential.reshape((days,4)), axis=1)
    y_wind_timeseries = np.average(y_wind.reshape((days,4)), axis=1)
    x_wind_timeseries = np.average(x_wind.reshape((days,4)), axis=1)
    water_vapour_timeseries = np.average(water_vapour.reshape((days,4)), axis=1)
    humidity_timeseries = np.average(humidity.reshape((days,4)), axis=1)
    air_temperature_timeseries = np.average(air_temperature.reshape((days,4)), axis=1)
    
    time_range = pd.date_range("{}-01-01".format(time_start), "{}-12-31".format(time_end), freq='1D')
    
    ### Putting them all together
    # We return 6 model fields (each row of X denotes a field, with legth - N columns - depending on the number of days)
    X = np.vstack((geopotential_timeseries, y_wind_timeseries, x_wind_timeseries, water_vapour_timeseries, humidity_timeseries, air_temperature_timeseries))
    
    # Calculating transpose such that each row corresponds for a day
    # X dimensions (days, model_fields) 
    X = X.T
    
    ### Calculating windspeed and consider that as a variable (we use that, instead of x_wind/y_wind)
    ### After adding the windspeed, our fields are:
    # 0: geopotential
    # 1: water_vapour
    # 2: humidity
    # 3: air_temperature
    # 4: windspeed
    X = np.concatenate((X[:,[0,3,4,5]],np.sqrt(pow(X[:,1],2)+pow(X[:,2],2)).reshape(-1,1)), axis=1)
    
    
    
    ### Standardize data (making each column having 0 mean and stdev 1)
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    
    return X, time_range

X_Cardiff, time_period = geophysical_city_time("Cardiff", longitude, latitude, 1979, 1979)
plt.plot(time_period, X_Cardiff[:,4])
plt.show()


X_Cardiff_2000, _ = geophysical_city_time("Cardiff", longitude, latitude, 2000, 2000)
X_Cardiff_2006, _ = geophysical_city_time("Cardiff", longitude, latitude, 2006, 2006)
X_Cardiff_2012, _ = geophysical_city_time("Cardiff", longitude, latitude, 2012, 2012)
X_Cardiff_2018, _ = geophysical_city_time("Cardiff", longitude, latitude, 2018, 2018)

np.save("model_fields_Cardiff_2000.npy", X_Cardiff_2000)
np.save("model_fields_Cardiff_2006.npy", X_Cardiff_2006)
np.save("model_fields_Cardiff_2012.npy", X_Cardiff_2012)
np.save("model_fields_Cardiff_2018.npy", X_Cardiff_2018)