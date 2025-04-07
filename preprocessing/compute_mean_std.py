import os
import json
import numpy as np
import time
from tqdm import tqdm

np.set_printoptions(suppress=True)

print('compute min max for features...')

root = '../Dataset/datasets/datasets_grl/npy/spatiotemporal'

val_year = 2019
negative = 'clc'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dynamic_features = [
    '1 km 16 days NDVI',
    '1 km 16 days EVI',
    'ET_500m',
    'LST_Day_1km',
    'LST_Night_1km',
    'Fpar_500m',
    'Lai_500m',
    'era5_max_u10',
    'era5_max_v10',
    'era5_max_d2m',
    'era5_max_t2m',
    'era5_max_sp',
    'era5_max_tp',
    'era5_min_u10',
    'era5_min_v10',
    'era5_min_d2m',
    'era5_min_t2m',
    'era5_min_sp',
    'era5_min_tp',
    'era5_avg_u10',
    'era5_avg_v10',
    'era5_avg_d2m',
    'era5_avg_t2m',
    'era5_avg_sp',
    'era5_avg_tp',
    'smian',
    'sminx',
    'fwi',
    'era5_max_wind_u10',
    'era5_max_wind_v10',
    'era5_max_wind_speed',
    'era5_max_wind_direction',
    'era5_max_rh',
    'era5_min_rh',
    'era5_avg_rh',
]

static_features = [
    'dem_mean',
    'aspect_mean',
    'slope_mean',
    'roughness_mean',
    'roads_distance',
    'waterway_distance',
    'population_density',
]

clc_features = ['clc_' + str(c) for c in range(10)]

dict_d = {'min': {f: 0.0 for f in dynamic_features}, 'max': {f: 0.0 for f in dynamic_features},
          'mean': {f: 0.0 for f in dynamic_features}, 'std': {f: 0.0 for f in dynamic_features}}
dict_s = {'min': {f: 0.0 for f in static_features}, 'max': {f: 0.0 for f in static_features},
          'mean': {f: 0.0 for f in static_features}, 'std': {f: 0.0 for f in static_features}}
dict_c = {'min': {f: 0.0 for f in clc_features}, 'max': {f: 0.0 for f in clc_features},
          'mean': {f: 0.0 for f in clc_features}, 'std': {f: 0.0 for f in clc_features}}

dict = {}
dict['dynamic'] = dict_d
dict['static'] = dict_s
dict['clc'] = dict_c

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dirs = os.listdir(root)

dirs = [file for file in dirs if file == 'negatives_{}'.format(negative) or file == 'positives']

files_all = []

for dir in dirs:

    path = os.path.join(root, dir)

    files = os.listdir(path)

    files = [file for file in files if int(file[:4]) <= val_year]

    files.sort()

    files = [os.path.join(path, file[:-12]) for file in files if file.endswith('dynamic.npy')]

    files_all += files

print('number of samples: ', len(files_all))

time.sleep(1)

# spatiotemporal
for feature in ['static', 'clc', 'dynamic']:

    for i, f in tqdm(enumerate(dict[feature]['mean']), total=dict[feature]['mean'].__len__(), leave=True):

        if feature == 'static' or feature == 'clc':
            data_all = np.empty((len(files_all), 25, 25))
        else:
            data_all = np.empty((len(files_all), 10, 25, 25))

        for k, file in tqdm(enumerate(files_all), total=len(files_all), leave=False, position=0):

            if feature == 'static':
                data_all[k] = np.load(file + '_static.npy')[i, :, :]
            elif feature == 'clc':
                data_all[k] = np.load(file + '_clc_vec.npy')[i, :, :]
            else:
                data_all[k] = np.load(file + '_dynamic.npy')[:, i, :, :]

        min = np.nanmin(data_all)
        max = np.nanmax(data_all)
        mean = np.nanmean(data_all)
        std = np.nanstd(data_all, ddof=1)

        dict[feature]['min'][f] = min
        dict[feature]['max'][f] = max
        dict[feature]['mean'][f] = mean
        dict[feature]['std'][f] = std

with open(os.path.join(root, 'mean_std_train.json'), 'w') as j_file:

    json.dump(dict, j_file)




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


print('compute min max for features...')

root = '../Dataset/datasets/datasets_grl/npy/temporal'

val_year = 2019
negative = 'clc'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dynamic_features = [
    '1 km 16 days NDVI',
    '1 km 16 days EVI',
    'ET_500m',
    'LST_Day_1km',
    'LST_Night_1km',
    'Fpar_500m',
    'Lai_500m',
    'era5_max_u10',
    'era5_max_v10',
    'era5_max_d2m',
    'era5_max_t2m',
    'era5_max_sp',
    'era5_max_tp',
    'era5_min_u10',
    'era5_min_v10',
    'era5_min_d2m',
    'era5_min_t2m',
    'era5_min_sp',
    'era5_min_tp',
    'era5_avg_u10',
    'era5_avg_v10',
    'era5_avg_d2m',
    'era5_avg_t2m',
    'era5_avg_sp',
    'era5_avg_tp',
    'smian',
    'sminx',
    'fwi',
    'era5_max_wind_u10',
    'era5_max_wind_v10',
    'era5_max_wind_speed',
    'era5_max_wind_direction',
    'era5_max_rh',
    'era5_min_rh',
    'era5_avg_rh',
]

static_features = [
    'dem_mean',
    'aspect_mean',
    'slope_mean',
    'roughness_mean',
    'roads_distance',
    'waterway_distance',
    'population_density',
]

clc_features = ['clc_' + str(c) for c in range(10)]

dict_d = {'min': {f: 0.0 for f in dynamic_features}, 'max': {f: 0.0 for f in dynamic_features},
          'mean': {f: 0.0 for f in dynamic_features}, 'std': {f: 0.0 for f in dynamic_features}}
dict_s = {'min': {f: 0.0 for f in static_features}, 'max': {f: 0.0 for f in static_features},
          'mean': {f: 0.0 for f in static_features}, 'std': {f: 0.0 for f in static_features}}
dict_c = {'min': {f: 0.0 for f in clc_features}, 'max': {f: 0.0 for f in clc_features},
          'mean': {f: 0.0 for f in clc_features}, 'std': {f: 0.0 for f in clc_features}}

dict = {}
dict['dynamic'] = dict_d
dict['static'] = dict_s
dict['clc'] = dict_c

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

dirs = os.listdir(root)

dirs = [file for file in dirs if file == 'negatives_{}'.format(negative) or file == 'positives']

files_all = []

for dir in dirs:

    path = os.path.join(root, dir)

    files = os.listdir(path)

    files = [file for file in files if int(file[:4]) <= val_year]

    files.sort()

    files = [os.path.join(path, file[:-12]) for file in files if file.endswith('dynamic.npy')]

    files_all += files

print('number of samples: ', len(files_all))

time.sleep(1)

# temporal
for feature in ['static', 'clc', 'dynamic']:

    for i, f in tqdm(enumerate(dict[feature]['mean']), total=dict[feature]['mean'].__len__(), leave=True):

        if feature == 'static' or feature == 'clc':
            data_all = np.empty(len(files_all))
        else:
            data_all = np.empty((len(files_all), 10))

        for k, file in tqdm(enumerate(files_all), total=len(files_all), leave=False, position=0):

            if feature == 'static':
                data_all[k] = np.load(file + '_static.npy')[i]
            elif feature == 'clc':
                data_all[k] = np.load(file + '_clc_vec.npy')[i]
            else:
                data_all[k] = np.load(file + '_dynamic.npy')[:, i]

        min = np.nanmin(data_all)
        max = np.nanmax(data_all)
        mean = np.nanmean(data_all)
        std = np.nanstd(data_all, ddof=1)

        dict[feature]['min'][f] = min
        dict[feature]['max'][f] = max
        dict[feature]['mean'][f] = mean
        dict[feature]['std'][f] = std

with open(os.path.join(root, 'mean_std_train.json'), 'w') as j_file:

    json.dump(dict, j_file)
