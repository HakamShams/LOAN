import numpy
import numpy as np
import os
import json

from torch.utils.data import Dataset
np.set_printoptions(suppress=True)


class FireCubeLoader(Dataset):
    """
    Dataloader for FireCube Dataset

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says
    name : str
        the name of the animal
    sound : str
        the sound that the animal makes
    num_legs : int
        the number of legs the animal has (default 4)

    Methods
    -------
    get_time(index)
        Helper method to get the year, month, and day from the file name

    get_xy(index)
        Helper method to get the coordinate x, y from the file name

    __loadjson_statistic(root)
        Private method to get the statistics of the dataset from the root directory

    __getpath(root)
        Private method to get the files names with corresponding labels of the dataset inside the root directory

    __loadnpy(npy_file)
        Private method to get data by file path

    min_max_scale(array, min_alt, max_alt, min_new, max_new)
        Helper method to normalize an array between new minimum and maximum values

    generate_conditional_map(data_mode)
        Helper method to convert CORINE land cover map into a map of 10 main classes (reduce the number of classes)

    """

    def __init__(self, root: str, mode: str = 'train', is_scale: bool = True, neg_pos_ratio: int | None = 2, val_year: int = 2019,
                 negative: str = 'clc', nan_fill: float = 0., is_aug: bool = False, is_shuffle: bool = True,
                 lag: int = 10, static_features: list = None, dynamic_features: list = None, clc_features: list = None,
                 seed: int = 0):

        super(FireCubeLoader, self).__init__()

        """
        Parameters
        ----------
        root : str
            Directory of the dataset
        mode : str
            training, val, or test (default train)
        is_scale : bool (default True)
            option to scale the data
        neg_pos_ratio: int, optional
            ratio of negative samples to positives (default 2)
        val_year : int (default 2019)
            year used for validation
            all years after this year will be considered for testing and before it will be considered as training  
        negative : str (default clc)
            whether to take negative samples randomly or with stratified strategy according to land cover
        nan_fill : float (default 0.)
            value to replace missing values
        is_aug : bool (default False)
            option to use data augmentation
        is_shuffle : bool (default True)
            option to shuffle the data
        lag : int (default 10)
            number of days
        static_features : list (default None)
            list of static features names
        dynamic_features : list (default None)
            list of dynamic features names
        clc_features : list (default None)
            list of fraction classes names
        seed : int (default 0)
            random seed
        """

        self.root = root
        self.mode = mode
        self.lag = lag
        self.is_scale = is_scale
        self.is_aug = is_aug
        self.shuffle = is_shuffle
        self.nan_fill = nan_fill
        self.negative = negative
        self.val_year = val_year
        self.neg_pos_ratio = neg_pos_ratio

        self.seed = seed

        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.clc_features = clc_features

        self.__leap_year = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

        # check the input
        assert mode in ['train', 'test', 'val']
        assert negative in ['clc', 'random']

        self.__getpath(self.root)
        self.__loadjson_statistic(root)

        if is_shuffle:
            np.random.shuffle(self.files)

    def get_time(self, index: int):

        """

        Helper method to get the year, month, and day from the file name

        Parameters
        ----------
        index : int
            The index of the file

        Returns
        ----------
        year: int, month: int, day: int

        """

        assert index < len(self.files)

        name = os.path.basename(os.path.normpath(self.files[index][0]))

        return int(name[:4]), int(name[4:6]), int(name[6:8])

    def get_xy(self, index):

        """

        Helper method to get the coordinate x, y from the file name

        Parameters
        ----------
        index : int
            The index of the file

        Returns
        ----------
        x: int, y: int

        """

        assert index < len(self.files)

        name = os.path.basename(os.path.normpath(self.files[index][0]))[9:]
        ind = name.find('_')

        return int(name[:ind]), int(name[ind+1:])

    def __loadnpy(self, npy_file):

        """

        Private method to get data by file path

        Parameters
        ----------
        npy_file : str
            path to file

        Returns
        ----------
        array_dynamic : numpy array
            dynamic features
        array_static : numpy array
            static features
        array_clc_vec : numpy array
            fraction of classes
        array_clc_mode : numpy array
            CORINE land cover

        """

        array_dynamic = np.load(npy_file + '_dynamic.npy')[:, self.dynamic_features_ind, :, :]
        array_static = np.load(npy_file + '_static.npy')[self.static_features_ind, :, :]
        array_clc_vec = np.load(npy_file + '_clc_vec.npy')[self.clc_features_ind, :, :]
        #array_clc_mode = np.load(npy_file + '_clc_mode.npy')

        # flip to get first day in index 0
        array_dynamic = np.flip(array_dynamic, axis=0)
        # get the lag days
        array_dynamic = array_dynamic[:self.lag, :, :, :]
        # swap days and feature axis
        array_dynamic = np.swapaxes(array_dynamic, 0, 1)

        return array_dynamic, array_static, array_clc_vec#, array_clc_mode

    def __loadjson_statistic(self, root):

        """

        Private method to get the statistics of the dataset from the root directory

        Parameters
        ----------
        root : str
            Directory to dataset

        """

        with open(os.path.join(root, 'mean_std_train.json'), 'r') as file:

            dict = json.load(file)

            self.min_static, self.min_dynamic, self.min_clc = [], [], []
            self.max_static, self.max_dynamic, self.max_clc = [], [], []

            self.mean_static, self.mean_dynamic, self.mean_clc = [], [], []
            self.std_static, self.std_dynamic, self.std_clc = [], [], []

            self.static_features_ind, self.dynamic_features_ind, self.clc_features_ind = [], [], []

            for feature in ['static', 'dynamic', 'clc']:
                for i, f in enumerate(dict[feature]['mean']):
                    if feature == 'static':
                        if f in self.static_features:
                            self.min_static.append(dict[feature]['min'][f])
                            self.max_static.append(dict[feature]['max'][f])
                            self.mean_static.append(dict[feature]['mean'][f])
                            self.std_static.append(dict[feature]['std'][f])
                            self.static_features_ind.append(i)
                    elif feature == 'clc':
                        if f in self.clc_features:
                            self.min_clc.append(dict[feature]['min'][f])
                            self.max_clc.append(dict[feature]['max'][f])
                            self.mean_clc.append(dict[feature]['mean'][f])
                            self.std_clc.append(dict[feature]['std'][f])
                            self.clc_features_ind.append(i)
                    else:
                        if f in self.dynamic_features:
                            self.min_dynamic.append(dict[feature]['min'][f])
                            self.max_dynamic.append(dict[feature]['max'][f])
                            self.mean_dynamic.append(dict[feature]['mean'][f])
                            self.std_dynamic.append(dict[feature]['std'][f])
                            self.dynamic_features_ind.append(i)

    def __getpath(self, root):

        """

        Private method to get the files names with corresponding labels of the dataset inside the root directory

        Parameters
        ----------
        root : str
            Directory to dataset

        """

        dirs = [['positives', 1], ['negatives_{}'.format(self.negative), 0]]

        self.files = []

        for dir, label in dirs:

            path = os.path.join(root, dir)

            files = os.listdir(path)

            if dir != 'negatives_random':
                if self.mode == 'train':
                    files = [file for file in files if int(file[:4]) < self.val_year]
                elif self.mode == 'val':
                    files = [file for file in files if int(file[:4]) == self.val_year]
                else:
                    files = [file for file in files if int(file[:4]) > self.val_year]

            files.sort()

            files = [(os.path.join(path, file[:-12]), label) for file in files if file.endswith('dynamic.npy')]

            if self.neg_pos_ratio and label < 1:

                assert len(self.files) * self.neg_pos_ratio <= len(files)

                np.random.seed(self.seed)

                indices_random = np.random.choice(len(files), len(self.files) * self.neg_pos_ratio, replace=False)

                files = [files[index] for index in indices_random]

            self.files += files

    def min_max_scale(self, array: numpy.array,
                      min_alt: float, max_alt: float,
                      min_new: float = 0., max_new: float = 1.):

        """
        Helper method to normalize an array between new minimum and maximum values

        Parameters
        ----------
        array : numpy array
            array to be normalized
        min_alt : float
            minimum value in array
        max_alt : float
            maximum value in array
        min_new : float
            minimum value after normalization
        max_new : float
            maximum value after normalization

        Returns
        ----------
       array : numpy array
            normalized numpy array

        """

        array = ((max_new - min_new) * (array - min_alt) / (max_alt - min_alt)) + min_new

        return array

    def generate_conditional_map(self, data_mode):

        """

        Helper method to convert CORINE land cover map into a map of 10 main classes (reduce the number of classes)

        Parameters
        ----------
        data_mode : numpy array
            land cover array

        Returns
        ----------
        data_mode : numpy array
            land cover array with the main 10 classes

        """

        data_mode[np.logical_and(data_mode <= 11, data_mode != 2)] = 0  # artificial_surfaces
        data_mode[data_mode == 2] = 1  # discontinuous_urban
        data_mode[np.logical_and(data_mode <= 14, data_mode > 11)] = 2  # arable_land
        data_mode[np.logical_and(data_mode <= 17, data_mode > 14)] = 3  # permanent_crops
        data_mode[data_mode == 18] = 4  # pastures
        data_mode[np.logical_and(data_mode <= 22, data_mode > 18)] = 5  # general_agricultural
        data_mode[np.logical_and(data_mode <= 25, data_mode > 22)] = 6  # forest
        data_mode[np.logical_and(data_mode <= 29, data_mode > 25)] = 7  # scrub and/or herbaceous associations
        data_mode[np.logical_and(data_mode <= 34, data_mode > 29)] = 8  # open spaces with little or no vegetation
        data_mode[data_mode >= 35] = 9  # water bodies

        # can also be normalized
        # data_mode = data_mode / 9

        return data_mode


    def __getitem__(self, index):

        """

        Method to get data by index

        Parameters
        ----------
        index : int
          The index of the file

        Returns
        ----------
        data_static : numpy array
            static features
        data_dynamic : numpy array
            dynamic features
        data_clc : numpy array
            fraction of classes
        label : numpy array
            labels (positive 1, negative 0)
        data_mode : numpy array
            land cover with the main 10 classes
        data_time : int
            day of the year

        """
        # get file names and label
        file_path, label = self.files[index]

        # get data
        #data_dynamic, data_static, data_clc, data_mode = self.__loadnpy(file_path)
        data_dynamic, data_static, data_clc = self.__loadnpy(file_path)

        # scale data
        if self.is_scale:
            for d in range(len(data_dynamic)):
                #data_dynamic[d, :, :, :] = (data_dynamic[d, :, :, :] - self.mean_dynamic[d]) / self.std_dynamic[d]
                data_dynamic[d, :, :, :] = self.min_max_scale(data_dynamic[d, :, :, :], self.min_dynamic[d], self.max_dynamic[d])

            for s in range(len(data_static)):
                #data_static[s, :, :] = (data_static[s, :, :] - self.mean_static[s]) / self.std_static[s]
                data_static[s, :, :] = self.min_max_scale(data_static[s, :, :], self.min_static[s], self.max_static[s])

            #for c in range(len(data_clc)):
            #    data_clc[c] = (data_clc[c] - self.mean_clc[c]) / self.std_clc[c]

        # augment data
        if self.is_aug:

            is_noise = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])

            if is_noise:
                data_static = data_static + np.random.randn(data_static.shape[0],
                              data_static.shape[1],
                              data_static.shape[2]).astype(np.float64) * 0.01

                data_dynamic = data_dynamic + np.random.randn(data_dynamic.shape[0],
                              data_dynamic.shape[1],
                              data_dynamic.shape[2],
                              data_dynamic.shape[3]).astype(np.float64) * 0.01

            is_rotate = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])

            if is_rotate:
                k = np.random.randint(1, 4)
                data_static = np.rot90(data_static, k=k, axes=(-1, -2))
                data_dynamic = np.rot90(data_dynamic, k=k, axes=(-1, -2))
                data_clc = np.rot90(data_clc, k=k, axes=(-1, -2))
                #data_mode = np.rot90(data_mode, k=k, axes=(-1, -2))

            is_flip = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])

            if is_flip:
                ax = np.random.randint(1, 2)
                data_static = np.flip(data_static, axis=-ax)
                data_dynamic = np.flip(data_dynamic, axis=-ax)
                data_clc = np.flip(data_clc, axis=-ax)
                #data_mode = np.flip(data_mode, axis=-ax)

            is_transpose = np.random.choice(np.arange(0, 2), p=[0.5, 0.5])

            if is_transpose:
                data_static = np.swapaxes(data_static, -1, -2)
                data_dynamic = np.swapaxes(data_dynamic, -1, -2)
                data_clc = np.swapaxes(data_clc, -1, -2)
                #data_mode = np.swapaxes(data_mode, -1, -2)


        # fill in missing values data
        data_static[np.isnan(data_static)] = self.nan_fill
        data_dynamic[np.isnan(data_dynamic)] = self.nan_fill
        data_clc[np.isnan(data_clc)] = self.nan_fill

        data_dynamic = data_dynamic.astype(np.float64)
        data_static = data_static.astype(np.float64)
        data_clc = data_clc.astype(np.float64)

        label = np.array(label)

        # generate conditional map based on semantic map (like SPADE paper)
        #data_mode = self.generate_conditional_map(data_mode)

        # get day of the year
        year, month, day = self.get_time(index)
        data_time = self.__leap_year[month - 1] + day

        #return data_static.copy(), data_dynamic.copy(), data_clc.copy(), label.copy(), data_mode.copy(), data_time
        return data_static.copy(), data_dynamic.copy(), data_clc.copy(), label.copy(), data_time

    def __len__(self):
        """
        get number of files
        """
        return len(self.files)



if __name__ == '__main__':

    root = '/home/shams/Projects/FireCube/Dataset/datasets/datasets_grl/npy/spatiotemporal'

    dynamic_features = [
        '1 km 16 days NDVI',
        'LST_Day_1km',
        'LST_Night_1km',
        'era5_max_d2m',
        'era5_max_t2m',
        'era5_max_sp',
        'era5_max_tp',
        'sminx',
        'era5_max_wind_speed',
        'era5_min_rh'
    ]

    static_features = [
        'dem_mean',
        'slope_mean',
        'roads_distance',
        'waterway_distance',
        'population_density',
    ]

    clc_features = ['clc_' + str(c) for c in range(10)]

    data = FireCubeLoader(root=root, mode='train', is_scale=True, neg_pos_ratio=2, val_year=2019, negative='clc',
                          nan_fill=0.0, is_aug=True, is_shuffle=False, lag=10, dynamic_features=dynamic_features,
                          static_features=static_features, clc_features=clc_features, seed=0)

    print('number of sampled data:', data.__len__())
    print('static data [0] shape:', data.__getitem__(0)[0].shape)
    print('dynamic data [0] shape:', data.__getitem__(0)[1].shape)
    print('clc data [0] shape:', data.__getitem__(0)[2].shape)
    print('label [0]:', data.__getitem__(0)[3])
    print('mode data [0] shape:', data.__getitem__(0)[4].shape)
    print('time [0]:', data.__getitem__(0)[5])

    # check
    is_test_run = False
    is_plot = False

    if is_test_run:

        import torch
        import time
        import random

        manual_seed = 0
        random.seed(manual_seed)

        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        train_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True, pin_memory=False)

        end = time.time()

        for i, (data_static, data_dynamic, data_clc, label, data_time) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()

    if is_plot:

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib
        import matplotlib.pyplot as plt

        #matplotlib.use('TkAgg')  # Ubutu

        colormap_s = ['terrain', 'viridis', 'nipy_spectral', 'turbo', 'jet']
        colormap_d = ['Spectral_r', 'Reds', 'Reds', 'PuBuGn', 'Blues', 'Spectral', 'PRGn', 'PRGn', 'PuBuGn', 'PuBuGn']
        colormap_c = ['PuRd', 'Purples', 'YlOrBr', 'Oranges', 'copper', 'pink', 'Greens', 'Greens',
                      'Greens', 'Blues']

        classes = ['negative', 'positive']

        for i in range(len(data)):

            #i = 500  # 2020-08-21_y36.72_x22.43.h5
            #i = np.random.choice(len(data), 1, replace=False)

            data_static, data_dynamic, data_clc, label, _ = data[int(i)]
            year, month, day = data.get_time(int(i))

            long, lat = data.get_xy(int(i))

            print("Time: {}-{}-{} _ {} class".format(year, month, day, classes[label]))

            colormap_sc = colormap_s + colormap_c + colormap_d

            data_static = np.repeat(data_static[:, np.newaxis, :, :], len(dynamic_features), axis=1)
            data_clc = np.repeat(data_clc[:, np.newaxis, :, :], len(dynamic_features), axis=1)

            data_features = np.concatenate((data_static, data_clc, data_dynamic), axis=0)

            names_features = static_features + clc_features + dynamic_features

            for feature_ind in range(data_features.shape[0]):

                #feature_ind = 16

                data_f = data_features[feature_ind, ...]

                fig = plt.figure()
                ax = plt.axes(projection="3d")

                norm = matplotlib.colors.Normalize(vmin=np.min(data_f), vmax=np.max(data_f))

                data_f = np.swapaxes(data_f, -2, -1)
                data_f = np.swapaxes(data_f, 0, 1)

                colors = plt.cm.get_cmap(colormap_sc[feature_ind])(norm(data_f))

                #cube = ax.voxels(np.ones_like(data_f), facecolors=colors, alpha=1, edgecolors='w', linewidth=0.2)
                cube = ax.voxels(np.ones_like(data_f), facecolors=colors, alpha=0.9)

                m = matplotlib.cm.ScalarMappable(cmap=plt.cm.get_cmap(colormap_sc[feature_ind]), norm=norm)
                m.set_array([])
                cbar = plt.colorbar(m)
                cbar.set_label(names_features[feature_ind])

                plt.tick_params(
                    axis='x',
                    which='both',
                    bottom=False,
                    top=False,
                    labelbottom=False

                    )

                plt.tick_params(
                    axis='z',
                    which='both',
                    left=False, labelleft=False,
                    right=False, labelright=False,
                    bottom=False, labelbottom=False,
                    top=False, labeltop=False
                    )

                ax.set(ylabel='time', xlabel='long', zlabel='lat')
                title = 'feature=' + names_features[feature_ind] + ' - time=' + str(year) + '-' + str(month) + '-' + str(day) + \
                        ' - x={}, y={}'.format(long, lat) + ' - {} class'.format(classes[label])
                plt.title(title)
                #plt.axis('off')

                ticks_labels = []
                for t in range(data_f.shape[1]):
                    if t == 0:
                        ticks_labels.append('t')
                    else:
                        ticks_labels.append('t-' + str(t))

                #plt.style.use('dark_background')
                ax.set_yticklabels(ticks_labels)
                ax.set_yticks(np.arange(data_f.shape[1]) + 0.5)
                ax.grid(False)

                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')

                ax.tick_params(axis='x', colors='w')
                ax.tick_params(axis='z', colors='w')

                for axis in [ax.xaxis, ax.zaxis]:
                    axis.set_ticks([])

                #manager = plt.get_current_fig_manager()
                #manager.resize(*manager.window.maxsize())
                #manager.full_screen_toggle()

                #fig.savefig('../images/sample_{}.png'.format(static_features[feature_ind]), format='png', dpi=1200)

                plt.show()
