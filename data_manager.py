import os
import re

import h5py
import pandas as pd

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PHANTOM_DB_FOLDER_NAME = 'database'
PARAMS_DB_FOLDER_NAME = 'optimal_params_db'
BIG_PARAM_DATA_FOLDER = 'big_param_data'

def get_params_from_dataframe(df, indx):
    blobns = df['blobiness'][indx]
    porosity = df['porosity'][indx]
    num_of_angles = df['num_of_angles'][indx]
    noise_info = df['noise'][indx]
    dim = df['dimension'][indx]

    return porosity, blobns, noise_info, num_of_angles, dim


def generate_phantom_file_name(porosity, blobns, noise_info, num_of_angles, dim):

    return f'por{porosity}_blob{blobns}_noise{noise_info}_angl{num_of_angles}_dim{dim}.h5'


def _get_params_from_phantom_file_name(file_path):
    attrs = re.split("_", file_path)
    porosity = attrs[0][3:]
    blobiness = attrs[1][4:]
    noise = attrs[2][5:]
    num_of_angles = attrs[3][4:]

    return porosity, blobiness, noise, num_of_angles


def _add_folder(current_directory, folder_name):
    save_path = os.path.join(current_directory, folder_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    return save_path


def _sort_df(df):
    df[['noise','blobiness','porosity']] = df[['noise','blobiness','porosity']].astype(float)
    df = df.sort_values(['porosity','noise','blobiness'], ascending=False)
    df[['blobiness']] = df[['blobiness']].astype(int)
    df[['noise']] = df[['noise']].astype(int)
    df[['porosity','noise','blobiness']] = df[['porosity','noise','blobiness']].astype(str)
    df = df.reset_index(drop=True)
    return df


def save(phantom, porosity, blobns, noise_info, num_of_angles, tag='original'):
    '''
    Saves phantom to the database.
        
    Parameters:
    -----------
    phantom: ndarray.
        phantom
    porosity: float.
        Phantom's porosiity
    
    blobns: int.
        Phantom's blobiness
    
    noise_info: float or str.
        Information about the noise (e.g. probability)
    
    num_of_angles: int.
        Number of Radon projections
    
    tag: str.
        Specifies phantom intent
    
    results:
    --------
    out: phantoms.h5 file
        See in the  script's directory in the folder 'database'
    '''

    db_folder = _add_folder(SCRIPT_PATH, PHANTOM_DB_FOLDER_NAME)

    dim = phantom.ndim
    db_name = generate_phantom_file_name(porosity, blobns, noise_info, num_of_angles, dim)
    db_path = os.path.join(db_folder, db_name)


    with h5py.File(db_path, 'a') as hdf:
        try:
            hdf.create_dataset(tag, data = phantom, compression='gzip', compression_opts=0)
        except BaseException:
            del hdf[tag]
            hdf.create_dataset(tag, data = phantom, compression='gzip', compression_opts=0)


def get_all_data_info():
    '''
    shows table with existed phantom content: dimension, id_inx, tags,
    and phantom at: porosity, number of angles, blobiness, noise info
    Returns:
    --------
    out: pandas.core.frame.DataFrame
        Table format. Use .head() to see first 5 rows.
    '''
    folder_path = os.path.join(SCRIPT_PATH, PHANTOM_DB_FOLDER_NAME)
    h5_files = os.listdir(folder_path)

    df = pd.DataFrame()

    for h5_name in h5_files:
        attrs = re.split("_", h5_name)
        file_path = os.path.join(folder_path, h5_name)

        with h5py.File(file_path, 'r') as hdf:
            tags = list(hdf)
        dim_data = {
                    'porosity': attrs[0][3:],
                    'blobiness': attrs[1][4:],
                    'noise': attrs[2][5:],
                    'num_of_angles': attrs[3][4:],
                    'dimension': attrs[4][3],
                    'tags': tags
                    }

        df = df.append(dim_data, ignore_index=True)
    
    return _sort_df(df)


def get_data(indx: int, 
            tag: str):
    '''
    Use this function to get needed data for ML process.
    Use show_data_info function to find out dimension, id_inx and tags, which exist.
    Parameters:
    -----------
    id_indx: 1,2,3,etc.
        id for phantom with certain porosity, blobiness and experiment parameters
    tag: 'test', 'train' or another
        This parameter controls conflicts if several csv files are generated for 1 phanom.
    Returns:
    ----------
    out: pandas.core.frame.DataFrame or ndarray.
        Phantom
    '''

    df = get_all_data_info()

    h5_name = generate_phantom_file_name(*get_params_from_dataframe(df, indx))
    file_path = os.path.join(SCRIPT_PATH, PHANTOM_DB_FOLDER_NAME, h5_name)

    with h5py.File(file_path, 'r') as hdf:
        dataset = hdf.get(tag)
        dataset = dataset[()]

    return dataset


def save_plot(figure, folder_name, name):
    save_path = os.path.join(SCRIPT_PATH, 'plots')
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, folder_name)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, name)

    if os.path.isfile(save_path):
        os.remove(save_path)
    figure.savefig(save_path)


def save_params(phantom_ind: int,
                filter_name: str,
                **kwargs
                ):
    df = get_all_data_info()
    pf_name = generate_phantom_file_name(*get_params_from_dataframe(df, phantom_ind))

    all_params_folder = _add_folder(SCRIPT_PATH, PARAMS_DB_FOLDER_NAME)
    phantom_params_folder = _add_folder(all_params_folder, pf_name)
    db_path = os.path.join(phantom_params_folder, filter_name)

    rve_shape = get_data(phantom_ind, 'rve_bin').shape

    with h5py.File(db_path, 'a') as hdf:
        hdf.attrs['rve_shape'] = rve_shape
        for key, value in kwargs.items():
            hdf.attrs[key] = value

    return list(kwargs.keys())


def add_csv_and_xlsx(df, csv_file_name, sort=True):
    if sort:
        df = _sort_df(df)
    csv_name = csv_file_name + '.csv'
    save_path = _add_folder(SCRIPT_PATH, BIG_PARAM_DATA_FOLDER)
    save_path_csv = os.path.join(save_path, csv_name)

    xlsx_name = csv_file_name + '.xlsx'
    save_path_xlxs = os.path.join(save_path, xlsx_name)

    df.to_excel(save_path_xlxs, index=False, header=True)
    df.to_csv(save_path_csv)


def get_all_params_info(
            saved_params_columns_order=None,
            csv_file_name = 'optimal_params_for_porespy_phants_500x500x500_real'
            ):
    
    main_columns_order=['porosity', 
                        'blobiness',
                        'noise',
                        'num_of_angles',
                        'rve_shape',
                        'filter_name']

    all_params_folder = os.path.join(SCRIPT_PATH, PARAMS_DB_FOLDER_NAME)
    sample_folders = os.listdir(all_params_folder)

    df = pd.DataFrame()

    for sample_folder in sample_folders:
        por, blob, noise, num_of_angl = _get_params_from_phantom_file_name(sample_folder)

        sample_path = os.path.join(all_params_folder, sample_folder)
        sample_filters = os.listdir(sample_path)

        for sample_filter in sample_filters:
            file_path = os.path.join(sample_path, sample_filter)

            with h5py.File(file_path, 'r') as hdf:
                params_and_metrics = dict(hdf.attrs)
                if saved_params_columns_order:
                    columns_order = main_columns_order + saved_params_columns_order
                else:
                    columns_order = main_columns_order + list(params_and_metrics.keys())

            dim_data = {
                        'porosity': por,
                        'blobiness': blob,
                        'noise': noise,
                        'num_of_angles': num_of_angl,
                        'filter_name': sample_filter
                        }
            dim_data.update(params_and_metrics)
            df = df.append(dim_data, ignore_index=True)

    df = df[columns_order]

    add_csv_and_xlsx(df, csv_file_name)
    df = _sort_df(df)
    return df


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER_NAME = 'dataframes'
DEFAULT_SEP = ";"
DEFAULT_DECIMAL = ","

def save_dataframe(df, file_name):
    """ сохраняет csv файл
    df : pandas.Dataframe
        таблица данных
    file_name : str
        Название csv или xlsx файла в который нужно сохранить df
    
    Returns
    -------
    """

    file_path = os.path.join(SCRIPT_PATH, DATA_FOLDER_NAME, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    if file_name[-3:]=="csv":
        df.to_csv(file_path, sep=DEFAULT_SEP, decimal=DEFAULT_DECIMAL)
    elif file_name[-3:]=="lsx":
        df.to_excel(file_path)
    else:
        raise ValueError(f"file_name must consist .csv or .xlsx")