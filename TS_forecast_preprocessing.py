import sklearn
from tsai.basics import *

# verified to work 2/7/24
def find_datetime_col(df):
    datetime_col_index = None
    for i in range(len(df.columns)):
        dtype=df.iloc[:, i].dtype
        if 'datetime' in str(dtype):
            print('datetime column:', df.columns[i])
            return df.columns[i]

find_dtype_cols = lambda df, dtypes: list(df.select_dtypes(include=dtypes).columns)

def find_TS_value_cols(df): # instead of df.columns[2:]
    TS_value_cols = find_dtype_cols(df, 'number')

    # hacks to avoid "time_key" column
    TS_value_cols = [col for col in TS_value_cols if not 'time' in col]
    assert len(TS_value_cols)==1 and TS_value_cols[0]=='series_value'
    
    print('TS_value_cols:', TS_value_cols)
    return TS_value_cols

# verified to work 2/8/24
# NOTE: we need this more complicated approach because data is messy so pd.infer_freq is inconsistent!
def find_freq(df, datetime_col, unique_id_cols):
    if type(unique_id_cols) is str: unique_id_cols=[unique_id_cols]
    all_freqs=df.sort_values(unique_id_cols+[datetime_col]) \
                .groupby(unique_id_cols).apply(lambda group: pd.infer_freq(group[datetime_col]))
    freq=all_freqs.mode().item()
    print('freq:', freq)
    proportion_consistent=(freq==all_freqs).mean()
    print('proportion freqs consistent:', proportion_consistent)
    #assert proportion_consistent>0.25, 'Frequencies are inconsistent!'
    assert freq is not None
    return freq

# Verified to work 2/9/24
# Master timeseries forecasting preprocessing function!
def get_preprocessed_forecasting_data(df_raw, unique_id_col:str, max_df_len:int=None, forecast_history:int=None,
                                      forecast_horizon:int=None, valid_size=0.1, test_size=0.2, scale=False, fill_value=None):
    """
    gets preprocessed forecasting data in a simple way comparable to tsai's higher level classification api
    :param df_raw: the raw df e.g. from monash data repository
    :param max_df_len: max df len to avoid OOM
    :param forecast_history: number of steps into the past for prediction (NOTE: auto selected if None)
    :param forecast_horizon: number of steps into the future for prediction (NOTE: auto selected if None)
    :param valid_size: proportion of validation data
    :param test_size: proportion of test data
    :param scale: whether to scale the data with the pipelines
    :param fill_value: for filling missing values, defaults to ffill.
    """
    
    datetime_col = find_datetime_col(df_raw)
    TS_value_cols = find_TS_value_cols(df_raw) # this also might be constant? But regardless it's helpful to be prepared to multivariate datasets
    freq = find_freq(df_raw, datetime_col, unique_id_col)

    print('Example `pd.date_range()` constructed from given timeseries frequency & start datetime:')
    print(pd.date_range(start=df_raw[datetime_col].iloc[0], freq=freq, periods=5))
    
    # NOTE: These preprocessing steps are apparently necessary; some of the Monash data is messy!
    method = 'ffill' if fill_value is None else None
    preproc_pipe = sklearn.pipeline.Pipeline([
        ('shrinker', TSShrinkDataFrame()), # shrink dataframe memory usage
        ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col, unique_id_cols=unique_id_col)), # drop duplicate rows (if any)
        ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, unique_id_cols=unique_id_col, freq=freq)), # ass missing timestamps (if any)
        ('fill_missing', TSFillMissing(columns=TS_value_cols, method=method, value=fill_value)), # fill missing data (1st ffill. 2nd value=0)
        ], verbose=True)
    df = preproc_pipe.fit_transform(df_raw)
    pipelines = [preproc_pipe]
    print('df.head():\n', df.head())
    print('\ndf.dtypes:')
    print(df.dtypes)
    
    # NOTE: reconstruct split distribution graph as seen for LTSF but for arbitrary dataset
    unique_subset_len = len(df)//len(df[unique_id_col].unique()) # need use len of unique subsets
    if forecast_history is None: forecast_history = unique_subset_len//8 # steps in the past (based on LTSF, possibly requires adjustment)
    if forecast_horizon is None: forecast_horizon = int(forecast_history*0.6) # steps in the future (based on LTSF, possibly requires adjustment)

    print('unique_subset_len:', unique_subset_len)
    print('steps in the past:', forecast_history)
    print('steps in the future:', forecast_horizon)

    # truncate dataset len sensibly using unique_subset_len
    df = df.sort_values([unique_id_col, datetime_col]).reset_index(drop=True)
    if max_df_len: df = df.iloc[:max_df_len-(max_df_len % unique_subset_len)]

    print('\nlen(df_raw):', len(df_raw))
    print('reduced df proportion:', len(df)/len(df_raw))
    
    splits = get_forecasting_splits(df, fcst_history=forecast_history, fcst_horizon=forecast_horizon, datetime_col=datetime_col,
                                    unique_id_cols=unique_id_col, valid_size=valid_size, test_size=test_size, show_plot=True)

    if scale:
        # GOTCHA: this scales the regular df *inplace* too even though it is supposed to be returning the scaled_df
        exp_pipe = sklearn.pipeline.Pipeline([
            ('scaler', TSStandardScaler(columns=TS_value_cols)), # standardize data using train_split
            ], verbose=True) # "experiment pipeline"
        train_split = splits[0]
        df = exp_pipe.fit_transform(df, scaler__idxs=train_split)
        pipelines.append(exp_pipe)
    
    # NOTE: they mention this does a sliding window & I also saw there is a TSSlidingWindow (dataset) class
    # This makes me think that maybe TSDatasets & similar classes are lower-level & implicitly invoked by these higher level functions!
    X, y = prepare_forecasting_data(df, fcst_history=forecast_history, fcst_horizon=forecast_horizon, 
                                    x_vars=TS_value_cols, y_vars=TS_value_cols)
    return X, y, splits, pipelines

def get_Monash_forecasting_data_cached(dsid):
    print('All Monash Forecasting Datasets:\n', Monash_forecasting_list)
    print(f'Chosen Monash Forecasting Dataset: {dsid}')
    if dsid in get_Monash_forecasting_data_cached.cache:
        print('Found the dataset in the cache!')
        df_raw = get_Monash_forecasting_data_cached.cache[dsid]
    else: 
        print('Dataset not found in the cache...')
        df_raw = get_Monash_forecasting_data(dsid)
        get_Monash_forecasting_data_cached.cache[dsid]=df_raw
    return df_raw
get_Monash_forecasting_data_cached.cache={}

def get_preprocessed_Monash_forecasting_data(dsid, *args, return_raw_df=False, **kwd_args):
    df_raw = get_Monash_forecasting_data_cached(dsid) # load the data (possibly from cache)
    
    # unique_id_cols seems to be constant for Monash
    results = list(get_preprocessed_forecasting_data(df_raw, *args, unique_id_col = 'series_name', **kwd_args))
    if return_raw_df: results.append(df_raw)
    return results