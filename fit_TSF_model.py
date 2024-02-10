import random
import torch, sklearn, pandas, numpy as np
from tsai.basics import *
import TS_forecast_preprocessing
from jsonargparse import CLI # automatic CLI!!

def _filter_kwargs(f):
    import inspect
    arg_names = inspect.getargspec(f).args
    def _f(*args, **kwargs):
        return f(*args, **{k: v for k, v in kwargs.items() if k in arg_names})
    return _f

# TODO: use with json argparse to include model hyper-params
# This is cheatcodes to make jsonargparse work for everything!
# NOTE: for InceptionTimePlus, you need to instead call this function on InceptionModulePlus
def _fit_kwds_for(f, kwargs): # maybe better b/c it works even for dict config?
    try: arg_names = inspect.getargspec(f).args
    except TypeError: return fit_kwds_for(f.__init__, kwargs)
    return {k: v for k, v in kwargs.items() if k in arg_names}

def seed_everything(seed):
    # seed everything
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fit_TSForecaster(dataset: str=None, n_epochs: int = 1, lr_max: float = 0.0025,
        batch_size: int=16, max_df_len: int=100_000, scale=True, seed: int=None, fit_UQ=True, **kwd_args):
    """
    fits a TSForecasting model to a given dataset
    :param n_epochs: training epochs
    :param lr_max: initial lr before decay
    :param batch_size: train batch size
    :param max_df_len: some datasets are too big to fit into memory
    :param scale: whether to scale the dataset for easier training (it's invertible)
    :param seed: rng seed for EVERYTHING (like PL)
    :param fit_UQ: whether to also a fit a UQ model to the primary regressor
    """

    if seed is not None:
        seed_everything(seed)

    if dataset is None:
        # NOTE: Monash is the ticket for forecasting! It has 30 different forecasting datasets!
        # This is a static list of possible Monash forecasting datasets!
        print('All Monash Forecasting Datasets:\n', Monash_forecasting_list)
        dataset=random.choice(Monash_forecasting_list) # TODO: make CLI arg & replace with df file path
        print(f'...Randomly selecting Monash (dsid):', dataset)
        prep_function = TS_forecast_preprocessing.get_preprocessed_Monash_forecasting_data
    else:
        dataset = pd.read_csv(dataset)
        prep_function = TS_forecast_preprocessing.get_preprocessed_forecasting_data

    X, y, splits, pipelines = prep_function(dataset, max_df_len=max_df_len, scale=scale, **kwd_args)

    # NOTE: This is the full (default) arch_config directly from InceptionTimePlus.__init__() 
    # https://timeseriesai.github.io/tsai/models.inceptiontimeplus.html#inceptiontimeplus
    arch_config = dict(residual=True, depth=6, coord=False,
                       norm='Batch', zero_norm=False, act=torch.nn.modules.activation.ReLU, act_kwargs={},
                       sa=False, se=None, stoch_depth=1.0, ks=40,
                       bottleneck=True, padding='same', separable=False,
                       dilation=1, stride=1, conv_dropout=0.0, bn_1st=True)
    model_arch = "InceptionTimePlus" # default, make CLI arg?

    # GOTCHA: sometimes the code will die here, the reason is that it runs OOM with a very large dataset!
    # in this case trying running again with another smaller dataset.
    learn = TSForecaster(X, y, splits=splits, batch_size=batch_size, path="models", pipelines=pipelines,
                         arch=model_arch, arch_config=arch_config, metrics=[mse, mae])#, cbs=ShowGraph())
    # Why do they supply the pipelines again? are they reapplied? # it seems not based on lack of verbose output
    learn.dls.valid.drop_last = True
    print(learn.summary())

    print('fitting mean model:')
    learn.fit_one_cycle(n_epochs, lr_max=lr_max)
    learn.export(f'{model_arch}_mean.pt')
    if fit_UQ:
        preds, *_ = learn.get_X_preds(X)
        residuals = y-to_np(preds) # this seems to be standard definition of residuals
        residual_learn = TSForecaster(X, residuals, splits=splits, batch_size=batch_size, path="models", pipelines=pipelines,
                                      arch=model_arch, arch_config=arch_config, metrics=[mse, mae])
        residual_learn.dls.valid.drop_last = True
       
        print('fitting residual model:')
        residual_learn.fit_one_cycle(n_epochs, lr_max=lr_max)
        residual_learn.export(f'{model_arch}_residual.pt')
    
    return learn, residual_learn

if __name__ == '__main__':
    CLI(fit_TSForecaster) # auto CLI! Wahoo!
    print('done')
