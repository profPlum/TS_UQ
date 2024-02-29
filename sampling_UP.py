import torch as th
from torch import nn
from torch.utils.data import Dataset, DataLoader

## verified to work: 1/29/24
## duplicates data while retaining order (for e.g. previous sorting)
#def make_ordered_copies(df,n_copies):
#    df['order']=range(len(df))
#    copied_df=pd.concat([df]*n_copies,axis=0)
#    del df['order']
#    return copied_df.sort_values(by='order').drop(columns='order')

# TSAI: you would probably still want to use this
class UQSyntheticMomentsDataset(Dataset):
    def __init__(self, X, y, sigma_max_coef=2.5, cum_sum=False):
        """
        Synthetic UQ Dataset which generates random variances, (but real datum means) to Produce UQ moments.
        Note: we pass the delegated params to TS_forecasting_preprocessing.get_preprocessed_Monash_forecasting_data().
        :param X: preprocessed timeseries inputs
        :param X: preprocessed timeseries outputs
        :param sigma_max_coef: Coefficient multiplied with full sample stds to get max valid sigma values, 10x sample variance means it's useless.
        :param cum_sum: whether to use monotonically increasing variance
        """
        self.sigma_max_coef = sigma_max_coef
        self.cum_sum = cum_sum
        
        # NOTE: none of mu_df, sigma_df, or outs_df are actually pandas df's they should all be tensors 
        (self.df_mu, self.df_sigma), self.outs_df = self.generate_moments_data(X, y)
    
    # NOTE: the interface includes outs_df to enable larger scale data-augmentation
    # (via multiple variance realizations) & enables original coursening idea too
    def generate_moments_data(self, inputs_df, outs_df):

        ## simple way to augment the dataset with even more synthetic variances.
        #inputs_df=make_ordered_copies(inputs_df, self.n_copies).values
        #outs_df=make_ordered_copies(outs_df, self.n_copies).values

        print('generating synthetic variance moments')

        variable_dim = 1 # should probably be true in general
        reduce_dims = list(range(len(inputs_df.shape)))
        del reduce_dims[variable_dim]

        # TODO: do cumsum of sigmas across time?
        # 10x sample variance means it's useless
        sigmas_max = self.sigma_max_coef*th.Tensor(inputs_df.std(axis=tuple(reduce_dims), keepdims=True)).detach()
        print('sigmas_max: ')
        print(sigmas_max)

        df_mu = th.Tensor(inputs_df).detach()
        df_sigma = sigmas_max*th.rand_like(df_mu).detach()

        if self.cum_sum:
            df_sigma = th.cumsum(df_sigma, dim=1)
        
        print('df_sigma.min(): ')
        print(df_sigma.min(axis=0))
        print('df_sigma.max(): ')
        print(df_sigma.max(axis=0))

        return (df_mu, df_sigma), th.Tensor(outs_df).detach()
    
    def to(self, device):
        self.df_mu=self.df_mu.to(device)
        self.df_sigma=self.df_sigma.to(device)
        self.outs_df=self.outs_df.to(device)
        return self

    def __len__(self):
        return self.df_mu.shape[0]

    def __getitem__(self, idx):
        # this indexing syntax should just get first (batch) dimension, 
        # which is what we want regardless of what comes after.
        return (self.df_mu[idx], self.df_sigma[idx]), self.outs_df[idx]

# TSAI: this should work as-is for TSAI it is very general
class UQSamplesDataset(Dataset):
    """ Wrapper for UQMomentsDataset which produces samples from the corresponding moments """
    def __init__(self, moments_dataset: UQSyntheticMomentsDataset, constant=False):
        self.moments_dataset = moments_dataset
        self.rand_coef = (int)(not constant)

    def sample(self, mu, sigma):
        return mu + th.randn(*mu.shape).to(mu.device)*sigma*self.rand_coef

    def __len__(self):
        return len(self.moments_dataset)

    def __getitem__(self, idx):
        (mu, sigma), outputs = self.moments_dataset[idx]
        inputs = self.sample(mu, sigma)
        return inputs, outputs

# TSAI: this should also mostly work for TSAI as-is!
# NOTE: this takes MULTIPLE samples per distribution then get the SE for the entire distribution & save that as training target!
# you can do this partially by using split-apply-combine with pandas
class UQErrorPredictionDataset(Dataset):
    def __init__(self, target_model: nn.Module, moments_dataset: UQSyntheticMomentsDataset,
                 samples_per_distribution=1000, use_MAD=False, cleanup_freq=15, chunk_size=None): 
                  #, scale_UQ_output=False):
        self.use_MAD = use_MAD
        self.target_model = target_model
        try: self.target_model.eval()
        except: pass # maybe this is a model wrapper function?
        self.moments_dataset = moments_dataset
        self.sampling_dataset = UQSamplesDataset(moments_dataset)

        if isinstance(self.target_model, nn.Module): # maybe this is a model wrapper function?
            try: self.to('cuda') # use GPU temporarily for faster sampling
            except Exception as e:
                print(e, file=sys.stderr)
                print('Warning: NOT using CUDA for input sampling, this will be slow...', file=sys.stderr)

        # NOTE: idea is for each UQ distribution we sample n=samples_per_distribution times
        # then we derive SE from accumulated SSE. This is better than MAE because we can assume
        # that errors are i.i.d which lets us compute total uncertainty during demo 
        # P.S. this slick addition method lets us avoid using groupby! groups are implicitly positions!

        with th.no_grad():
            self.mu_model = 0 # dummy value to be replaced by matrix
            self.SE_model = 0 # dummy value to be replaced by matrix
            #target_model = th.vmap(self.target_model, chunk_size=chunk_size) # apply chunksize to avoid OOM
            
            for i in range(samples_per_distribution):
                print('taking sample: ', i, flush=True)
                input_samples, outputs = self.sampling_dataset[:]
                preds = target_model(input_samples).detach() # use local version which is chunked
                self.mu_model = self.mu_model + preds
                self.SE_model = self.SE_model + ((preds-outputs)**2).detach()
                # accumulate SSE, NOTE: VAR(X+Y)=VAR(X)+VAR(Y) | X indep Y

                if i%cleanup_freq==0:
                    print('garbage collecting', flush=True)
                    import gc
                    while gc.collect(): pass 
                    th.cuda.empty_cache()

            self.mu_model /= samples_per_distribution
            self.SE_model /= samples_per_distribution-1 # derive MSE (bias corrected)
            self.SE_model = self.SE_model**(1/2) # MSE --> Standard Error
        self.to('cpu') # must be on CPU after the sampling to avoid errors
        #if scale_UQ_output:
        #    self.UQ_output_scaler = StandardScaler()
        #    self.SE_model[:] = th.from_numpy(self.UQ_output_scaler.fit_transform(self.SE_model))
    
    def to(self, device):
        try: self.target_model=self.target_model.to(device)
        except: pass # maybe this is a model wrapper function?
        self.moments_dataset=self.moments_dataset.to(device)
        if 'mu_model' in vars(self): self.mu_model=self.mu_model.to(device)
        if 'SE_model' in vars(self): self.SE_model=self.SE_model.to(device)

    def __len__(self):
        return len(self.moments_dataset)

    def __getitem__(self, index):
        sigma_coef = 0.7978201 if self.use_MAD else 1 # this coefficient converts std to mad for normal distributions
        
        # TODO: consider moving the sampling procedure here for lazy eval in DataLoader workers...
        (mu, sigma), outputs = self.moments_dataset[index]
        X, y = th.cat((mu, sigma_coef*sigma), axis=1), th.cat((self.mu_model[index], sigma_coef*self.SE_model[index]), axis=1)
        assert X.shape[:2]==y.shape[:2] # ^ TSAI, GOTCHA: verify that axis=1 is the right dimension!
        return X, y

# maybe better b/c it works even for dict config?
def consume_kwds_for(f, kwargs):
    import inspect
    if type(f) is type: # apply to classes' __init__
        return consume_kwds_for(f.__init__, kwargs)

    # else regular execution
    arg_names = list(inspect.signature(f).parameters)
    fitted_kwds = {}
    for k, v in kwargs.copy().items():
        if k in arg_names:
            fitted_kwds[k]=v
            del kwargs[k]
    return fitted_kwds

# simple interface we needed discards unnecessary structure
def get_synthetic_error_regression_dataset(X, y, target_model, **kwd_args):
    print('make sure that X & y was the same data used to train target_model!')
    moments_dataset = UQSyntheticMomentsDataset(X, y, **consume_kwds_for(UQSyntheticMomentsDataset, kwd_args))
    error_pred_dataset = UQErrorPredictionDataset(target_model, moments_dataset, **consume_kwds_for(UQErrorPredictionDataset, kwd_args))
    assert len(kwd_args)==0, f'Not all kwdargs consumed! These remain: {list(kwd_args.keys())}' # assert it has all been consumed!
    print('done sampling')
    return error_pred_dataset[:]
