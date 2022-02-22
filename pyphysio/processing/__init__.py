import xarray as _xr
import numpy as _np
# from ..signal import create_signal

#enable dask?
try:
    from dask import __name__ as _
    scheduler = 'threads'
    # available schedulers:
    # #distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads
    print('Using dask. Scheduler: threads')
except:
    scheduler = 'single-thread'

class Algorithm(object):
    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)  # already checked by __init__
        
        #if not specified algorithms operate on 1d signals
        #and return a signal with the same size
        self.dimensions = {}#'time': 0}
    
    @property
    def name(self):
        return(self.__class__.__name__)
        
    def __mapper_func__(self, signal_in):
        # print('-----> Algorithm.__mapper_func__')
        result_numpy = self.algorithm(signal_in)
        result_out = self.__finalize__(result_numpy, signal_in)
        # print(result_out.coords)
        # print('<----- Algorithm.__mapper_func__')
        return(result_out)

    def __call__(self, signal_in, add_signal=True, dimensions=None):
        # print('----->', self.name, '__call__')
        #This function iteratively calls the self.algorithm on each signal
        #(i.e. channel+component)
        # print('-----> Algorithm.__call__')
        
        #The user will mainly call Algorithms on a Dataset
        #but the __call__ rolling mechanism assumes to operate on DataArray
        if isinstance(signal_in, _xr.Dataset):
            signal = signal_in.p.main_signal.copy(deep=True)
        else:
            # print(type(signal_in))
            # print(signal_in.shape)
        
            signal = signal_in.copy(deep=True)
        
        signal_name = signal.name
        #from here, signal is a DataArray
        
        #rely on algorithm self.dimensions to know how to proceed
        if dimensions is None: 
            dimensions = self.dimensions

        if dimensions == 'none': 
            #process all information at once
            #used to avoid chunks in internal calls
            signal_out = self.__mapper_func__(signal)
            
        else:
            #use mapper
            if dimensions == 'special':
                #special algorithms that return DataArrays with non conventional
                #dimensions (e.g. frequencies)
                
                #get chunk_dict and template from the algorithm's class
                chunk_dict, template = self.__get_template__(signal)

                
            else:#typical usage
                #will include all dimensions except for those
                #along which the algorithm is applied
                
                # print(signal)
                # print(dimensions)
                # # 
                chunk_dict = {}
                template_shape = []
                template_coords = {}
                
                for dim in ('time', 'channel', 'component'):
                    size_in_dim = signal.sizes[dim]
                    
                    if dim not in dimensions.keys():
                        #the dimension is not used
                        chunk_dict[dim] = 1
                        size_out_dim = size_in_dim
                        coords = signal.coords[dim].values
                    else:
                        if dimensions[dim] == 0:
                            size_out_dim = size_in_dim
                            coords = signal.coords[dim].values
                        else:
                            size_out_dim = dimensions[dim]
                            if size_out_dim <= size_in_dim:
                                coords = signal.coords[dim].values[:size_out_dim]
                    template_coords[dim] = coords
                    template_shape.append(size_out_dim)
                
                #create template
                output = _np.zeros(template_shape)
                template = _xr.DataArray(output, dims = ('time', 'channel', 'component'),
                                         coords=template_coords,
                                         name=signal.name)
                
                template.name = signal_name

            template_dask = template.chunk(chunk_dict)
            signal_dask = signal.chunk(chunk_dict)

            mapper =  _xr.map_blocks(self.__mapper_func__, 
                                      signal_dask.copy(deep=True), 
                                      template = template_dask)
            #distributed, multiprocessing, processes, single-threaded, sync, synchronous, threading, threads
            signal_out = mapper.load(scheduler=scheduler) #distributed, single-threaded
        
        #The user will mainly call Algorithms on a Dataset
        #so it will expect a Dataset as result
        if isinstance(signal_in, _xr.Dataset):
            #add windowing info
            strange_result = False
            for dim in signal_out.dims:
                if dim in list(signal_in.coords):
                    if signal_out.sizes[dim] == 1 and signal_in.sizes[dim] != 1:
                        #there has been a windowing operation
                        coord_start = signal_in.coords[dim].values[0]
                        coord_stop = signal_in.coords[dim].values[-1]
                            
                        signal_out = signal_out.assign_coords({f'{dim}_start': (dim, [coord_start])})
                        signal_out = signal_out.assign_coords({f'{dim}_stop': (dim, [coord_stop])})
                    
                    elif signal_out.sizes[dim] != signal_in.sizes[dim]:
                        #there has been a different type of change in the shape
                        # we should just convert the result to a dataset and return it
                        # so we flag this as strange_result
                        strange_result = True
            
            #transform to Dataset
            #ISSUE: if "expanding" a coordinate (e.g. see pyphysio.FunctionalSeparationFilter)
            #these steps reset the original shape (e.g. from 4 to 2)
                
            signal_name = signal.name
            output_name = f'{signal_name}_{self.name}'
            
            if strange_result:
                signal_ds_out = signal_out.to_dataset(name = output_name)
                signal_ds_out.attrs['MAIN'] = output_name
                signal_ds_out.attrs['history'] = [output_name]
                signal_ds_out.p.main_signal.attrs = signal_in.p.main_signal.attrs
                return(signal_ds_out)
            
            signal_ds_out = signal_in.copy(deep=True)
            
            signal_ds_out = signal_ds_out.assign({output_name:signal_out})
            signal_ds_out.attrs['MAIN'] = output_name
            
            if add_signal:
                signal_ds_out.attrs['history'].append(self.name)
            else:
                signal_ds_out = signal_ds_out.drop(signal_name)
                signal_ds_out.attrs['history'] = [output_name]
            
            signal_ds_out.p.main_signal.attrs = signal_in.p.main_signal.attrs
            
            # print('<-----', self.name, '__call__ [Dataset]')
            return(signal_ds_out)
        
        signal_out.attrs = signal_in.attrs
        # print('<-----', self.name, '__call__')
        return(signal_out)        
        
    def __finalize__(self, result, signal_in, dimensions='none'):
        '''
        General function to obtain a coherent output 
        from the calls to self.algorithm.
        The output should be a dataaarry or dataset
        '''
                
        # print('-----> Algorithm.__finalize__')
        # print(result.shape)
        # print(type(signal_in))
        if result.ndim == 1:
            result = _np.expand_dims(result, [1,2])

        signal_out = _xr.DataArray(result, 
                                   dims=('time', 'channel', 'component'), 
                                   name=signal_in.name)
        
        for dim in ('time', 'channel', 'component'):
            
            if signal_in.sizes[dim] == signal_out.sizes[dim]:
                #same size --> same coords
                signal_out = signal_out.assign_coords({dim:signal_in.coords[dim].values})
                
            elif signal_out.sizes[dim] == 1: 
                #indicators or windowed algorithms
                coord_start = signal_in.coords[dim].values[0]
                signal_out = signal_out.assign_coords({dim:[coord_start]})
                
            else:
                #we do not know how to assign coordinates to this dimension
                pass
                    
        signal_out.attrs = signal_in.attrs.copy()
        # print('<----- Algorithm.__finalize__')       
        return(signal_out)
    
    def __repr__(self):
        return self.__class__.__name__ + str(self._params) if 'name' not in self._params else self._params['name']

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def set(self, **kwargs):
        kk = self.get()
        kk.update(kwargs)
        self.__init__(**kk)

    def get(self, param=None):
        """
        Placeholder for the subclasses
        @return
        """
        if param is None:
            return self._params
        else:
            return self._params[param]

    def algorithm(cls, signal):
        """
        Placeholder for the subclasses
        @raise NotImplementedError: Ever
        :param params:
        :param data:
        """
        pass

def algo(function, **kwargs):
    """
    Builds on the fly a new algorithm class using the passed function and params if passed.
    :param function: function(data, params) to be called
    :param kwargs: parameters to pass to the function.
    :return: An algorithm class if params is None else a parametrized algorithm instance.
    """

    class Custom(Algorithm):
        def __init__(self, **kwargs):
            Algorithm.__init__(self, **kwargs)

        def algorithm(self, signal):
            params = self._params
            return function(signal, params)

    if len(kwargs) == 0:
        return Custom
    else:
        return Custom(**kwargs)