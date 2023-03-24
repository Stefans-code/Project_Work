import numpy as _np
import xarray as _xr
_xr.set_options(keep_attrs=True)

from . import scheduler
    
class _Algorithm(object):
    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)  # already checked by __init__
        
        self.dimensions = {}#'time': 0}
    
    @property
    def name(self):
        return(self.__class__.__name__)
    
    def __get_template__(self, signal):
        '''
        Obtain the template of the output.
        Should be overwritten by algorithms that have a special output format
        (e.g. add a coordinate like frequency or change substantially the
        shape).
        
        Used by __call__ to know how to create chunks and compose the results
        on the different chunks

        Parameters
        ----------
        signal : TYPE
            DESCRIPTION.

        Returns
        -------
        chunk_dict : dictionary
            dictionary with information on how to perform the rolling
        template : xarray.DataArrat
            template of the output

        '''
        dimensions = self.dimensions
        
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
                    else:
                        coords = _np.arange(len(size_out_dim))
            template_coords[dim] = coords
            template_shape.append(size_out_dim)
        
        #create template
        output = _np.zeros(template_shape)
        template = _xr.DataArray(output, dims = ('time', 'channel', 'component'),
                                 coords=template_coords,
                                 name=signal.name)
        
        # template.name = signal
        return(chunk_dict, template)
    
    def __call__(self, signal_in, add_signal=True, dimensions=None, scheduler=scheduler, **kwargs):
        '''
        This function iteratively calls the self.algorithm on signal's chunks.
        If dask is installed and properly configured, this allows to parallelize
        the executon, for instance in cases of multi-channel/multi-components
        data.
        
        The workflow is the following:
        __call__() will apply the function __mapper_func__() to each signal chunk
        using the _xr.map_blocks function.
        
        __mapper_func__ will call the algorithm() function on each signal chunk
        algorithm() will return a numpy array, which is properly formatted into
        a DataArray by the subsequent call to __finalize__.
        
        the _xr.map_blocks function takes care of composing the results from 
        different chunks into a DataArray.
        
        This mechanism requires a dictionary to inform how to create the chunks
        and a template of the output of the parallelization (e.g. format of the 
        expected result). Both are obtained by the call to __get_template__(), 
        which uses information in self.dimensions.
        
        Special cases can be managed as follows:
        
        - self.dimensions = 'none'
          For algorithms on which the parallelization should not be applied.
          The user will properly implement the __mapper_func__ to return the
          desired result
         
        - By properly overwriting the __finalize__() and/or __get_template__() 
          functions, for algorithm with a special output shape.
        

        Parameters
        ----------
        signal_in : xarray.Dataset
            The input signal.
        add_signal : boolean, optional
            Whether to return a signal which also stores the input signal. 
            The default is True.
            
        #TODO: remove dimensions
        dimensions : 'none', None, dict
            This is to allow fancy uses... 
        
        
        scheduler : string, optional
            To allow changing the scheduler at runtime. Useful for debugging.
            The default is scheduler.

        Returns
        -------
        result : xarray.Dataset or xarray.DataArray
            The result of the algorithm applied on the input signal.

        '''
        
        #The user will mainly call Algorithms on a Dataset
        #but the __call__ "rolling" mechanism assumes to operate on a DataArray.
        #These lines convert the input Dataset to a DataArray, making a
        #COPY of the input Dataset.
        if isinstance(signal_in, _xr.Dataset):
            signal = signal_in.p.main_signal.copy(deep=True)
        else:
            signal = signal_in.copy(deep=True)
        
        signal_name = signal.name
        
        if dimensions is None: 
            dimensions = self.dimensions

        if dimensions == 'none': 
            #This is to allow special implementations, where the "rolling"
            #mechanism is avoided
            signal_out = self.__mapper_func__(signal, kwargs)
        
        #Typical behaviour
        #All dimensions except those specified in dimensions are rolled
        else:
            #get chunk_dict and template from the algorithm's class
            chunk_dict, template = self.__get_template__(signal)

            template_dask = template.chunk(chunk_dict)
            signal_dask = signal.chunk(chunk_dict)
 
            #create the rolling mechanism
            #which calls self.__mapper_func__ on all chunks
            mapper =  _xr.map_blocks(self.__mapper_func__, 
                                      signal_dask.copy(deep=True), 
                                      kwargs = kwargs,
                                      template = template_dask)
            #apply the rollink mechanism and compose the results
            signal_out = mapper.load(scheduler=scheduler) #distributed, single-threaded

        #The user will mainly call Algorithms on a Dataset
        #so it will expect a Dataset as result
        if isinstance(signal_in, _xr.Dataset):
            output_name = f'{signal_name}_{self.name}'

            #TODO: why are we repeating these? they are also in __mapper_func__?
            # we should decide what happens to a dataset when an algorithm is applied!!!
            # we could just transform to _xr.Dataset?
            
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
            
            
            #TODO: ISSUE: if "expanding" a coordinate (e.g. see pyphysio.FunctionalSeparationFilter)
            #these steps reset the original shape (e.g. from 4 to 2)
            if strange_result:
                signal_ds_out = signal_out.to_dataset(name = output_name)
                signal_ds_out.attrs['MAIN'] = output_name
                signal_ds_out.attrs['history'] = [output_name]
                signal_ds_out.p.main_signal.attrs = signal_in.p.main_signal.attrs
                return(signal_ds_out)


            #TODO: if we are changing the size 
            #(e.g. reducing components/channels, timepoints)            
            #the next steps will recover the original dataset "shape"
            #which might be an unwanted result
            #create the output Dataset
            signal_ds_out = signal_in.copy(deep=True)
            signal_ds_out = signal_ds_out.assign({output_name:signal_out})
            signal_ds_out.attrs['MAIN'] = output_name
            
            #whether to keep the previous versions of the signal
            if add_signal:
                signal_ds_out.attrs['history'].append(self.name)
            else:
                signal_ds_out = signal_ds_out.drop(signal_name)
                signal_ds_out.attrs['history'] = [output_name]
            
            signal_ds_out.p.main_signal.attrs = signal_in.p.main_signal.attrs
            return(signal_ds_out)
        
        signal_out.attrs = signal_in.attrs
        return(signal_out)        
        
    def __mapper_func__(self, signal_in, **kwargs):
        '''
        This function is called by __call__, which parallelizes the execution
        
        Its main role is to decouple the application of the algorithm
        from the composition of the output as a xarray.DataArray.
        In fact the output returned by the algorithm function is (typically)
        a numpy array.
        This is formatted as a xarray.DataArray by the call to 
        the __finalize__ function.
        
        Parameters
        ----------
        signal_in : xarray.DataArray
            Signal on which the algorithm is called. 
            Can be a partition of the input signal (the one on which the user)
            applies the algorithm.

        Returns
        -------
        result_out : xarray.DataArray
            Output of the algorithm applied on the signal partition,
            formatted as a xarray.DataArray, which is then composed by __call__ 
            to create the general outcome (returned to the user).
        '''
        
        result_numpy = self.algorithm(signal_in, **kwargs)
        result_out = self.__finalize__(result_numpy, signal_in)
        
        return(result_out)

    def __finalize__(self, result, signal_in, dimensions='none'):
        '''
        General function to obtain a coherent output 
        from the calls to self.algorithm.
        The output should be a dataaarry or dataset
        '''

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
                signal_out = signal_out.assign_coords({dim:_np.arange(signal_out.sizes[dim])})
                    
        signal_out.attrs = signal_in.attrs.copy()
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