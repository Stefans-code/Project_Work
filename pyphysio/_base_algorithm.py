from abc import ABC, abstractmethod
import numpy as _np
import xarray as _xr
_xr.set_options(keep_attrs=True)

from . import scheduler
    
class _Algorithm(object):
    """
    Base class for all algorithms in pyphysio.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of parameters to be set for the algorithm.

    Attributes
    ----------
    _params : dict
        Dictionary of parameters set for the algorithm.
    required_dims : list
        List of named dimensions to be used by the algorithm.
    name : str
        Name of the algorithm.

    Methods
    -------
    __get_template__(self, signal)
        Obtain the template of the output.
    __call__(self, signal_in, add_signal=True, dimensions=None, scheduler=scheduler, **kwargs)
        Apply the algorithm on the input signal.
    __mapper_func__(self, signal_in, **kwargs)
        Function called by __call__ to parallelize the execution.
    __finalize__(self, result, signal_in, dimensions='none')
        Obtain a coherent output from the calls to self.algorithm.
    set_params(self, **kwargs)
        Set parameters for the algorithm.
    set(self, **kwargs)
        Set parameters for the algorithm and reinitialize the object.
    get(self, param=None)
        Get the parameters set for the algorithm.
    algorithm(cls, signal)
        Placeholder for the subclasses.
    """

    def __init__(self, **kwargs):
        self._params = {}
        self.set_params(**kwargs)  # already checked by __init__
        self.required_dims = []
        
    @property
    def name(self):
        return(self.__class__.__name__)
    
    @abstractmethod
    def __get_template__(self, signal):
        """
        Used by __call__ to know how to create chunks and compose the results
        on the different chunks.
        Should be implemented by each algorithm.
        In most cases, the __get_template__ function will just be a call to
        __compute_chunk_dict to compute the chunk_dict and to
        __compute_template__ with a specification of the output dimensions 
        (out_dims parameter).
        
        For more complex cases it can be adapted as needed.

        Parameters
        ----------
        signal : xarray.DataArray
            Input signal.

        Returns
        -------
        chunk_dict : dict
            Dictionary with information on how to perform the rolling.
        template : xarray.DataArray
            Template of the output.
        """
        pass
    

    def __get_template_timeonly__(self, signal):
        chunk_dict = self.__compute_chunk_dict__(signal)
        template = self.__compute_template__(signal)
        return(chunk_dict, template)
    def __compute_chunk_dict__(self, signal):
        """
        Helper function to obtain the chunk_dict to perform the rolling.
        Should be overwritten by algorithms that require a special behavior
        
        Used by __call__ to know how to create chunks

        Parameters
        ----------
        signal : xarray.DataArray
            Input signal.
        
        Returns
        -------
        chunk_dict: dict
            The chunk_dict.
        """
        
        signal_dims = signal.dims
        
        chunk_dict = {}
        for dim in signal_dims:
            if dim not in self.required_dims:
                chunk_dict[dim] = 1
        return(chunk_dict)        
        
    def __compute_template__(self, signal, out_dims = None):
        """
        Helper function to obtain the template of the output.
        Should be overwritten by algorithms that have a special output format
        
        Used by __call__ to know how to compose the results
        from the different chunks

        Parameters
        ----------
        signal : xarray.DataArray
            Input signal.
        out_dims: None or dict

        Returns
        -------
        template : xarray.DataArray
            Template of the output.
        """
        
        if out_dims is None: #no changes in dimensions or coordinates
            return(signal)
        
        shape_out = []
        coords_out = {}
        
        signal_dims = signal.dims
    
        #first process required dimensions
        for dim in signal_dims:
            
            if dim in out_dims.keys(): #if dim is changed
                out_coord = out_dims[dim]
                #if only integer, that is the new size of the dimension
                if isinstance(out_coord, int):
                    shape_out.append(out_coord)
                    coords_out[dim] = _np.arange(out_coord)
                #otherwise assume it is an iterable with the coords values
                #NOTE: for indicators this is the mechanism used. 
                # See the __Indicator class
                else:
                    shape_out.append(len(out_coord))
                    coords_out[dim] = out_coord
                
                #delete info from out_dims
                del out_dims[dim]
                
            else: #dim is not changed
                #keep the information from the input signal
                shape_out.append(signal.sizes[dim])
                coords_out[dim] = signal.coords[dim].values
            
        
        #add any other dimension 
        for dim in out_dims.keys():
            out_coord = out_dims[dim]
            if isinstance(out_coord, int):
                shape_out.append(out_coord)
                coords_out[dim] = _np.arange(out_coord)
            else:
                shape_out.append(len(out_coord))
                coords_out[dim] = out_coord
        
        out_data = _np.empty(shape_out)
        
        template = _xr.DataArray(out_data, 
                                 dims = coords_out.keys(),
                                 coords = coords_out,
                                 name=signal.name)
        
        return(template)

    
    def __call__(self, signal_in, scheduler=scheduler, **kwargs):
        '''
        This function iteratively calls the self.algorithm on signal's chunks.
        If dask is installed and properly configured, this allows to parallelize
        the executon, for instance in cases of multi-channel/multi-components
        data.
        This mechanism requires a dictionary to inform how to create the chunks
        and a template of the output of the parallelization (e.g. format of the 
        expected result). Both are obtained by the call to __get_template__(), 
        which uses information in self.chunk_dict.
        
        The typical workflow is the following:
        1. __get_template__ is called to get information on how to split the signal
          (chink_dict) and how the resulting xarray should look like (template).
          Then chunks of the input signal and of the template are created
          
        2. _xr.map_blocks will call __mapper_func__ to process each chunk.
        
        3. __mapper_func__ will call the algorithm() function on each signal chunk.
        algorithm() will return a numpy array, which is then properly formatted
        into a DataArray based on the template.
        
        4. the _xr.map_blocks function receives the results from all the chunks and
        takes care of composing them into a unique DataArray.
        
        5. attributes of the original signal are copied to the resulting xarray
        
        
        Parameters
        ----------
        signal_in : xarray.Dataset
            The input signal.
        
        scheduler : string, optional
            To allow changing the scheduler at runtime. Useful for debugging.
            The default is 'threads'.

        Returns
        -------
        result : xarray.Dataset or xarray.DataArray
            The result of the algorithm applied on the input signal.

        '''
        
        #The user will mainly call Algorithms on a Dataset
        #but the __call__ "rolling" mechanism assumes to operate on a DataArray.
        #These lines convert the input Dataset to a DataArray, making a
        #COPY of the input Dataset.
        signal = signal_in.copy(deep=True)
        
        signal_name = signal.name

        if len(self.required_dims) == 0:
            
            #This is to allow special implementations, where the "rolling"
            #mechanism is avoided
            result_numpy = self.algorithm(signal, **kwargs)
            
            try:
                #use __finalize__ if implemented
                result_out = self.__finalize__(result_numpy, signal)
                return(result_out)
            except:
                #just return whatever the algorithm method returns
                return(result_numpy)
            
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
            
            
        output_name = f'{signal_name}_{self.__repr__()}'
        signal_out.name = output_name
        
        #copy attributes of input dataset
        for k,v in signal_in.attrs.items():
            try:
                signal_out.attrs[k] = v.copy()
            except:
                signal_out.attrs[k] = v
        
        return(signal_out)        
        
    def __mapper_func__(self, signal_in, **kwargs):
        '''
        This function is called by __call__, which parallelizes the execution
        
        Its main role is to decouple the application of the algorithm
        from the composition of the output as a xarray.DataArray.
        In fact the output returned by the algorithm function is (typically)
        a numpy array.
        This is then formatted as a xarray.DataArray.
        
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
        
        chunk_dict, template_out = self.__get_template__(signal_in)
        for k,v in chunk_dict.items():
            assert signal_in.sizes[k] == v
            
        result_numpy = self.algorithm(signal_in, **kwargs)
        
        result_ndims = result_numpy.ndim
        template_ndims = template_out.ndim
        assert result_ndims <= template_ndims
        
        if result_ndims > len(self.required_dims): #adding a dimension
            add_axis = -2            
        else:
            add_axis = -1
        
        n_dims_to_add = template_ndims - result_ndims        
        
        for i in _np.arange(n_dims_to_add):
            result_numpy = _np.expand_dims(result_numpy, add_axis)
        
        
        coords_out = {}
        for dim in template_out.dims:
            coords_out[dim] = []
            
        #coordinates of dimensions that are in chunk dict
        #are taken from signal_in    
        for dim in chunk_dict.keys():
            coords_out[dim] = signal_in[dim].values

        #coordinates of dimensions that are not in chunk dict
        #are taken from template_out
        for dim in template_out.dims:
            if dim not in chunk_dict.keys():
                coords_out[dim] = template_out[dim].values
            
        signal_out = _xr.DataArray(result_numpy,
                                   coords = coords_out, 
                                   name=signal_in.name)

        signal_out.attrs = signal_in.attrs.copy()
        
        return(signal_out)

    def __repr__(self):
        return self.__class__.__name__ if 'name' not in self._params else self._params['name']

    def set_params(self, **kwargs):
        self._params.update(kwargs)

    def set(self, **kwargs):
        kk = self.get()
        kk.update(kwargs)
        self.__init__(**kk)

    def get(self, param=None):
        """
        Placeholder for the subclasses
        """
        if param is None:
            return self._params
        else:
            return self._params[param]

    @abstractmethod
    def algorithm(cls, signal):
        """
        This method is the algorithm that is applied to the signal.
        It should return a numpy array with a number of dimension that 
        is not lower than the number of required dimensions in self.required_dims
        """
        pass
