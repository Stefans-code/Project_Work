import numpy as _np
from scipy.interpolate import interp1d
from ..._base_algorithm import _Algorithm
import xarray as _xr

# TODO: reset info
def create_SD(nirs):
    if isinstance(nirs, _xr.Dataset):
        attributes = nirs.p.main_signal.attrs
    else:
        attributes = nirs.attrs
    
    SD = {}
    SD['Lambda'] = attributes['Lambda']
    SD['nSrcs'] = attributes['nSrcs']
    SD['nDets'] = attributes['nDets']
    SD['SpatialUnit'] = attributes['SpatialUnit']
    
    
    SD['MeasList'] = attributes['MeasList'].reshape(attributes['MeasList_DIM'], -1)
    SD['SDmask'] = attributes['SDmask'].reshape(attributes['SDmask_DIM'], -1)
    SD['SrcPos'] = attributes['SrcPos'].reshape(attributes['SrcPos_DIM'], -1)
    SD['DetPos'] = attributes['DetPos'].reshape(attributes['DetPos_DIM'], -1)
    SD['ChPos'] = attributes['ChPos'].reshape(attributes['ChPos_DIM'], -1)
    
    return(SD)


#%%
def _get_dfp(wavelengths, AGE):
    # see: Scholkmann. F. et al: "General equation for the differential pathlength factor of the frontal human head depending on wavelength and age" (2013)
    alpha = 223.3
    beta = 0.05624
    gamma = 0.8493
    delta = -5.723*1e-7
    epsilon = 0.001245
    zeta = -0.9025
    
    DPF_LA = _np.array([alpha + beta*AGE**gamma + delta*(LAMBDA**3) + epsilon*(LAMBDA**2) + zeta*LAMBDA for LAMBDA in wavelengths])
    return(DPF_LA)

def _get_ppf(LAMBDA, AGE):
    #http://support.nirx.de/question/how-should-we-choose-the-value-for-partial-pathlength-factor/
    PVC = 1/50
    dpf = _get_dfp(LAMBDA, AGE)
    ppf = dpf*PVC
    return(ppf)
    
def _getExtinctions(L, spectrum=1):
    '''
    Returns the specific absorption coefficients for
    [HbO Hb H2O lipid aa3]
    for the specified wavelengths. Note that the specific
    absorption coefficient (defined base e) is equal to the 
    specific extinction coefficient (defined base 10) times 2.303.
    '''
    assert spectrum in [1,2,3]
   
    if spectrum == 1:
        #print('W. B. Gratzer, Med. Res. Council Labs, Holly Hill,London N. Kollias, Wellman Laboratories, Harvard Medical School, Boston')
        '''
        These values for the molar extinction coefficient e in [cm-1/(moles/liter)] were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
        
        W. B. Gratzer, Med. Res. Council Labs, Holly Hill, London
        N. Kollias, Wellman Laboratories, Harvard Medical School, Boston
        To convert this data to absorbance A, multiply by the molar concentration and the pathlength. For example, if x is the number of grams per liter and a 1 cm cuvette is being used, then the absorbance is given by
        
                (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
          A =  ---------------------------------------------------
                                  66,500 [g/mole]
        
        using 66,500 as the gram molecular weight of hemoglobin.
        To convert this data to absorption coefficient in (cm-1), multiply by the molar concentration and 2.303,
        
        a = (2.303) e (x g/liter)/(66,500 g Hb/mole)
        where x is the number of grams per liter. A typical value of x for whole blood is x=150 g Hb/liter.
        '''
        from ._data import vLambdaHbOHb_1 as vLambdaHbOHb
        
    elif spectrum ==2:
        #print('J.M. Schmitt, "Optical Measurement of Blood Oxygenation by Implantable Telemetry," Technical Report G558-15, Stanford. \nM.K. Moaveni, "A Multiple Scattering Field Theory Applied to Whole Blood," Ph.D. dissertation, Dept. of Electrical Engineering, University of Washington, 1970')
        '''
        These values for the molar extinction coefficient e in [cm-1/(moles/liter)] were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
        J.M. Schmitt, "Optical Measurement of Blood Oxygenation by Implantable Telemetry," Technical Report G558-15, Stanford."
        M.K. Moaveni, "A Multiple Scattering Field Theory Applied to Whole Blood," Ph.D. dissertation, Dept. of Electrical Engineering, University of Washington, 1970.
        To convert this data to absorbance A, multiply by the molar concentration and the pathlength. For example, if x is the number of grams per liter and a 1 cm cuvette is being used, then the absorbance is given by
                 (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
          A =  ---------------------------------------------------
                                  66,500 [g/mole]
        using 66,500 as the gram molecular weight of hemoglobin.
        To convert this data to absorption coefficient in (cm-1), multiply by the molar concentration and 2.303,
        a = (2.303) e (x g/liter)/(66,500 g Hb/mole)
        '''
        from ._data import vLambdaHbOHb_2 as vLambdaHbOHb
        
    elif spectrum ==3:
        #print('S. Takatani and M. D. Graham, "Theoretical analysis of diffuse reflectance from a two-layer tissue model," IEEE Trans. Biomed. Eng., BME-26, 656--664, (1987). ');
        '''
        These values for the molar extinction coefficient e in [cm-1/(moles/liter)] were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
        
        S. Takatani and M. D. Graham, "Theoretical analysis of diffuse reflectance from a two-layer tissue model," IEEE Trans. Biomed. Eng., BME-26, 656--664, (1987).
        To convert this data to absorbance A, multiply by the molar concentration and the pathlength. For example, if x is the number of grams per liter and a 1 cm cuvette is being used, then the absorbance is given by
        
                 (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
           A =  ---------------------------------------------------
                                   66,500 [g/mole]
        
        using 66,500 as the gram molecular weight of hemoglobin.
        To convert this data to absorption coefficient in (cm-1), multiply by the molar concentration and 2.303,
        
        a = (2.303) e (x g/liter)/(66,500 g Hb/mole)
        where x is the number of grams per liter. A typical value of x for whole blood is x=150 g Hb/liter.
        '''
        from ._data import vLambdaHbOHb_3 as vLambdaHbOHb
        
    vLambdaHbOHb = vLambdaHbOHb.copy()
    vLambdaHbOHb[:,1] = vLambdaHbOHb[:,1] * 2.303
    vLambdaHbOHb[:,2] = vLambdaHbOHb[:,2] * 2.303
    '''
    %
    % ABSORPTION SPECTRUMOF H20
    % FROM G. M. Hale and M. R. Querry, "Optical constants of water in the 200nm to
    % 200um wavelength region," Appl. Opt., 12, 555--563, (1973).
    %
    % ON THE WEB AT
    % http://omlc.ogi.edu/spectra/water/abs/index.html
    %
    '''
    from ._data import vLambdaH2O
    
    '''
    %
    % Extinction coefficient for lipid.
    % I got this from Brian Pogue who got this from Matcher and Cope (DAB)
    % In units of per mm and convert to per cm.
    %
    '''
    from ._data import vLambdaLipid
    from ._data import vLambdaAA3

    n_lambda = len(L)
    exs = _np.zeros((n_lambda, 5))

    L = _np.array(L)
    
    # HbO, Hb
    idx = _np.where((L>=250) & (L<=1000))[0]
    exs[idx,0] = interp1d(vLambdaHbOHb[:,0],vLambdaHbOHb[:,1], kind='cubic')(L[idx])
    exs[idx,1] = interp1d(vLambdaHbOHb[:,0],vLambdaHbOHb[:,2], kind='cubic')(L[idx])

    # H2O
    idx=_np.where((L>=200) & (L<=1000))[0]
    exs[idx,2] = interp1d(vLambdaH2O[:,0],vLambdaH2O[:,1], kind='cubic')(L[idx])
    
    # lipid
    idx=_np.where((L>=650) & (L<=1058))[0]
    exs[idx,3]= interp1d(vLambdaLipid[:,0],vLambdaLipid[:,1], kind='cubic')(L[idx])
    
    # AA3
    idx=_np.where((L>=650) & (L<=950))[0]
    exs[idx,4] = interp1d(vLambdaAA3[:,0],vLambdaAA3[:,1], kind='cubic')(L[idx])
    return(exs)
    
#%%
def _intensity2OD(x):
    dm = _np.mean(abs(x), axis=0)
    x_out = -_np.log(abs(x)/(_np.ones(shape = x.shape)*dm))
    return(x_out)

#%%
def _OD2Conc(nirs, SD, channel, ppf=[6,6], force_max_dist=True):
    '''
    dc = hmrOD2Conc( dod, SD, ppf )
   
    UI NAME:
    OD_to_Conc
   
    dc = hmrOD2Conc( dod, SD, ppf )
    Convert OD to concentrations
   
    INPUTS:
    dod: the change in OD (#time points x #channels)
    SD:  the SD structure. A spatial unit of mm is assumed, but if
         SD.SpatialUnit = 'cm' then cm will be used.
    ppf: partial pathlength factors for each wavelength. If there are 2
         wavelengths of data, then this is a vector ot 2 elements.
         Typical value is ~6 for each wavelength if the absorption change is 
         uniform over the volume of tissue measured. To approximate the
         partial volume effect of a small localized absorption change within
         an adult human head, this value could be as small as 0.1.
   
    OUTPUTS:
    dc: the concentration data (#time points x 3 x #SD pairs
        3 concentrations are returned (HbO, HbR, HbT)
    '''
    L = SD['Lambda']
    n_waves = len(L)
    assert len(ppf)==n_waves, 'The length of PPF must match the number of wavelengths in SD.Lambda'
    
    n_samples = len(nirs)
    
    e = _getExtinctions(L)
    if SD['SpatialUnit'] == 'mm':
        e = e[:,0:2] / 10  # convert from cm to mm
    elif SD['SpatialUnit'] == 'cm':
        e = e[:,0:2]
    
    einv = _np.dot(_np.linalg.inv(_np.dot(e.T,e)), e.T)
    concentration = _np.zeros((n_samples, 1, 2))
    srcPos = SD['SrcPos']
    detPos = SD['DetPos']
    
    if 'Channels' in SD:
        rho = SD['Channels'][channel, 3]
    else:
        SDkey = SD['SDkey']
        src_idx = SDkey[channel,0]
        det_idx = SDkey[channel,1]
        
        rho = _np.linalg.norm(srcPos[src_idx,:] - detPos[det_idx,:])

    if force_max_dist and rho>3:
        rho=3
    current_dod = _np.stack([nirs[:,0, 0], nirs[:,0,1]], axis=1)
    concentration[:,0,:] =  _np.dot(einv, ( current_dod / (rho*ppf)).T).T

    return(concentration)

class Raw2Oxy(_Algorithm):
    def __init__(self, age=None, **kwargs):
        _Algorithm.__init__(self, age=age, **kwargs)
        self.dimensions = {'time':0, 'component':0}
    
    # def __call__(self, signal_in, manage_original='add'):
    #     return _Algorithm.__call__(self, signal_in,
    #                                by='channel', 
    #                                manage_original=manage_original)
    
    def algorithm(self, signal):
        SD = signal.attrs
        
        Lambda = SD['Lambda']
        age = self._params['age']
        
        ppf = [6.,6.] if age is None else _get_ppf(Lambda, age)
        ppf = _np.array(ppf)
        
        signal_values = signal.values
        # print(ppf)
        OD = _intensity2OD(signal_values)
        
        channel = int(signal.coords['channel'])
        oxy = _OD2Conc(OD, SD, channel, ppf)
        
        return(oxy)
        
        
    