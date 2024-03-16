'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2022-11-14 20:18:06
* LastEditors: LiuFeng
* LastEditTime: 2024-03-12 14:45:08
* FilePath: /ADsurf/ADsurf/_utils.py
* Description: 
* Copyright (c) 2022 by liufeng2317 email: 2397664955@qq.com, All Rights Reserved.
'''
import numpy as np
import torch
from scipy.io import savemat,loadmat
from tqdm import tqdm
import warnings
from collections import namedtuple

##########################################################################
#                            cps setting  
##########################################################################
ifunc_list = {
    "dunkin": {"love": 1, "rayleigh": 2},
    "fast-delta": {"love": 1, "rayleigh": 3},
}

DispersionCurve = namedtuple(
    "DispersionCurve", ("period", "velocity", "mode", "wave", "type")
)

# transform dictionary to objective
class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dictToObj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dictToObj(v)
    return d

##########################################################################
#                            generate model   
##########################################################################
def gen_model(thick,vs,area=False):
    """
    generate the initial model based on empirical formula 
    developed by Thomas M.Brocher (2005).
    ---------------------
    Input Parameters:
        thick : Array(1D) 
            => the thickness of layer 
        vs    : Array(1D)
            => the shear wave velocity
        area  : boolen 
            => the output format
    --------------------
    Output parameters:
        model:Dict 
            the generated model
    """
    thick   = np.array(thick)
    vs      = np.array(vs)
    vp      = 0.9409 + 2.0947*vs - 0.8206*vs**2+ 0.2683*vs**3 - 0.0251*vs**4
    rho     = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return thick,vp,vs,rho
    else:
        return model

def gen_model1(thick,vs,vp_vs_ratio=2.45,rho=2,area=False):
    """
    generate the initial model based on constant poisson ratio and density
    ---------------------
    Input Parameters:
        thick : Array(1D) 
            => the thickness of layer 
        vs    :    Array(1D)
            => the shear wave velocity
        vp_vs_ratio: float
            => the poisson's ratio 
        rho   : float
            => the density (kg/m3)
        area  : boolen 
            => the output format
    --------------------
    Output parameters:
        model:Dict 
            the generated model
    """
    thick = np.array(thick)
    vs = np.array(vs)
    vp = vs*vp_vs_ratio
    rho = np.ones_like(vs)*rho
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return thick,vp,vs,rho
    else:
        return model

def gen_init_model(t,cg_obs,thick,area=False):
    """
    generate the initial model based on empirical formula 
    developed by Thomas M.Brocher (2005).
    ---------------------
    Input Parameters:
        t : 1D numpy array 
            => period of observaton dispersion points
        cg_obs: 1D numpy array 
            => phase velocity of observation dispersion points
        thick : 1D numpy array 
            => thickness of each layer
    Output: the initialize model
        thick : 1D numpy array 
            => thickness
        vs : 1D numpy array 
            => the shear wave velocity
        vp : 1D numpy array 
            => the compress wave velocity
        rho: 1D numpy array 
            => the density
    --------------------
    Output parameters:
        model:Dict 
            => the generated model
    """
    t = tensor2numpy(t).reshape(-1)
    cg_obs = tensor2numpy(cg_obs).reshape(-1)
    thick = tensor2numpy(thick).reshape(-1)
    t = t[np.argsort(t)]
    cg_obs = cg_obs[np.argsort(cg_obs)]
    wavelength = t*cg_obs
    nlayer = len(thick)
    lambda2L = 0.65 # the depth faction 0.63L
    beta = 0.92     # the poisson's ratio
    eqv_lambda = lambda2L*wavelength
    lay_model = np.zeros((nlayer,2))
    lay_model[:,0] = thick
    for i in range(nlayer-1):
        if i == 0:
            up_bound = 0
        else:
            up_bound = up_bound + lay_model[i-1,0] # the top-layer's depth
        low_bound = up_bound + lay_model[i,0] # the botton-layer's depth
        # vs for every layer
        lambda_idx = np.argwhere((eqv_lambda>up_bound) & (eqv_lambda<low_bound))
        if len(lambda_idx)>0:
            lay_model[i,1] = np.max(cg_obs[lambda_idx])/beta # phase velocity -> vs
        else:
            lambda_idx = np.argmin(np.abs(eqv_lambda - low_bound))
            lay_model[i,1] = cg_obs[lambda_idx]/beta
    # set the last layer
    lay_model[nlayer-1,0] = 0
    lay_model[nlayer-1,1] = np.max(cg_obs)*1.1
    thick = lay_model[:,0]
    vs = lay_model[:,1]
    vp = 0.9409 + 2.0947*vs - 0.8206*vs**2+ 0.2683*vs**3 - 0.0251*vs**4
    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return thick,vp,vs,rho 
    else:
        return model

def gen_init_model1(t,cg_obs,thick,vp_vs_ratio=2.45,rho=2,area=False):
    """
    generate the initial model based on constant poisson ratio and density
    ---------------------
    Input Parameters:
        t : 1D numpy array 
            => period of observaton dispersion points
        cg_obs: 1D numpy array 
            => phase velocity of observation dispersion points
        thick : 1D numpy array 
            => thickness of each layer
    Output: the initialize model
        thick : 1D numpy array 
            => thickness
        vs : 1D numpy array 
            => the shear wave velocity
        vp : 1D numpy array 
            => the compress wave velocity
        rho: 1D numpy array 
            => the density
    --------------------
    Output parameters:
        model:Dict 
            => the generated model
    """
    t = tensor2numpy(t).reshape(-1)
    cg_obs = tensor2numpy(cg_obs).reshape(-1)
    thick = tensor2numpy(thick).reshape(-1)
    t = t[np.argsort(t)]
    cg_obs = cg_obs[np.argsort(cg_obs)]
    wavelength = t*cg_obs
    nlayer = len(thick)
    lambda2L = 0.65 # the depth faction 0.63L
    beta = 0.92 # the poisson's ratio
    eqv_lambda = lambda2L*wavelength
    lay_model = np.zeros((nlayer,2))
    lay_model[:,0] = thick
    for i in range(nlayer-1):
        if i == 0:
            up_bound = 0
        else:
            up_bound = up_bound + lay_model[i-1,0] # the top-layer's depth
        low_bound = up_bound + lay_model[i,0] # the botton-layer's depth
        # vs for every layer
        lambda_idx = np.argwhere((eqv_lambda>up_bound) & (eqv_lambda<low_bound))
        if len(lambda_idx)>0:
            lay_model[i,1] = np.max(cg_obs[lambda_idx])/beta # phase velocity -> vs
        else:
            lambda_idx = np.argmin(np.abs(eqv_lambda - low_bound))
            lay_model[i,1] = cg_obs[lambda_idx]/beta
    # set the last layer
    lay_model[nlayer-1,0] = 0
    lay_model[nlayer-1,1] = np.max(cg_obs)*1.1
    thick = lay_model[:,0]
    vs = lay_model[:,1]
    vp = vs*vp_vs_ratio
    rho = np.ones_like(vs)*rho
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    if area:
        return thick,vp,vs,rho 
    else:
        return model

##########################################################################
#                              load model      
##########################################################################

def model_rms_misfit(vs_true,vs_compare):
    """
    Description:
        calculate the misfit between true model and the reverse model
    """
    if torch.is_tensor(vs_true):
        vs_true = vs_true.detach().numpy()
    if torch.is_tensor(vs_compare):
        vs_compare = vs_compare.detach().numpy()
    rms = np.sum(vs_true-vs_compare)/len(vs_true)
    return rms

def load_model(path):
    """
        load model from python
    """
    model = np.loadtxt(path)
    thick = model[:,0]
    vp = model[:,1]
    vs = model[:,2]
    rho = model[:,3]
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    return model

def load_initModel(path):
    """
        load initial model from matlab
    """
    init_data = loadmat(path)
    thick = init_data["initmod"][0][0][0].reshape(-1)
    vs = init_data["initmod"][0][0][1].reshape(-1)
    vp = init_data["initmod"][0][0][2].reshape(-1)
    rho = init_data["initmod"][0][0][3].reshape(-1)
    model = {
        "thick":thick,
        "vp":vp,
        "vs":vs,
        "rho":rho
    }
    return model

##########################################################################
#                              Parameterization      
##########################################################################

def check_wavelengths(wmin, wmax):
    """Check wavelength input.

    Specifically:
    1. Cast Wavelength to `float`.
    2. Wavelengths are > 0.
    3. Minimum wavelength is less than maximum wavelength.
    """
    wmin = float(wmin)
    wmax = float(wmax)

    # Check type and wavelength > 0.
    for val in [wmin, wmax]:
        if val <= 0:
            raise ValueError("Wavelength must be > 0.")

    # Compare wavelengths
    if wmin > wmax:
        msg = "Minimum wavelength must be less than maximum wavelength. Swapping!"
        warnings.warn(msg)
        wmin, wmax = wmax, wmin

    return (wmin, wmax)

def check_depth_factor(depth_factor):
    """Check input value for factor."""
    if not isinstance(depth_factor, (int, float)):
        msg = f"`factor` must be `int` or `float`. Not {type(depth_factor)}."
        raise TypeError(msg)
    if depth_factor < 2:
        depth_factor = 2
        msg = "`factor` must be >=2. Setting `factor` equal to 2."
        warnings.warn(msg)
    return depth_factor

def check_layering_ratio(lr):
    """Check input value for factor."""
    if not isinstance(lr, (int, float)):
        msg = f"`lr` must be `int` or `float`, not {type(lr)}."
        raise TypeError(msg)
    if lr <= 1:
        raise ValueError("`lr` must be greater than 1.")
    return lr

def depth_to_thick(depths):
    """Convert depths (top of each layer) to thicknesses

    Parameters
    ----------
    depth : list
        List of consecutive depths.

    Returns
    -------
    list
        Thickness for each layer. Half-space is defined with zero
        thickness.

    """
    depths = np.array(depths)
    if depths[0] != 0:
        depths = np.insert(depths,0,0)
    depths = np.array(depths)
    return np.diff(depths)

def thick_to_depth(thicknesses):
    """Convert thickness to depth (at top of each layer).

    Parameters
    ----------
    thickness : list
        List of thicknesses defining a ground model.

    Returns
    -------
    list
        List of depths at the top of each layer.

    """
    depths = [0]
    for clayer in range(1, len(thicknesses)):
        depths.append(sum(thicknesses[:clayer]))
    return depths

def depth_lr(wmin,wmax,lr,depth_factor):
    """Return minimum and maximum depth for each layer using the
    Layering Ratio approach developed by Cox and Teague (2016).
    rewrite by Joseph P. Vantassel and Brady R.Cox(2021)

    Note that the Layering Ratio approach implemented here has been
    modified slightly to ensure the maximum depth of the last layer
    does not exceed dmax. Suggestions for solving this issue are
    hinted at in Cox and Teague (2016), but not provided explicitly.

    Parameters
    ----------
    wmin, wmax : float
        Minimum and maximum measured wavelength from the
        fundamental mode Rayleigh wave dispersion.
    lr : float
        Layering Ratio, this controls the number of layers and
        their potential thicknesses, refer to Cox and Teague
        2016 for details.
    depth_factor : [float, int], optional
        Factor by which the maximum wavelength is
        divided to estimate the maximum depth of profiling,
        default is 2.

    Returns
    -------
        list of random depth of layer
    """
    wmin, wmax = check_wavelengths(wmin, wmax)
    depth_factor = check_depth_factor(depth_factor)
    lr = check_layering_ratio(lr)

    layer_mindepth = [wmin/3]
    layer_maxdepth = [wmin/2]
    dmax = wmax/depth_factor
    laynum = 1
    while layer_maxdepth[-1] < dmax:
        layer_mindepth.append(layer_maxdepth[-1])
        if laynum == 1:
            layer_maxdepth.append(
                layer_maxdepth[-1]*lr + layer_maxdepth[-1])
        else:
            layer_maxdepth.append(
                (layer_maxdepth[-1]-layer_maxdepth[-2])*lr + layer_maxdepth[-1])
        laynum += 1
    # If the distance between the deepest potential depth of the
    # bottom-most layer and dmax is greater than the potential
    # thickness of the bottom-most layer:
    # ---> Add a new layer
    if (dmax - layer_maxdepth[-2]) > (layer_maxdepth[-2] - layer_maxdepth[-3]):
        # Set the current max depth (which is > dmax) equal to dmax
        layer_maxdepth[-1] = dmax
        # Add half-space starting at dmax
        layer_mindepth.append(dmax)
        layer_maxdepth.append(dmax+1)  # Half-space
    # ---> Otherwise, extend the current last layer
    else:
        # Extend the deepest potential depth of the bottom-most layer
        # to dmax.
        layer_maxdepth[-2] = dmax
        # Set the old last layer to the half-space
        layer_mindepth[-1] = dmax
        layer_maxdepth[-1] = dmax+1  # Half-space
    
    return layer_mindepth,layer_maxdepth

def depth_ln(wmin, wmax, nlayers, depth_factor=2):
    """Calculate min and max depth for each layer using LN.
    first writed by Joseph P. Vantassel and Brady R.Cox(2021)
    
    Parameters
    ----------
    wmin, wmax : float
        Minimum and maximum measured wavelength from the
        fundamental mode Rayleigh wave disperison.
    nlayers : int
        Desired number of layers.
    depth_factor : [float, int], optional
        Factor by which the maximum wavelength is
        divided to estimate the maximum depth of profiling,
        default is 2.

    Returns
    -------
    Tuple
        Tindicating thickness of the form
        (thickness,min thickness, max thickness).

    """
    wmin, wmax = check_wavelengths(wmin, wmax)

    if not isinstance(nlayers, int):
        msg = f"`nlayers` must be `int`. Not {type(nlayers)}."
        raise TypeError(msg)
    if nlayers < 1:
        raise ValueError("Number of layers for must be >= 1.")

    depth_factor = check_depth_factor(depth_factor)
    layer_mindepth = np.array([wmin/3]*nlayers)
    layer_maxdepth = np.array([wmax/depth_factor]*nlayers)
    return layer_mindepth,layer_maxdepth

##########################################################################
#                          numpy <=====> tensor     
##########################################################################

def numpy2tensor(a):
    """
        transform numpy data into tensor
    """
    if not torch.is_tensor(a):
        return torch.tensor(a).to(torch.float32)
    else:
        return a.to(torch.float32)

def tensor2numpy(a):
    """
        transform tensor data into numpy
    """
    if not torch.is_tensor(a):
        return a 
    else:
        return a.detach().numpy()
    
##########################################################################
#                          list <=====> numpy     
##########################################################################

def list2numpy(a):
    """
        transform numpy data into tensor
    """
    if isinstance(a,list):
        return np.array(a)
    else:
        return a

def numpy2list(a):
    """
        transform numpy data into tensor
    """
    if not isinstance(a,list):
        return a.tolist()
    else:
        return a

##########################################################################
#                             calculate rms misfit      
##########################################################################

def model_rms_misfit(vs_true,vs_compare):
    """
    Description:
        calculate the misfit between true model and the reverse model
    """
    if torch.is_tensor(vs_true):
        vs_true = vs_true.detach().numpy()
    if torch.is_tensor(vs_compare):
        vs_compare = vs_compare.detach().numpy()
    rms = np.sum(vs_true-vs_compare)/len(vs_true)
    return rms