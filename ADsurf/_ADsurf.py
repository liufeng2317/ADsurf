'''
* Author: LiuFeng(USTC) : liufeng2317@mail.ustc.edu.cn
* Date: 2022-11-14 20:18:06
* LastEditors: LiuFeng
* LastEditTime: 2024-03-12 14:27:05
* FilePath: /AD_github/ADsurf/_ADsurf.py
* Description: 
* Copyright (c) 2022 by liufeng2317 email: 2397664955@qq.com, All Rights Reserved.
'''
from ADsurf._plotting import *
from ADsurf._utils import *
from ADsurf._MonteCarlo import *
from ADsurf._model import *
import ADsurf._surf96 as surf
import numpy as np
import torch
import os 

#########################################################################
#                   Model parameter
#########################################################################
class Model_param():
    """
    Model parameter setting
    ---------------
    dc : the sampling interval of phase velocity
        => float
    vmin : the minimum phase velocity
        => float
    vmax : the maximum phase velocity
        => float
    Nt : the sampling number of period/frequency
        => int
    tmin : the minimum period
        => float
    tmax : the maximum period
        => float
    layering_method : the parameterize method
        => string: None : 
        => string: "LN" 
                the layering by number method (requires depth_factor and layer_number)
        => string: "LR" 
                the layering ratio method (requires depth_factor and layering ratio)
                ..
                    Brady R. Cox, David P. Teague, 
                    Layering ratios: a systematic approach to the inversion of surface wave data in the absence of a priori information, 
                    Geophysical Journal International, 
                    Volume 207, 
                    Issue 1, October 2016, 
                    Pages 422-438, 
                    https://doi.org/10.1093/gji/ggw282
                ..
    initialize_method :
        => string: "" 
                you can specify the initial model manually
        => string: "Brocher" 
                use the empirical equation from Brocher (2005) to get the initial vs/vp/rho/thickness
                ..
                    Brocher, T. M., 2005, 
                    Empirical relations between elastic wavespeeds and density in the earth’s crust: 
                    Bulletin of the Seismological Society of America, 
                    95, 2081-2092, doi: 10.1785/0120050077.
                ..
        => string: "Constant"
                the vp with a constant ratio with vs, and rho is a constant value

    layering_ratio : Layering ratio
        => float: e.g. LR = 1/2/3/5
    depth_factor : depth factor
        => float: e.g. DF = 2/3
    layer_number : the number of layers (for LN method)
        => int
    vp_vs_ratio : the ratio of vp/vs (for 'Constant' initialize method )
        => float
    rho : a constant value for density (for 'Constant' initialize method)
        => float 
    """
    def __init__(self,
                dc=None,
                vmin=None,
                vmax=None,
                Nt =None,
                tmin = None,
                tmax = None,
                layering_method = "",
                initialize_method = "",
                layering_ratio = None,
                depth_factor=None,
                layer_number = None,
                vp_vs_ratio = None,
                rho = None,
                fundamental_range=[],
                ):
        self.dc = dc 
        self.vmin = vmin
        self.vmax = vmax
        self.Nt = Nt,
        self.tmin = tmin
        self.tmax = tmax
        self.clist = np.arange(self.vmin,self.vmax,self.dc)
        # parameterization
        self.layering_method = layering_method
        self.layering_ratio = layering_ratio
        self.depth_factor = depth_factor
        self.layer_number = layer_number
        # initialize model
        self.initialize_method = initialize_method
        self.vp_vs_ratio = vp_vs_ratio
        self.rho = rho
        self.fundamental_range = fundamental_range
        
    def _checkInput(self):
        if self.vmin>self.vmax:
            raise ValueError("the minimum phase velocity must less than the maximum phase velocity")
        if self.tmin>self.tmax:
            raise ValueError("the minimum period must less than the maximum period")
        if self.layering_method not in ["LN","LR","",None]:
            raise ValueError("you can take one of layering method including[LR/LN]")
        if len(self.fundamental_range) != 0 and len(self.fundamental_range):
            raise ValueError("The fundamental period range may set 2 number")
    
#########################################################################
#                   inversion parameter
#########################################################################
class inv_param():
    """
    Inversion parameter setting
    ---------------------------
    inversion method: invert only vs or vs and depth simultaneous.
        => string: "Vs"
        => string: "VsAndThick"
    wave: the dtype of surface wave 
        => string: "Rayleigh" (default)
    algorithm : the dunkin matrix
        => string: "Dunkin"
            ..
                COMPUTER PROGRAMS IN SEISMOLOGY 
                VOLUME IV
                COPYRIGHT 1986, 1991
                D. R. Russell, R. B. Herrmann
                Department of Earth and Atmospheric Sciences Saint Louis University
                221 North Grand Boulevard St. Louis, Missouri 63103 U. S. A.
            ..
    mode : the mode of dispersion curve
        => int: 0
    compress : Whether to compress the value of the output result forward operator determinant (-1, 1)
        => Boolean: True
    compress_method :
        => string: "exp"
    normalized : Whether to use the same frequency of all positive determinant values for normalization 
        => Boolean: True
    lr : learning rate
        => float
    damp_vertical : Regularization factor of Tikhonov Regularization (vertical direction)
        => float
    damp_horizontal : Regularization factor of Tikhonov Regularization (horizontal direction)
        => float
    iteration : the number of iterations
        => int
    step_size : Using StepLR method to reduce the learning rate, step_size indicates the reduction period
        => float
    gamma : Using StepLR method to reduce the learning rate, gamma indicates the reducton ratio
        => float
    optimizer : the method to optimize the model
        => string: "Adam"
    """
    def __init__(self,
                inversion_method = "vs",
                wave = "rayleigh",
                algorithm = "dunkin",
                mode = 0,
                compress = True,
                compress_method="exp",
                normalized = True,
                lr = 0,
                damp_vertical = 0,
                damp_horizontal = 0,
                iteration = 200,
                step_size = 100,
                gamma = 0.85,
                optimizer="Adam",
                ):
        self.inversion_method = inversion_method
        self.wave = wave
        self.algorithm = algorithm
        self.mode = mode 
        self.compress = compress
        self.compress_method = compress_method
        self.normalized = normalized
        self.lr = lr 
        self.damp_vertical = damp_vertical
        self.damp_horizontal = damp_horizontal
        self.iteration = iteration
        self.step_size = step_size
        self.gamma = gamma
        self.optimizer = optimizer

###########################################################################
##                           Forward Simulation
###########################################################################
def cal_determinant(vel_model,model_param,sampling_method="log-wavelength",sampling_num=100):
    """
        calculate the determinant of dispersion function
    """
    a   = numpy2tensor(list2numpy(vel_model["vp"]))
    b   = numpy2tensor(list2numpy(vel_model["vs"]))
    rho = numpy2tensor(list2numpy(vel_model["rho"]))
    d   = numpy2tensor(list2numpy(vel_model["thick"]))
    tmin = model_param.tmin; tmax=model_param.tmax
    fmin = 1/tmax;           fmax=1/tmin
    Nt   = sampling_num
    # sampling method
    if sampling_method=="frequency":
        t = 1/np.linspace(fmin,fmax,Nt)[::-1]
    elif sampling_method=="period":
        t = np.linspace(tmin,tmax,Nt)
    elif sampling_method=="log-wavelength":
        t = np.exp(np.linspace(np.log(tmin),np.log(tmax),Nt))
    tlist = numpy2tensor(t)
    # velocity list
    vmin = model_param.vmin; vmax=model_param.vmax
    dc   = model_param.dc
    vlist = numpy2tensor(np.arange(vmin,vmax,dc))
    # the fundation setting
    wave      = "rayleigh"
    algorithm = "dunkin"
    ifunc     = ifunc_list[algorithm][wave]
    llw = 0 if b[0] <= 0.0 else -1
    F =  surf_matrix.dltar_matrix(vlist, tlist, d,a, b,rho, ifunc, llw)
    return tlist,vlist,F
    
def cal_dispersion_curve(vel_model,model_param,sampling_method="log-wavelength",sampling_num=100,dispOrder=[0]):
    """
        dispOrder: [0,1,2]
    """
    b = numpy2tensor(list2numpy(vel_model["vs"]))
    a = numpy2tensor(list2numpy(vel_model["vp"]))
    rho = numpy2tensor(list2numpy(vel_model["rho"]))
    d = numpy2tensor(list2numpy(vel_model["thick"]))
    # tlist
    tmin = model_param.tmin; tmax=model_param.tmax
    fmin = 1/tmax;           fmax=1/tmin
    Nt   = sampling_num
    if sampling_method=="frequency":
        t = 1/np.linspace(fmin,fmax,Nt)[::-1]
    elif sampling_method=="period":
        t = np.linspace(tmin,tmax,Nt)
    elif sampling_method=="log-wavelength":
        t = np.exp(np.linspace(np.log(tmin),np.log(tmax),Nt))
    # velocity list
    vmin = model_param.vmin; vmax=model_param.vmax
    dc   = model_param.dc
    # the matrix
    wave      = "rayleigh"
    algorithm = "dunkin"
    ifunc     = ifunc_list[algorithm][wave]
    # calculate the dispersion curve through search roots
    if 0 in dispOrder:
        mode  = 0
        itype = 0
        pvs_true_0 = surf.surf96(t,d,a,b,rho,mode,itype,ifunc,dc)
        pvs_true_0 = tensor2numpy(pvs_true_0)
        mask0 = (pvs_true_0>0)
        t0 = numpy2tensor(t)[mask0]
        pvs_true0 = pvs_true_0[mask0]
    # mask0 = (pvs_true_0>0)* ((t<0.05) + (t>0.1))
    if 1 in dispOrder:
        # first order
        mode = 1
        pvs_true_1 = surf.surf96(t,d,a,b,rho,mode,itype,ifunc,dc)
        pvs_true_1 = tensor2numpy(pvs_true_1)
        mask1 = (pvs_true_1>0)
        t1 = numpy2tensor(t)[mask1]
        pvs_true1 = pvs_true_1[mask1]
    if 2 in dispOrder:
        # second order
        mode = 2
        pvs_true_2 = surf.surf96(t,d,a,b,rho,mode,itype,ifunc,dc)
        pvs_true_2 = tensor2numpy(pvs_true_2)
        mask2 = (pvs_true_2>0)
        t2 = numpy2tensor(t)[mask2]
        pvs_true2 = pvs_true_2[mask2]
    
    if len(dispOrder)==3:
        if len(t0)>0 and len(t1)>0 and len(t2)>0:
            pvs_true_disp = np.hstack((pvs_true0,pvs_true1,pvs_true2))
            pvs_true_t = np.hstack((t0,t1,t2))
            pvs_order = np.zeros_like(pvs_true_t)
            pvs_order[len(t0):len(t0)+len(t1)] = 1
            pvs_order[len(t0)+len(t1):] = 2
            pvs_true = np.hstack((pvs_true_t.reshape(-1,1),pvs_true_disp.reshape(-1,1),pvs_order.reshape(-1,1)))
        elif len(t0)>0 and len(t1)>0:
            pvs_true_disp = np.hstack((pvs_true0,pvs_true1))
            pvs_true_t = np.hstack((t0,t1))
            pvs_order = np.zeros_like(pvs_true_t)
            pvs_order[len(t0):] = 1
            pvs_true = np.hstack((pvs_true_t.reshape(-1,1),pvs_true_disp.reshape(-1,1),pvs_order.reshape(-1,1)))
        else:
            pvs_order = np.zeros(len(t0))
            pvs_true = np.hstack((t0.reshape(-1,1),pvs_true_0.reshape(-1,1),pvs_order.reshape(-1,1)))
    elif 0 in dispOrder and 1 in dispOrder:
        pvs_true_disp = np.hstack((pvs_true0,pvs_true1))
        pvs_true_t = np.hstack((t0,t1))
        pvs_order = np.zeros_like(pvs_true_t)
        pvs_order[len(t0):] = 1
        pvs_true = np.hstack((pvs_true_t.reshape(-1,1),pvs_true_disp.reshape(-1,1),pvs_order.reshape(-1,1)))
    else:
        pvs_order = np.zeros(len(t0))
        pvs_true = np.hstack((t0.reshape(-1,1),pvs_true_0.reshape(-1,1),pvs_order.reshape(-1,1)))
    return pvs_true
        
#########################################################################
#             Ground Truth model for Synthetic Test
#########################################################################
class Model():
    """
        A class used for synthetic Test
    """
    def __init__(self):
        self.true_model = {}
        self.init_model = {}

class True_model():
    """
    True model
    ---------------------
    model_param : the class of model parameter
        => class
    pvs_obs : the observed dispersion data
        => 2-D list/array [period,phase velocity]
    thick : the specified thickness (km)
        => 1-D array/list
    vs : the specified shear-wave velocity (km/s)
        => 1-D array/list
    vp : the specified compression-wave velocity (km/s)
        => 1-D array/list
    rho : the specified density (kg/m3)
        => 1-D array/list
    """
    def __init__(self,
                model_param,
                thick=[],
                vs = [],
                vp = [],
                rho= [],
                ):
        # observed dispersion curve data
        self.thick = list2numpy(thick)
        self.vp = list2numpy(vp)
        self.vs = list2numpy(vs)
        self.rho = list2numpy(rho)
        self.model_param = model_param
        
        self.tlist = []
        self.vlist = []
        
        # model
        self.vel_model = {
            "vs":self.vs,
            "vp":self.vp,
            "rho":self.rho,
            "thick":self.thick,
        }
    
    def _cal_determinant(self,sampling_method="log-wavelength",sampling_num=100):
        tlist,vlist,F = cal_determinant(self.vel_model,self.model_param,sampling_method=sampling_method,sampling_num=sampling_num)
        self.tlist = tlist
        self.vlist = vlist
        return F
    
    def _cal_dispersion_curve(self,sampling_method="log-wavelength",sampling_num=100,dispOrder=[0]):
        """
            dispOrder: [0,1,2]
        """
        pvs = cal_dispersion_curve(self.vel_model,self.model_param,sampling_method=sampling_method,sampling_num=sampling_num,dispOrder=dispOrder)
        return pvs

#########################################################################
#             Parameterization and initialize model
#########################################################################
class Init_model():
    """
    initial model
    ---------------------
    model_param : the class of model parameter
        => class
    pvs_obs : the observed dispersion data
        => 2-D list/array [period,phase velocity]
    thick : the specified thickness (km)
        => 1-D array/list
    vs : the specified shear-wave velocity (km/s)
        => 1-D array/list
    vp : the specified compression-wave velocity (km/s)
        => 1-D array/list
    rho : the specified density (kg/m3)
        => 1-D array/list
    """
    def __init__(self,
                model_param,
                pvs_obs=[],
                thick=[],
                vs = [],
                vp = [],
                rho= []
                ):
        # observed dispersion curve data
        self.pvs_obs = pvs_obs
        self.thick = list2numpy(thick)
        self.vp = list2numpy(vp)
        self.vs = list2numpy(vs)
        self.rho = list2numpy(rho)
        self.model_param = model_param
        
        self.tlist = []
        self.vlist = []
        # model
        self.init_model = {
            "vs":[],
            "vp":[],
            "rho":[],
            "thick":[],
            "layer_mindepth":[],
            "layer_maxdepth":[]
        }
        
        # model parameterization
        self._parameterization()
        
        # check the observed data
        self._checkDispData()
        
        # initialized the initial model
        self._genInitModel()
        
        
    def _parameterization(self):
        """
            the thickness and range of thickness for each layer
        """
        # model parameter
        layering_method = self.model_param.layering_method
        layering_ratio = self.model_param.layering_ratio
        depth_factor = self.model_param.depth_factor
        layer_number = self.model_param.layer_number
        if self.pvs_obs.shape[1] == 3:
            mask = self.pvs_obs[:,2]==0
        else:
            mask = self.pvs_obs[:,0]>0
        
        if len(self.model_param.fundamental_range)>0:
                mask = mask*(self.pvs_obs[:,0]>self.model_param.fundamental_range[0])*(self.pvs_obs[:,0]<self.model_param.fundamental_range[1])
        
        pvs_obs = list2numpy(self.pvs_obs)
        
        if layering_method.lower() == "lr":
            # layering ratio
            wmin,wmax = np.min(pvs_obs[:,0][mask]*pvs_obs[:,1][mask]),np.max(pvs_obs[:,0][mask]*pvs_obs[:,1][mask])
            layer_mindepth,layer_maxdepth = depth_lr(wmin=wmin,wmax=wmax,lr=layering_ratio,depth_factor=depth_factor)
            # the init depth = the average depth of maximum and minimum depth
            layer_depth = (np.array(layer_mindepth)+np.array(layer_maxdepth))/2
            # depth ==> thickness
            layer_thick = depth_to_thick(layer_depth)
            # the thickness of last layer
            layer_thick[-1] = layer_thick[-2]
            self.init_model["thick"] = np.array(layer_thick)
            self.init_model["layer_mindepth"] = layer_mindepth
            self.init_model["layer_maxdepth"] = layer_maxdepth
    
        elif layering_method.lower() == "ln":
            # layeing by number
            wmin,wmax = np.min(pvs_obs[:,0][mask]*pvs_obs[:,1][mask]),np.max(pvs_obs[:,0][mask]*pvs_obs[:,1][mask])
            layer_mindepth,layer_maxdepth = depth_ln(wmin=wmin,wmax=wmax,nlayers=layer_number,depth_factor=depth_factor)
            # take the average thickness of maximum thickness as the initiali depth
            layer_thick = [layer_maxdepth[0]/layer_number]*layer_number
            self.init_model["thick"] = np.array(layer_thick)
            self.init_model["layer_mindepth"] = layer_mindepth
            self.init_model["layer_maxdepth"] = layer_maxdepth
        elif layering_method.lower() == "none":
            # using the input thickness as the initialized thickness
            if len(numpy2list(self.thick))==0:
                raise ValueError("No specify the layering method and not input the layer's parameter !!")
            self.init_model["thick"] = self.thick
            wmin,wmax = np.min(pvs_obs[:,0][mask]*pvs_obs[:,1][mask]),np.max(pvs_obs[:,0][mask]*pvs_obs[:,1][mask])
            layer_mindepth,layer_maxdepth = depth_ln(wmin=wmin,wmax=wmax,nlayers=len(self.thick),depth_factor=depth_factor)
            self.init_model["layer_mindepth"] = layer_mindepth
            self.init_model["layer_maxdepth"] = layer_maxdepth
    
    def _genInitModel(self):
        """
            generate the inital model
        """
        initialize_method = self.model_param.initialize_method
        thick   = list2numpy(self.init_model["thick"])
        pvs_obs = list2numpy(self.pvs_obs)
        # select the fundamental data to initialize the model
        if pvs_obs.shape[1] == 3:
            mask = pvs_obs[:,2]==0
        else:
            mask = pvs_obs[:,0]>0
        # initialize the velocity model
        if initialize_method == "Brocher":
            _,init_vp,init_vs,init_rho = gen_init_model(pvs_obs[:,0][mask],pvs_obs[:,1][mask],thick=thick,area=True)
        elif initialize_method == "Constant":
            vp_vs_ratio = self.model_param.vp_vs_ratio
            rho = self.model_param.rho
            _,init_vp,init_vs,init_rho = gen_init_model1(pvs_obs[:,0][mask],pvs_obs[:,1][mask],thick=thick,vp_vs_ratio=vp_vs_ratio,rho=rho,area=True)
        else:
            init_vp  = self.vp
            init_vs  = self.vs
            init_rho = self.rho
            if len(numpy2list(init_vp))==0:
                raise NameError("No specify the initialize method and input vs/vp/rho values!!")
        self.init_model["vp"] = init_vp
        self.init_model["vs"] = init_vs
        self.init_model["rho"] = init_rho
    
    def _checkDispData(self):
        """
            chekc the dispersion data
        """
        self.pvs_obs = list2numpy(self.pvs_obs)
        if not self.pvs_obs.ndim==2:
            raise ValueError("The observed dispersion curve need a 2-D array")

    def _cal_determinant(self,sampling_method="log-wavelength",sampling_num=100):
        tlist,vlist,F = cal_determinant(self.init_model,self.model_param,sampling_method=sampling_method,sampling_num=sampling_num)
        self.tlist = tlist
        self.vlist = vlist
        return F
    
    def _cal_dispersion_curve(self,sampling_method="log-wavelength",sampling_num=100,dispOrder=[0]):
        """
            dispOrder: [0,1,2]
        """
        pvs = cal_dispersion_curve(self.init_model,self.model_param,sampling_method=sampling_method,sampling_num=sampling_num,dispOrder=dispOrder)
        return pvs

#########################################################################
#                   Initialize model using MonteCarlo method
#########################################################################
class init_model_MonteCarlo():
    """
    Generate init model using MonteCarlo method
    ------------------------------------------
    model_param : the model parameters
        => class
    sampling_num : the sampling number for initial model
        => int
    sampling method : Random model with normal distribution according to the reference model
        => string
    vsrange_sign : Methods for constructing a range of normally distributed stochastic models
        => string: "mul"
        => string: "plus"
    vsrange : Range of random distribution of normal distribution
        => if "mul" : vs in [reference_model*(1-vsrange[0]), reference_model*(1+vsrange[1])]
        => if "plus" : vs in [reference_model-vsrange[0], reference_model+vsrange[1]]
    sigma_vs :  controls the degree of aggregation of the normal distribution, 
                the larger the value, the closer the stochastic model and the reference model are approximately
        => float
    sigma_vs : controls the degree of aggregation of the normal distribution, 
                the larger the value, the closer the stochastic model and the reference model are approximately
        => float
    AK135_data : Additional data for handling vs>4.6
        => 2-D matrix
            more infomation can be find in
                ..
                    Wu, G.‐x., Pan, L., Wang, J.‐n., & Chen,X. (2020). 
                    Shear velocity inversion usingmultimodal dispersion curves fromambient seismic noise data of USArray transportable array. 
                    Journal of Geophysical Research: Solid Earth, 
                    125,e2019JB018213. https://doi.org/10.1029/2019JB018213
                ..
    """
    def __init__(self,
                model_param,
                init_model,
                sampling_num = 200,
                sampling_method = "normal",
                vsrange_sign = "mul",
                vsrange = [-0.2,0.2],
                sigma_vs = 10,
                sigma_thick = 0,
                AK135_data=[],
                forward=True
                ):
        self.model_param = model_param
        self.init_model = init_model
        self.sampling_num = sampling_num
        self.sampling_method = sampling_method
        self.vsrange_sign = vsrange_sign
        self.vsrange = vsrange
        self.sigma_vs = sigma_vs
        self.sigma_thick = sigma_thick
        self.AK135_data = AK135_data
        self.MonteCarlo_model = {
            "vs":[],
            "vp":[],
            "rho":[],
            "thick":[],
            "loss":[],
        }
        self.forward=forward
        self._MonteCarlo()
    
    def _MonteCarlo(self):
        pvs_obs = self.init_model.pvs_obs
        clist   = self.model_param.clist
        d = list2numpy(self.init_model.init_model["thick"])
        a = list2numpy(self.init_model.init_model["vp"])
        b = list2numpy(self.init_model.init_model["vs"])
        rho = list2numpy(self.init_model.init_model["rho"])
        depth_up_boundary   = list2numpy(self.init_model.init_model["layer_maxdepth"])
        depth_down_boundary = list2numpy(self.init_model.init_model["layer_mindepth"])
        vsrange = self.vsrange
        initialize_method = self.model_param.initialize_method
        vp_vs_ratio = self.model_param.vp_vs_ratio
        sampling_num = self.sampling_num
        sampling_method = self.sampling_method
        layering_method = self.model_param.layering_method
        sigma_vs = self.sigma_vs
        sigma_thick = self.sigma_thick
        MonteCarlo_res = ADsurf_MonteCarlo(
                                    pvs_obs,clist,d,a,b,rho,
                                    layering_method = layering_method,
                                    depth_down_boundary=depth_down_boundary,
                                    depth_up_boundary=depth_up_boundary,
                                    vsrange_sign=self.vsrange_sign,
                                    vsrange=vsrange,
                                    sampling_num=sampling_num,
                                    sampling_method=sampling_method,
                                    gen_velMethod=initialize_method,
                                    vp_vs_ratio=vp_vs_ratio,
                                    sigma_vs=sigma_vs,
                                    sigma_thick=sigma_thick,
                                    AK135_data=self.AK135_data,
                                    forward=self.forward
                                    )
        MonteCarlo_vs = MonteCarlo_res["model"]["vs"]
        MonteCarlo_thick = MonteCarlo_res["model"]["thick"]
        MonteCarlo_vp = MonteCarlo_res["model"]["vp"]
        MonteCarlo_rho = MonteCarlo_res["model"]["rho"]
        MonteCarlo_loss = MonteCarlo_res["loss"]
        self.MonteCarlo_model["vs"] = MonteCarlo_vs
        self.MonteCarlo_model["vp"] = MonteCarlo_vp
        self.MonteCarlo_model["rho"] = MonteCarlo_rho
        self.MonteCarlo_model["thick"] = MonteCarlo_thick
        self.MonteCarlo_model["loss"] = MonteCarlo_loss
        
###########################################################################
##                           Inversion Setting
###########################################################################
def inversion(model_param,inv_param,init_model,pvs_obs,vsrange_sign="mul",vsrange=[0.1,2],AK135_data=[],device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')):
    """
    inversion process
    -----------------
    model_param : the model parameters
        => class
    inv_param : the inversion parameters
        => class
    init_model : the initial model
        => class
    pvs_obs : the observed dispersion curve [period,phase velocity]
        => 2-D list/array
    vsrange_sign : the boundary setting method for vs inversion
        => string: "mul"
        => string: "plus"
    vsrange : [down_boundary,upper_boundary]
        => list/array
    AK135_data : the additional data for vs>4.6km/s
        => array
    """
    init_vs = init_model.init_model["vs"]
    if list2numpy(init_vs).ndim==1:
        return iter_inversion(model_param,inv_param,init_model,pvs_obs,vsrange_sign,vsrange,AK135_data,device)
    elif list2numpy(init_vs).ndim==2:
        return multi_inversion(model_param,inv_param,init_model,pvs_obs,vsrange_sign,vsrange,AK135_data,device)
    