'''
Author: liufeng2317 2397664955@qq.com
Date: 2022-10-19 21:07:44
* LastEditors: LiuFeng
* LastEditTime: 2024-03-13 17:34:45
* FilePath: /AD_github/ADsurf/_MonteCarlo.py
Description: 
'''
import sys
sys.path.append("ADsurf")
sys.path.append("ADsurf/_cps")
import ADsurf._cps._surf96_vector as surf_vector
import ADsurf._cps._surf96_matrix as surf_matrix
from tqdm import tqdm
from ADsurf._utils import *

def ADsurf_MonteCarlo(
                    pvs_obs,clist,d,a,b,rho,
                    layering_method = "",           # the parameterization method ["","LN","LR"]
                    depth_up_boundary=[],depth_down_boundary=[], # the range of thickness/depth
                    vsrange_sign="mul",
                    vsrange=[-0.2,0.2],             # the range of vs 
                    gen_velMethod="Brocher",        # the initialized method ["Brocher","Constant"]
                    vp_vs_ratio = 2.45,             # use the uniform initialized method 
                    sampling_num = 200,             # the number of sampling points
                    sampling_method = "normal",     # the method to sampling (Brocher of Constant)
                    sigma_vs=10,                    # a ratio of sigma = (vs_up - vs_down)/ratio
                    sigma_thick=0,                  # a ratio of sigma = (vs_up - vs_down)/ratio
                    ifunc=2,
                    AK135_data=[],
                    forward=True,
                    ):
    """
    generete initial model from a given reference model
    ---------------------------------------------------
    pvs_obs : the observed dispersion data
        => 2-D list/array
    clist : the phase velocity range
        => list/array : e.g. [2,2.001,2.002,...,3.001,...,4]
    d: depth
        => 1-D list
    a: compression wave velocity
        => 1-D list
    b: shear-wave velocity
        => 1-D list
    rho: dencity
        => 1-D list
    layering_method: the method for parameterize
        => string: None
        => string: "LN"
        => string: "LR"
    depth_up_boundary:
        => list/array
    depth_down_boundary:
        => list/array
    vsrange_sign:
        => string: "mul"
        => string: "plus"
    vsrange:
        => list/array
    gen_velMethod: the method for generating the velocity model
        => string: "Brocher"
        => string: "Constant"
        => string: "USArray"
    vp_vs_ratio: the ratio of vp/vs (for "Constant")
        => float
    sampling_num: the number of sampling
        => int
    sampling_method: the method for sampling
        => string: "normal"
    sigma_vs: degree of aggregation of normal distribution  
        => int
    sigma_thick: degree of aggregation of normal distribution
        => int
    ifunc
    AK135_data: additional data for vs>4.6
    forward: forward model the model or not 
        => boolean : default True
    """
    # the boundary of model
    if vsrange_sign == "mul":
        vs_down_boundary = b*vsrange[0]
        vs_up_boundary   = b*vsrange[1]
    else:
        vs_down_boundary = b+vsrange[0]
        vs_up_boundary   = b+vsrange[1]
    
    # the generated model
    MonteCarlo_vs       = np.ones((sampling_num,b.shape[0]))*b
    MonteCarlo_thick    = np.ones_like(MonteCarlo_vs)*d
    MonteCarlo_vp       = np.ones_like(MonteCarlo_vs)*a
    MonteCarlo_rho      = np.ones_like(MonteCarlo_vs)*rho

    # the disturbation vs
    for i in range(b.shape[0]):
        if sampling_method=="normal":
            MonteCarlo_vs[1:,i] = np.random.normal(loc=b[i],scale=(vs_up_boundary[i]-vs_down_boundary[i])/sigma_vs,size=sampling_num-1)
            MonteCarlo_vs[1:,i] = np.clip(MonteCarlo_vs[1:,i],vs_down_boundary[i],vs_up_boundary[i])

    # the disturbation depth
    if len(depth_up_boundary)==d.shape[0] and len(depth_down_boundary)==d.shape[0] and sigma_thick>0:
        if depth_up_boundary[0] == 0:
            depth_up_boundary = depth_up_boundary[1:]
        if depth_down_boundary[0] == 0:
            depth_down_boundary = depth_down_boundary[1:]
        if layering_method == "LR":
            MonteCarlo_depth = MonteCarlo_thick
            depth = np.cumsum(d)
            for i in range(d.shape[0]):
                if sampling_method=="normal":
                    MonteCarlo_depth[:,i] = np.random.normal(loc=depth[i],scale=(depth_up_boundary[i] - depth_down_boundary[i])/sigma_thick,size=sampling_num)
                    MonteCarlo_depth[:,i] = np.clip(MonteCarlo_depth[:,i],a_min=depth_down_boundary[i],a_max=depth_up_boundary[i])
            MonteCarlo_depth = np.insert(MonteCarlo_depth,0,0,axis=1)
            MonteCarlo_thick = np.diff(MonteCarlo_depth,axis=1)
            # 保证厚度大于wmin/3
            MonteCarlo_thick = np.clip(MonteCarlo_thick,a_min=depth_down_boundary[1],a_max=None)
            # 第一个厚度是输入值
            MonteCarlo_thick[0] = d
            # 最后一层的厚度等于倒数第二层厚度
            MonteCarlo_thick[:,-1] = MonteCarlo_thick[:,-2]
        elif layering_method == "LN":
            MonteCarlo_depth = MonteCarlo_thick
            depth = np.cumsum(d)
            for i in range(d.shape[0]):
                if sampling_method=="normal":
                    # LN方法每一层最小深度 layer[i-1] + wmin/3，最大厚度wmax
                    MonteCarlo_depth[:,i] = np.random.normal(loc=depth[i],scale=(depth_up_boundary[i] - depth_down_boundary[i])/sigma_thick,size=sampling_num)
                    if i==0:
                        MonteCarlo_depth[:,i] = np.clip(MonteCarlo_thick[:,i],a_min=depth_down_boundary[i],a_max=depth_up_boundary[i])
                    else:
                        MonteCarlo_depth[:,i] = np.clip(MonteCarlo_thick[:,i],a_min=MonteCarlo_depth[:,i-1] + depth_down_boundary[i],a_max=depth_up_boundary[i])
            MonteCarlo_depth = np.insert(MonteCarlo_depth,0,0,axis=1)
            MonteCarlo_thick = np.diff(MonteCarlo_depth,axis=1)
            # 保证厚度大于wmin/3
            MonteCarlo_thick = np.clip(MonteCarlo_thick,a_min=depth_down_boundary[1],a_max=None)
            # 第一个厚度是输入值
            MonteCarlo_thick[0] = d
            # 最后一层的厚度等于倒数第二层厚度
            MonteCarlo_thick[:,-1] = MonteCarlo_thick[:,-2]
    if gen_velMethod=="Brocher":
        for i in range(1,sampling_num):
            thick,vp,vs,rho     = gen_model(MonteCarlo_thick[i],MonteCarlo_vs[i],area=True)
            MonteCarlo_thick[i] = thick 
            MonteCarlo_vp[i]    = vp 
            MonteCarlo_vs[i]    = vs 
            MonteCarlo_rho[i]   = rho
    elif gen_velMethod=="Constant":
        for i in range(1,sampling_num):
            thick,vp,vs,rho = gen_model1(MonteCarlo_thick[i],MonteCarlo_vs[i],vp_vs_ratio=vp_vs_ratio,rho=rho,area=True)
            MonteCarlo_thick[i] = thick 
            MonteCarlo_vp[i] = vp 
            MonteCarlo_vs[i] = vs 
            MonteCarlo_rho[i] = rho
    elif gen_velMethod == "USArray":
        # if velocity over 4.6 km/s, we take AK135 model replace
        for i in tqdm(range(1,sampling_num)):
            thick,vp,vs,rho = gen_model(MonteCarlo_thick[i],MonteCarlo_vs[i],area=True)
            mask = vs>4.6
            mask_vp = []
            mask_rho = []
            for j in range(len(vp[mask])):
                mask_vp.append(AK135_data[np.argmin(np.abs(AK135_data[:,2]-vp[mask][j]))][2])
                mask_rho.append(AK135_data[np.argmin(np.abs(AK135_data[:,1]-rho[mask][j]))][1])
            vp[mask] = mask_vp
            rho[mask] = mask_rho
            MonteCarlo_thick[i] = thick 
            MonteCarlo_vp[i] = vp 
            MonteCarlo_vs[i] = vs 
            MonteCarlo_rho[i] = rho
        
    loss_lists = np.zeros(sampling_num)
    tlist = numpy2tensor(pvs_obs[:,0].copy().reshape(-1))
    vlist = numpy2tensor(pvs_obs[:,1].copy().reshape(-1))
    
    if forward:
        # MonteCarlo search
        pbar = tqdm(range(sampling_num))
        for i in pbar:
            d  = numpy2tensor(MonteCarlo_thick[i])
            rho=numpy2tensor(MonteCarlo_rho[i])
            a = numpy2tensor(MonteCarlo_vp[i])
            b = numpy2tensor(MonteCarlo_vs[i])
            llw = 0 if b[0] <= 0.0 else -1
            loss = surf_vector.dltar_vector(vlist, tlist, d, a, b, rho, ifunc,llw)
            Olist = tlist.reshape(1,-1)
            Clist = numpy2tensor(clist.reshape(1,-1))
            det  = surf_matrix.dltar_matrix(Clist, Olist, d, a, b, rho, ifunc, llw)
            loss = loss/(torch.max(det,dim=0).values - torch.min(det,dim=0).values)
            loss_mean = torch.sum(torch.abs(loss))/len(loss)
            loss_lists[i] = loss_mean
            pbar.set_description("MonteCarlo: Iter:{},loss:{:.5}".format(i,loss_mean.detach().numpy()))
            
    MonteCarlo_search = {
        "model":{
            "thick":MonteCarlo_thick,
            "vs":MonteCarlo_vs,
            "vp":MonteCarlo_vp,
            "rho":MonteCarlo_rho
        },
        "loss":loss_lists
    }
    return MonteCarlo_search