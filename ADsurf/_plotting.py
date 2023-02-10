import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import pandas as pd
from ADsurf._utils import *
from scipy.interpolate import interp1d

plot_param = {
    'label_font':{
                'fontsize':25,
                'color':'k',
                'family':'Times New Roman',
                'weight':'normal',
                # 'style':'italic',
                },
    'title_font':{
                'fontsize':25,
                'color':'k',
                'family':'Times New Roman',
                'weight':'normal',
                'style':'italic',
                },
    'legend_fontsize':20,
    'ticks_fontsize':20,
}
plot_param = dictToObj(plot_param)

##########################################################################
#                           plot velocity model
##########################################################################
def plot_velModel(thick=[],vp=[],vs=[],save_path="",show=True):
    """
    Description:
        plot the velocity model 
    Input:
        thick : the thick of layers
        vp : the velocity of pressure wave 
        vs : the velocity of shear wave
        save_path : the save figure path.
    Outpue:
        the figure of velocity model
    """
    # tensor to numpy
    thick = tensor2numpy(thick)
    vp = tensor2numpy(vp)
    vs = tensor2numpy(vs)
    # plot velocity model
    plt.figure(figsize=(12,8))
    vp = np.insert(vp,0,vp[0])
    vs = np.insert(vs,0,vs[0])
    thick = np.insert(thick,0,0)
    if len(vp)>0:
        plt.step(vp,np.cumsum(thick),where="post",color='r',label="Vp")
    if len(vs)>0:
        plt.step(vs,np.cumsum(thick),where="post",color="k",label="Vs")
    else:
        print("Warning: Please input the shear wave velocity")
    # title / label and legend
    plt.xlabel("Velocity",fontdict=plot_param.label_font)
    plt.ylabel("Depth",fontdict=plot_param.label_font)
    plt.xticks(fontsize=plot_param.ticks_fontsize)
    plt.yticks(fontsize=plot_param.ticks_fontsize)
    plt.title("Velocity Model",fontdict=plot_param.title_font)
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right',fontsize=plot_param.legend_fontsize,frameon=True)
    plt.grid()
    
    # save figure or not
    if not save_path=="":
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()

##########################################################################
#                     plot disperation energy map  
##########################################################################

def plot_dispEnergyMap(t,vc,mat_dispEnergy,pvs_true=[],pvs_init = [],pvs_inversion=[],ptype="period",cmap="seismic",save_path="",show=True):
    """
    Description
        plot the disperision energy map
    input:
        t : the period(s) used in calculate the dispersion function
        vc : the phase velocity used in calculate the dispersion function
        det_dispEnergy : the value of dispersion functoin in each (t_i,vc_i) points,it's a matrix
        pvs_true : the ground truth of dispersion curve (1D for line,2D for scatter)
        pvs_init : the initialized model's dispersion curve (1D for line,2D for scatter)
        pvs_inversion : the inversion model's dispersion curve (1D for line,2D for scatter)
        ptype : the plot type of x axis : period default while you can change to frequency
        save_path : save figure or not
    """
    # torch to numpy
    t = tensor2numpy(t)
    vc = tensor2numpy(vc)
    mat_dispEnergy = tensor2numpy(mat_dispEnergy)
    pvs_true = tensor2numpy(pvs_true)
    pvs_init = tensor2numpy(pvs_init)
    pvs_inversion = tensor2numpy(pvs_inversion)

    # plot dispersion energy map
    if ptype=="frequency":
        t = 1/t
        if len(pvs_init)>0:
            if pvs_init.ndim == 2:
                pvs_init[:,0] = 1/pvs_init[:,0]
        if len(pvs_true)>0:
            if pvs_true.ndim==2:
                pvs_true[:,0] = 1/pvs_true[:,0]
        if len(pvs_inversion)>0:
            if pvs_inversion.ndim == 2:
                pvs_inversion[:,0] = 1/pvs_inversion[:,0]
    
    X,Y = np.meshgrid(t,vc)
    plt.figure(figsize=(12,8))
    plt.contourf(X,Y,mat_dispEnergy,120,cmap=cmap)
    # plt.contourf(X,Y,mat_dispEnergy,120,cmap="coolwarm")
    plt.colorbar()

    # plot dispersion curve
    if len(pvs_true)>0:
        if pvs_true.ndim==2:
            plt.scatter(pvs_true[:,0],pvs_true[:,1],c="#00ff00",s=20,label="Observed")
        else:
            plt.scatter(t,pvs_true,c="#00ff00",s=20,label="Observed")
    if len(pvs_init)>0:
        if pvs_init.ndim == 2:
            plt.scatter(pvs_init[:,0],pvs_init[:,1],c="#7B7D7D",s=20,label = "Initial")
        else:
            plt.scatter(t,pvs_init,c="#7B7D7D",s=20,label = "Initial")
    if len(pvs_inversion)>0:
        if pvs_inversion.ndim == 2:
            plt.scatter(pvs_inversion[:,0],pvs_inversion[:,1],c="black",s=20,label = "inversion")
        else:
            plt.scatter(t,pvs_inversion,c="black",s=20,label = "Inversion")
            # plt.plot(t,pvs_inversion,color="black",linestyle="--",linewidth=5,label="inversion")

    # title、label and legend
    plt.title("Determinant Image",fontdict=plot_param.title_font)
    if ptype=="frequency":
        plt.xlabel("frequency",fontdict=plot_param.label_font)
    else:
        plt.xlabel("period",fontdict=plot_param.label_font)
    plt.xticks(fontsize=plot_param.ticks_fontsize)
    plt.yticks(fontsize=plot_param.ticks_fontsize)
    plt.ylabel("Phase Velocity",fontdict=plot_param.label_font)
    plt.legend(loc='upper right',fontsize=plot_param.legend_fontsize,frameon=True)
    if not save_path=="":
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()

def plot_allModel(vs_inversion=[],thick_inversion=[],
                vs_true=[],thick_true=[],
                vs_init=[],thick_init=[],
                vs_matlab=[],thick_matlab=[],
                vs_up_boundary = [],vs_down_boundary=[],boundary_thick=[],
                xlim=[],
                best_index=-1,largest_depth=0.1,
                plot_all=False,save_path="",show=True):
    """
    plot the inversion result in step stairs map
    --------------
    Input Parameter:
        vs_inversion : Array(MxN) 
            => the inversion shear wave velocity of Adisba(M is the iteration and N is the reuslts dimensional)
        thick_inversion : Array(MxN) 
            => the inversion depth of Adisba(M is the iteration and N is the reuslts dimensional)
        vs_true : Array(N) 
            => the ground truth velocity 
        thick_true : Array(N) 
            => the ground truth thick
        vs_init : Array(N) 
            => the initialize velocity
        thick_tinit : Array(N) 
            => the initialize thick
        vs_matlab : Array(N) 
            => the matlab inversion velocity
        thick_matlab : Array(N) 
            => the matlab inversion thick
        best_index : Int 
            => the best iteration during inversion
        plot_all : Bool 
            => plot the medium process or not(iteration process)
        save_path : string 
            => the figure's saving path
        show : Bool 
            =>figure the picture or not
    """
    if torch.is_tensor(vs_inversion):
        vs_inversion = tensor2numpy(vs_inversion)
    if torch.is_tensor(thick_inversion):
        thick_inversion = tensor2numpy(thick_inversion)
    plt.figure(figsize=(12,8))
    # plot the medium process
    if plot_all:
        for i in range(vs_inversion.shape[0]-1):
            vs_inversion_single = np.insert(vs_inversion[i,:],0,vs_inversion[i,0])
            if np.array(thick_inversion).ndim==2:
                thick_inversion_single = np.insert(thick_inversion[i,:],0,0)
            else:
                thick_inversion_single = np.insert(thick_inversion,0,0)
            thick_inversion_single[-1] = largest_depth - np.cumsum(thick_inversion_single)[-2]
            plt.step(vs_inversion_single,np.cumsum(thick_inversion_single),color='silver',where="post",linewidth=1)
    # plot the ground truth model
    if len(vs_true):
        thick_true = np.insert(thick_true,0,0)
        vs_true = np.insert(vs_true,0,vs_true[0])
        thick_true[-1] = largest_depth - np.cumsum(thick_true)[-2]
        plt.step(vs_true,np.cumsum(thick_true),color='r',label="Ground truth",linewidth=6,where="post")
    # plot the initial model
    if len(vs_init):
        thick_init = np.insert(thick_init,0,0)
        vs_init = np.insert(vs_init,0,vs_init[0])
        thick_init[-1] = largest_depth - np.cumsum(thick_init)[-2] 
        plt.step(vs_init,np.cumsum(thick_init),color='dimgray',linestyle="-",label="Initial",linewidth=4,where="post")  
    # plot the matlab inversion result 
    if len(vs_matlab)>0:
        thick_matlab = np.insert(thick_matlab,0,0)
        vs_matlab = np.insert(vs_matlab,0,vs_matlab[0])
        plt.step(vs_matlab,np.cumsum(thick_matlab),color='cyan',linestyle="-.",label="Matlab inversion",linewidth=6,where="post")  
        # plot the best inversion
    if np.array(thick_inversion).ndim==2:
        thick_inversion_best = np.insert(thick_inversion[best_index,:],0,0)
    else:
        thick_inversion_best = np.insert(thick_inversion,0,0)
    thick_inversion_best[-1] = largest_depth - np.cumsum(thick_inversion_best)[-2] 
    vs_inversion_best = np.insert(vs_inversion[best_index,:],0,vs_inversion[best_index,0])
    plt.step(vs_inversion_best,np.cumsum(thick_inversion_best),color="blue",linestyle="--",label="AD inversion best",linewidth=6,where="post")
    # boundary
    if len(boundary_thick)>0 and len(vs_up_boundary)>0 and len(vs_down_boundary)>0:
        boundary_thick = np.insert(boundary_thick,0,0)
        vs_up_boundary = np.insert(vs_up_boundary,0,vs_up_boundary[0])
        vs_down_boundary = np.insert(vs_down_boundary,0,vs_down_boundary[0])
        boundary_thick[-1] = largest_depth - np.cumsum(boundary_thick)[-2]
        plt.step(vs_down_boundary,np.cumsum(boundary_thick),color='#4596cf',linewidth=4,linestyle='--',where="post")
        plt.step(vs_up_boundary,np.cumsum(boundary_thick),color='#4596cf',linewidth=4,linestyle='--',where="post")
    if len(xlim)==2:
        plt.xlim(xlim[0],xlim[1])
    plt.xlabel("Velocity",fontdict=plot_param.label_font)
    plt.ylabel("Depth",fontdict=plot_param.label_font)
    plt.xticks(fontsize=plot_param.ticks_fontsize)
    plt.yticks(fontsize=plot_param.ticks_fontsize)
    plt.title("Velocity Model",fontdict=plot_param.label_font)
    plt.gca().invert_yaxis()
    plt.legend(loc='upper right',fontsize=10,frameon=True)
    if not save_path=="":
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()

def plot_velVariation(iter_vs=[],iter_thick=[],vs_true=[],thick_true=[],save_path="",show=True):
    """
    plot the inversion result's variation in step stairs map
    --------------
    Input Parameter:
        iter_vs : Array(MxN) 
            => the inversion shear wave velocity of Adisba(M is the iteration and N is the reuslts dimensional)
        iter_thick : Array(MxN) 
            => the inversion depth of Adisba(M is the iteration and N is the reuslts dimensional)
        vs_true : Array(N) 
            => the ground truth velocity 
        thick_true : Array(N) 
            => the ground truth thick
        save_path : string 
            => the figure's saving path
        show : Bool 
            =>figure the picture or not
    """
    import matplotlib.animation as animation
    if torch.is_tensor(iter_vs):
        iter_vs = tensor2numpy(iter_vs)
    if torch.is_tensor(iter_thick):
        iter_thick = tensor2numpy(iter_thick)
    
    fig = plt.figure(figsize=(12,8))
    plt.grid(ls="--")
    i = 0
    if np.array(iter_thick).ndim == 2:
        thick_single = np.insert(iter_thick[i,:],0,0)
        ymax = np.max(np.cumsum(iter_thick,axis=1)[:,-1])
    else:
        thick_single = np.insert(iter_thick,0,0)
        ymax = np.cumsum(iter_thick)[-1]
    vs_single = np.insert(iter_vs[i,:],0,iter_vs[i,0])
    if len(thick_true)>0:
        thick_true = np.insert(thick_true,0,0)
        vs_true = np.insert(vs_true,0,vs_true[0])
        plt.step(vs_true,np.cumsum(thick_true),color='red',label="Ground truth",linewidth=6,where="post")
    vs_ani = plt.step(vs_single,np.cumsum(thick_single),color='blue',where="post",linewidth=5,label="iteration {}".format(i))[0]
    plt.legend()
    if len(thick_true)>0:
        plt.xlim(np.array([np.min(iter_vs),np.min(vs_true)]).min()*0.9,
                np.array([np.max(iter_vs),np.max(vs_true)]).max()*1.1)
        plt.ylim(0,ymax)
    else:
        plt.xlim(np.min(iter_vs)*0.9,
                np.max(iter_vs)*1.1)
        plt.ylim(0,ymax)
    plt.xticks(fontsize=plot_param.ticks_fontsize)
    plt.yticks(fontsize=plot_param.ticks_fontsize)
    plt.title("iteration {} ".format(i))
    plt.gca().invert_yaxis()

    def update(i):
        if np.array(iter_thick).ndim == 2:
            thick_single = np.insert(iter_thick[i,:],0,0)
        else:
            thick_single = np.insert(iter_thick,0,0)
        vs_single = np.insert(iter_vs[i,:],0,iter_vs[i,0])
        vs_ani.set_data(vs_single,np.cumsum(thick_single))
        plt.title("iteration {} ".format(i))
        return vs_ani

    ani = animation.FuncAnimation(fig=fig,func=update,frames=np.arange(1,iter_vs.shape[0]),interval=200)
    if not save_path=="":
        ani.save(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()


##########################################################################
#      plot data distribution / train loss  and the misfit 
##########################################################################

def plot_dataDistribution(t,save_path="",show=True):
    """
    Description:
        plot the data's distribution(period or frequency)
    Input:
        t : period
        save_path : save the result or not 
    """
    t = tensor2numpy(t)
    data = np.hstack((t.reshape(-1,1),(1/t).reshape(-1,1)))
    data = pd.DataFrame(data,columns=["period","frequency"])
    h = sns.jointplot(x="period",y="frequency",data=data,kind='scatter',
                    cmap="Blues",marginal_kws=dict(bins=20, fill=True))
    h.fig.set_figheight(12)
    h.fig.set_figwidth(12)
    h.set_axis_labels("Period(s)","Frequency(Hz)",fontsize=20) #
    if not save_path=="":
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()


def plot_loss(loss,save_path="",show=True):
    """
    Description:
        plot the loss curve
    """
    plt.figure(figsize=(12,8))
    plt.plot(loss,color='k',label="loss curve")
    plt.xlabel("Iteration",fontdict=plot_param.label_font)
    plt.ylabel("Misfit",fontdict=plot_param.label_font)
    plt.xticks(fontsize=plot_param.ticks_fontsize)
    plt.yticks(fontsize=plot_param.ticks_fontsize)
    plt.title("Loss curve",fontdict=plot_param.label_font)
    plt.legend(loc='upper right',fontsize=plot_param.legend_fontsize,frameon=True)
    if not save_path=="":
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()

def plot_layer_misfit(thickness,vs_true,vs_inv=[],vs_matlab=[],thickness_true=[],save_path=""):
    """
    Description:
        plot the comparision between matlab and AD inversion result
    """
    plt.figure(figsize=(12,8))
    if len(thickness)==0:
        thickness_true = thickness
    thickness = np.insert(thickness,0,0)
    thickness_true = np.insert(thickness_true,0,0)
    vs_true_interp = np.ones_like(vs_inv)*vs_true[-1]
    cumthick_init = np.cumsum(thickness)
    cumthick_true = np.cumsum(thickness_true)
    for i in range(1,len(thickness)-1):
        for j in range(1,len(thickness_true)-1):
            if (cumthick_init[i]<cumthick_true[j] and cumthick_init[i]>cumthick_true[j-1]) or cumthick_init[i]==cumthick_true[j]:
                vs_true_interp[i-1] = vs_true[j-1]
    vs_true_interp[-1] = vs_true[-1]
    vs_true_interp = np.insert(vs_true_interp,0,vs_true_interp[0])

    if len(vs_inv)>0:
        vs_inv = np.insert(vs_inv,0,vs_inv[0])
        plt.plot(cumthick_init,vs_true_interp-vs_inv,'b*--',markersize=15,label="ADisba")

    if len(vs_matlab)>0:
        vs_matlab = np.insert(vs_matlab,0,vs_matlab[0])
        plt.plot(cumthick_init,vs_true_interp-vs_matlab,'*--',color="cyan",markersize=15,label="Matlab")
    
    plt.xlabel("Depth",fontdict=plot_param.label_font)
    plt.ylabel("Misfit",fontdict=plot_param.label_font)
    plt.xticks(fontsize=plot_param.ticks_fontsize)
    plt.yticks(fontsize=plot_param.ticks_fontsize)
    plt.title("The misfit between each layer",fontdict=plot_param.label_font)
    plt.axhline(y=0,linestyle="--",color="red")
    plt.legend(loc='upper right',fontsize=plot_param.legend_fontsize,frameon=True)
    if not save_path=="":
        plt.savefig(save_path)
    plt.show()

def plot_2DProfile(vs,thick,vmax,vmin,cmap="jet",interpolation_interval=0.1,plot_depest=3,save_path="",show=True):
    vs = tensor2numpy(vs)
    thick = tensor2numpy(thick)
    # plot the inversion model
    thick_inv_single = thick[0,:]
    # inpterpolate data for ploting
    thick_start = 0
    thick_end = np.sum(thick_inv_single)
    thick_interval = interpolation_interval
    depth_bc_interp = np.insert(np.cumsum(thick_inv_single.reshape(-1)),0,0)
    depth_af_interp = np.arange(thick_start,thick_end,step=thick_interval)
    vs_interp = np.zeros((vs.shape[0],depth_af_interp.shape[0]))
    for i in range(vs.shape[0]):
        vs_bc_interp = np.insert(vs[i,:],0,vs[i,0])
        f = interp1d(depth_bc_interp,vs_bc_interp)
        vs_af_interp = f(depth_af_interp)
        vs_interp[i,:] = vs_af_interp

    # figure to show
    largest_depth = plot_depest
    # ploting
    plt.figure(figsize=(12,8))
    plt.imshow(vs_interp[:,:int(largest_depth/thick_interval)].T,vmin=vmin, vmax=vmax,extent=[0,10,largest_depth,0],cmap=cmap)
    plt.colorbar(fraction=0.012)
    if not save_path=="":
        plt.savefig(save_path)
    if show:
        plt.show()
    else:
        pass
    plt.close()
