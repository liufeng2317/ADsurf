import ADsurf._cps._surf96_vector as surf_vector_iter_cpu
import ADsurf._cps._surf96_vector_gpu as surf_vector_iter_gpu
import ADsurf._cps._surf96_vectorAll as surf_vector_all_cpu
import ADsurf._cps._surf96_vectorAll_gpu as surf_vector_all_gpu
import ADsurf._cps._surf96_matrix as surf_matrix_iter_cpu
import ADsurf._cps._surf96_matrix_gpu as surf_matrix_iter_gpu
import ADsurf._cps._surf96_matrixAll as surf_matrix_all_cpu
import ADsurf._cps._surf96_matrixAll_gpu as surf_matrix_all_gpu
from ADsurf._utils import *
import torch


#########################################################################
#                           iterative inversion
#########################################################################
class iter_inversion():
    """
        inversion process
    """
    def __init__(self,
                model_param,
                inv_param,
                init_model,
                pvs_obs,
                vsrange_sign = 'mul',
                vsrange = [0.1,2],
                AK135_data = [],
                device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
                ):
        self.init_model = init_model
        self.model_param = model_param
        self.inv_param = inv_param
        self.pvs_obs = pvs_obs
        self.vsrange_sign = vsrange_sign
        self.vsrange = vsrange
        self.AK135_data = AK135_data
        self.device = device
        # inversion result definition
        self.inv_model = {
            "vs":[],
            "vp":[],
            "rho":[],
            "thick":[]
        }
        self.inv_process ={
            "iter_vs":[],
            "iter_thick":[],
            "loss":[]
        }
        self._run()
        
        
    def _run(self):
        # initial model
        init_vs = self.init_model.init_model["vs"]
        init_vp = self.init_model.init_model["vp"]
        init_thick = self.init_model.init_model["thick"]
        init_rho = self.init_model.init_model["rho"]
        layer_mindepth = self.init_model.init_model["layer_mindepth"]
        layer_maxdepth = self.init_model.init_model["layer_maxdepth"]
        b = numpy2tensor(list2numpy(init_vs)).to(self.device)
        a = numpy2tensor(list2numpy(init_vp)).to(self.device)
        rho = numpy2tensor(list2numpy(init_rho)).to(self.device)
        d = numpy2tensor(list2numpy(init_thick)).to(self.device)
        layer_mindepth = numpy2tensor(list2numpy(layer_mindepth)).to(self.device)
        layer_maxdepth = numpy2tensor(list2numpy(layer_maxdepth)).to(self.device)
        
        # observed dispersion data
        tlist = numpy2tensor(list2numpy(self.pvs_obs)[:,0].reshape(-1)).to(self.device)
        vlist = numpy2tensor(list2numpy(self.pvs_obs)[:,1].reshape(-1)).to(self.device)

        # initialized the calculator
        damp_vertical   = self.inv_param.damp_vertical
        damp            = damp_vertical
        Clist = numpy2tensor(list2numpy(self.model_param.clist).reshape(1,-1)).to(self.device)
        compress = self.inv_param.compress
        normalized = self.inv_param.normalized
        compress_method = self.inv_param.compress_method
        inversion_method = self.inv_param.inversion_method
        layering_method = self.model_param.layering_method
        initialize_method = self.model_param.initialize_method
        vp_vs_ratio = self.model_param.vp_vs_ratio
        
        # the dispersion funtion forward simulation
        calculator = iter_cal_grad(d,a,b,rho,Clist,damp,compress,normalized,compress_method,inversion_method,initialize_method,vp_vs_ratio,self.AK135_data,device=self.device)
        calculator = calculator.to(self.device)

        # optimizer
        lr = self.inv_param.lr
        iteration = self.inv_param.iteration
        step_size = self.inv_param.step_size
        gamma = self.inv_param.gamma
        if self.inv_param.optimizer == "Adam":
            optimizer = torch.optim.Adam(calculator.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
        else:
            raise NameError("The input optimizer {} can not find in Pytorch".format(self.inv_param.optimizer))
        
        # intermediate process data
        pbar = tqdm(range(iteration))
        iter_vs = torch.zeros((iteration,b.shape[0])).to(self.device)
        iter_thick = torch.zeros((iteration,d.shape[0])).to(self.device)
        loss_list = (torch.ones(iteration)*10).to(self.device)
        
        # inversion boundary
        if self.vsrange_sign =="mul":
            lower_b = self.vsrange[0]*b
            upper_b = self.vsrange[1]*b
        else:
            lower_b = b + self.vsrange[0]
            upper_b = b + self.vsrange[1]
            # lower_b[-1] = 4.5
        
        # early stoopint param
        patience = 50
        eps = 1e-4
        trigger_times = 0
        
        ###############################
        # single station inversion
        ###############################
        for j in pbar:
            loss = calculator(vlist,tlist)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            if inversion_method.lower()=="vs-and-thick":
                for k,para in enumerate(calculator.parameters()):
                    # constrain the velocity
                    if k == 0:
                        for ind in range(len(para.data)):
                            para.data[ind].clip_(min=lower_b[ind],max=upper_b[ind])
                        # last layer > second from the bottom
                        para.data[-1].clip_(min=(para.data[-2]).detach())
                        iter_vs[j,:len(para.data.detach())] = para.data.detach()
                    # constrain the depth
                    elif k ==1:
                        depth_temp = torch.cumsum(para.data,dim=0)
                        for ind in range(len(depth_temp)):
                            if layering_method == "LR":
                                depth_temp[ind] = torch.clip(depth_temp[ind],layer_mindepth[ind],a_max=layer_maxdepth[ind])
                            elif layering_method == "LN":
                                if ind == 0:
                                    depth_temp[ind] = torch.clip(depth_temp[ind],min=layer_mindepth[ind],max=layer_maxdepth[ind])
                                else:
                                    depth_temp[ind] = torch.clip(depth_temp[ind],min=(depth_temp[ind-1]+layer_mindepth[ind]),max=layer_maxdepth[ind])
                        depth_temp = torch.cat((torch.zeros(1).to(self.device),depth_temp),dim=0)
                        thick_temp = torch.diff(depth_temp)
                        # constrain the depth at least wmin/3
                        thick_temp = torch.clip(thick_temp,min=layer_mindepth[0],max=None)
                        for ind in range(len(para.data)):
                            para.data[ind] = thick_temp[ind]
                        iter_thick[j,:len(para.data.detach())] = para.data.detach()
            else:
                for para in calculator.parameters():
                    for ind in range(len(para.data)):
                        para.data[ind].clip_(min=lower_b[ind],max=upper_b[ind])
                    # last layer > second from the bottom
                    # para.data[0].clip_(min=lower_b)
                    # para.data[-1].clip_(min=(para.data[-2]).detach())
                    if self.device=="cpu":
                        iter_vs[j,:len(para.data.detach())] = para.data.detach()
                    else:
                        iter_vs[j,:len(para.data.detach())] = para.data.cpu.detach()
            optimizer.step()
            scheduler.step()
            loss_list[j]=loss.detach()
            if self.device=="cpu":
                pbar.set_description("Iter:{},lr:{},DampV:{},loss sum:{:.4}".format(j,lr,damp_vertical,np.sum(loss.detach().numpy())))
            else:
                pbar.set_description("Iter:{},lr:{},DampV:{},loss sum:{:.4}".format(j,lr,damp_vertical,np.sum(loss.cpu().detach().numpy())))
            # early stopping
            if j>0:
                if loss_list[j]>0.6:
                    trigger_times = 0
                if (loss_list[j]>loss_list[j-1]) or (torch.abs(loss_list[j]-loss_list[j-1])/torch.abs(loss_list[j-1])<eps):
                    trigger_times += 1
                    # print('Trigger Times:', trigger_times)
                    if trigger_times >= patience:
                        print("Early Stopping in Iteration:{}".format(j))
                        break
                else:
                    trigger_times = 0
            
            if torch.isnan(loss_list[j]):
                # loss_list[j] = loss_list[j+1]
                iter_vs = iter_vs[:j,:]
                iter_thick = iter_thick[:j,:]
                loss_list = loss_list[:j]
                print("Early Stopping in Iteration:{}".format(j))
                break
        ###############################
        # get the best inversoin result
        ###############################
        if self.device == "cpu":
            loss_list  = list2numpy(loss_list)
            iter_vs    = list2numpy(iter_vs)
            iter_thick = list2numpy(iter_thick)
        else:
            loss_list = loss_list.cpu().detach().numpy()
            iter_vs   = iter_vs.cpu().detach().numpy()
            iter_thick= iter_thick.cpu().detach().numpy()

        best_num = np.argmin(loss_list)
        if inversion_method.lower()=="vs-and-thick":
            inv_thick = iter_thick[best_num]
        else:
            inv_thick = list2numpy(init_thick)
        inv_vs =  iter_vs[best_num]

        if initialize_method == "Brocher":
            _,inv_vp,inv_vs,inv_rho = gen_model(thick=inv_thick,vs=inv_vs,area=True)
        elif initialize_method == "Constant":
            rho = self.model_param.rho
            _,inv_vp,inv_vs,inv_rho = gen_model1(thick=inv_thick,vs=inv_vs,vp_vs_ratio=vp_vs_ratio,rho=rho,area=True)
        elif initialize_method == "USArray":
            _,inv_vp,inv_vs,inv_rho = gen_model(thick=inv_thick,vs=inv_vs,area=True)
            mask = inv_vs>4.6
            mask_vp = []
            mask_rho = []
            for j in range(len(inv_vp[mask])):
                mask_vp.append(self.AK135_data[np.argmin(np.abs(self.AK135_data[:,2]-inv_vp[mask][j]))][2])
                mask_rho.append(self.AK135_data[np.argmin(np.abs(self.AK135_data[:,1]-inv_rho[mask][j]))][1])
            inv_vp[mask] = mask_vp
            inv_rho[mask] = mask_rho
        else:
            raise NameError("The initialize method can only select from [Brocher,Constant]")
        
        self.inv_model = {
            "thick":inv_thick.tolist(),
            "vp":inv_vp.tolist(),
            "vs":inv_vs.tolist(),
            "rho":inv_rho.tolist()
        }
        self.inv_process ={
            "iter_vs":iter_vs.tolist(),
            "iter_thick":iter_thick.tolist(),
            "loss":loss_list.tolist()
        }

class iter_cal_grad(torch.nn.Module):
    """
    The forward operator (Computational graph construction ) and gradient calculation
    ---------------------------------------------------------------------------------
    d : thickness
        => 1-D tensor 
    a : Compression wave velocity
        => 1-D tensor
    b : shear wave velocity
        => 1-D tensor
    rho : density
        => 1-D tenosr
    Clist : the determin phase velocity 
        => 1-D tensor: e.g. [1,1.001,1.002,1.003,...,2.001,0.002,...3.999,4]
    damp : the vertical damp
        => float
    compress : Whether to compress the value of the output result forward operator determinant (-1, 1)
        => default: True
    compress_method :
        => string: "exp"
    normalized : Whether to use the same frequency of all positive determinant values for normalization 
        => boolean: True
    inversion_method:
        => string: "vs"
        => string: "VsAndThick"
    initial_method:
        => string: None
        => string: "Brocher"
        => string: "Constant"
    vp_vs_ratio: the ratio of vs/vp (for "Constant" initial_method)
        => float
    AK135_data: additional data for vs>4.6km/s
        => array
    device : cpu/gpu device
        => string: "cpu"
        => string: "cuda:0"
    """
    def __init__(self,d,a,b,rho,Clist,damp,compress,normalized,compress_method,
                inversion_method,initial_method,vp_vs_ratio,AK135_data=[],device="cpu"):
        super(iter_cal_grad,self).__init__()
        self.device = device
        self.d = d
        self.a = a
        self.rho = rho
        self.b = torch.nn.Parameter(b)
        if inversion_method.lower()=="vs-and-thick":
            self.d = torch.nn.Parameter(d)
        self.Clist = Clist
        wave = "rayleigh"
        algorithm = "dunkin"
        self.ifunc = ifunc_list[algorithm][wave]
        # Check for water layer
        self.llw = 0 if b[0] <= 0.0 else -1
        # Model Constraint
        nL = len(d)
        self.damp = damp
        L0 = np.diag(-1*np.ones(nL-1),1) + np.diag(-1*np.ones(nL-1),-1) + np.diag(2*np.ones(nL))
        L0[0,0] = 1
        L0[nL-1,nL-1] =1
        self.L = numpy2tensor(L0).to(device)
        # normalizing
        self.normalized = normalized
        self.compress = compress
        self.compress_method = compress_method
        self.initial_method = initial_method
        self.vp_vs_ratio = vp_vs_ratio
        # AK135
        self.AK135_data = numpy2tensor(list2numpy(AK135_data)).to(device)

    def forward(self,vlist,tlist):
        if self.initial_method == "Brocher":
            self.a = 0.9409 + 2.0947*self.b - 0.8206*self.b**2+ 0.2683*self.b**3 - 0.0251*self.b**4
            self.rho = 1.6612*self.a - 0.4721*self.a**2 + 0.0671*self.a**3 - 0.0043*self.a**4 + 0.000106*self.a**5
        elif self.initial_method == "Constant":
            self.a = self.b*self.vp_vs_ratio
        elif self.initial_method == "USArray":
            mask1 = self.b>4.6
            for j in range(len(self.a[mask1])):
                self.a[mask1][j] = self.AK135_data[torch.argmin(torch.abs(self.AK135_data[:,2]-self.a[mask1][j]))][2]
                self.rho[mask1][j] = self.AK135_data[torch.argmin(torch.abs(self.AK135_data[:,1]-self.rho[mask1][j]))][1]
            # other
            mask2 = self.b<=4.6
            self.a[mask2] = 0.9409 + 2.0947*self.b[mask2] - 0.8206*self.b[mask2]**2+ 0.2683*self.b[mask2]**3 - 0.0251*self.b[mask2]**4
            self.rho[mask2] = 1.6612*self.a[mask2] - 0.4721*self.a[mask2]**2 + 0.0671*self.a[mask2]**3 - 0.0043*self.a[mask2]**4 + 0.000106*self.a[mask2]**5
        F = surf_vector_iter_gpu.dltar_vector(vlist, tlist, self.d, self.a, self.b, self.rho, self.ifunc, self.llw,device=self.device)
        # compress
        if self.compress:
            if self.normalized:
                # compress with the normalized result
                with torch.no_grad():
                    Olist = tlist.reshape(1,-1)
                    det  = surf_matrix_iter_gpu.dltar_matrix(self.Clist, Olist, self.d, self.a, self.b, self.rho, self.ifunc, self.llw, device=self.device)
                F = F/(torch.max(det,dim=0).values - torch.min(det,dim=0).values)
                if self.compress_method=="log":
                    F = torch.log(torch.abs(F)+1)
                elif self.compress_method == "exp":
                    det = det/(torch.max(det,dim=0).values - torch.min(det,dim=0).values)
                    F = (1e-1)**torch.abs(F) - 1
                    det = (1e-1)**torch.abs(det) - 1
            else:
                F = 1e-1**torch.abs(F) - 1
        F_all = torch.sum(torch.abs(F))/len(F)
        if self.damp>0:
            m_norm = self.damp*torch.matmul(self.L,self.b)/len(self.b)
            m_norm_all = torch.sum(torch.abs(m_norm))
            F_return = F_all+m_norm_all
        else:
            F_return = F_all
        return F_return

###########################################################################
##                          parallel inversion
###########################################################################
class multi_cal_grad(torch.nn.Module):
    """
    The forward operator (Computational graph construction ) and gradient calculation
    ---------------------------------------------------------------------------------
    d : thickness
        => 2-D tensor 
    a : Compression wave velocity
        => 2-D tensor
    b : shear wave velocity
        => 2-D tensor
    rho : density
        => 2-D tenosr
    Clist : the determin phase velocity 
        => 2-D tensor: e.g. [1,1.001,1.002,1.003,...,2.001,0.002,...3.999,4]
    damp : the vertical damp
        => float
    compress : Whether to compress the value of the output result forward operator determinant (-1, 1)
        => default: True
    compress_method :
        => string: "exp"
    normalized : Whether to use the same frequency of all positive determinant values for normalization 
        => boolean: True
    inversion_method:
        => string: "vs"
        => string: "VsAndThick"
    initial_method:
        => string: None
        => string: "Brocher"
        => string: "Constant"
    vp_vs_ratio: the ratio of vs/vp (for "Constant" initial_method)
        => float
    AK135_data: additional data for vs>4.6km/s
        => array
    device : cpu/gpu device
        => string: "cpu"
        => string: "cuda:0"
    """
    def __init__(self,d,a,b,rho,Clist,damp_vertical,damp_horizontal,compress,normalized,compress_method,inversion_method,initial_method,vp_vs_ratio,AK135_data=[],device="cpu"):
        super(multi_cal_grad,self).__init__()
        self.device = device
        self.d = d
        self.a = a
        self.rho = rho
        self.b = torch.nn.Parameter(b)
        if inversion_method.lower()=="vs-and-thick":
            self.d = torch.nn.Parameter(d)
        self.Clist = Clist
        wave = "rayleigh"
        algorithm = "dunkin"
        self.ifunc = ifunc_list[algorithm][wave]
        # Check for water layer
        self.llw = 0 if b[0][0] <= 0.0 else -1
        ################################################# Model Constraint   #######################################
        N_horizontal = d.shape[0]
        N_vertical   = d.shape[-1]
        self.damp_vertical = damp_vertical
        self.damp_horizontal = damp_horizontal
        # vertical constraint
        L0 = np.diag(-1*np.ones(N_vertical-1),1) + np.diag(-1*np.ones(N_vertical-1),-1) + np.diag(2*np.ones(N_vertical))
        L0[0,0] = 1
        L0[N_vertical-1,N_vertical-1] =1
        L0 = numpy2tensor(L0)
        L0 = tensor2numpy(L0.unsqueeze(dim=0))
        L0 = np.repeat(L0,d.shape[0],axis=0)
        self.L_vertical = numpy2tensor(L0).to(device)
        # Horizontal constraint
        L1 = np.diag(-1*np.ones(N_horizontal-1),1) + np.diag(-1*np.ones(N_horizontal-1),-1) + np.diag(2*np.ones(N_horizontal))
        L1[0,0] = 1
        L1[N_horizontal-1,N_horizontal-1] =1
        L1 = numpy2tensor(L1)
        self.L_horizontal = numpy2tensor(L1).to(device)
        ################################################# Model Normalization #######################################
        # normalizing
        self.normalized = normalized
        self.compress = compress
        self.compress_method = compress_method
        self.initial_method = initial_method
        self.vp_vs_ratio = vp_vs_ratio
        # AK135
        self.AK135_data = numpy2tensor(list2numpy(AK135_data)).to(device)

    def forward(self,vlist,tlist):
        if self.initial_method == "Brocher":
            self.a = 0.9409 + 2.0947*self.b - 0.8206*self.b**2+ 0.2683*self.b**3 - 0.0251*self.b**4
            self.rho = 1.6612*self.a - 0.4721*self.a**2 + 0.0671*self.a**3 - 0.0043*self.a**4 + 0.000106*self.a**5
        elif self.initial_method == "Constant":
            self.a = self.b*self.vp_vs_ratio
        elif self.initial_method == "USArray":
            # for k in range(self.b.shape[0]):
            # for the iteration of number of velocity model
            mask1 = self.b>4.6
            for j in range(len(self.a[mask1])):
                self.a[mask1][j] = self.AK135_data[torch.argmin(torch.abs(self.AK135_data[:,2]-self.a[mask1][j]))][2]
                self.rho[mask1][j] = self.AK135_data[torch.argmin(torch.abs(self.AK135_data[:,1]-self.rho[mask1][j]))][1]
            # other
            mask2 = self.b<=4.6
            self.a[mask2] = 0.9409 + 2.0947*self.b[mask2] - 0.8206*self.b[mask2]**2+ 0.2683*self.b[mask2]**3 - 0.0251*self.b[mask2]**4
            self.rho[mask2] = 1.6612*self.a[mask2] - 0.4721*self.a[mask2]**2 + 0.0671*self.a[mask2]**3 - 0.0043*self.a[mask2]**4 + 0.000106*self.a[mask2]**5

        F = surf_vector_all_gpu.dltar_vector(vlist, tlist, self.d, self.a, self.b, self.rho, self.ifunc, self.llw,device=self.device)
        # compress
        if self.compress:
            if self.normalized:
                # compress with the normalized result
                with torch.no_grad():
                    det  = surf_matrix_all_gpu.dltar_matrix(self.Clist, tlist, self.d, self.a, self.b, self.rho, self.ifunc, self.llw,device=self.device)
                F = F/(torch.max(det,dim=1).values - torch.min(det,dim=1).values)
                if self.compress_method=="log":
                    F = torch.log(torch.abs(F)+1)
                elif self.compress_method == "exp":
                    det = det/(torch.max(det,dim=1).values - torch.min(det,dim=1).values).reshape(det.shape[0],1,det.shape[2])
                    F   = (1e-1)**torch.abs(F) - 1
            else:
                F = (1e-1)**torch.abs(F) - 1
        # model constraint
        F_all = torch.sum(torch.abs(F),dim=1)/F.shape[-1]

        if self.damp_vertical>0 :
            m_norm_vertical   = self.damp_vertical*torch.matmul(self.L_vertical,torch.unsqueeze(self.b,dim=2)).squeeze(dim=2)/self.b.shape[1]
            m_norm_vertical   = torch.sum(torch.abs(m_norm_vertical),dim=1)
        else:
            m_norm_vertical = 0
        if self.damp_horizontal>0:    
            m_norm_horizontal = self.damp_horizontal*torch.matmul(self.L_horizontal,self.b)/self.b.shape[1]
            m_norm_horizontal = torch.sum(torch.abs(m_norm_horizontal),dim=1)
        else:
            m_norm_horizontal = 0
            
        F_return          = F_all + m_norm_vertical + m_norm_horizontal
        return F_return

class multi_inversion():
    """
        several model are inversion simultaneous
    """
    def __init__(self,
                model_param,
                inv_param,
                init_model,
                pvs_obs,
                vsrange_sign = 'mul',
                vsrange = [0.1,2],
                AK135_data = [],
                device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
                ):
        self.init_model = init_model
        self.model_param = model_param
        self.inv_param = inv_param
        self.pvs_obs = pvs_obs
        self.vsrange_sign = vsrange_sign
        self.vsrange = vsrange
        self.AK135_data = AK135_data
        self.device = device
        # inversion result definition
        self.inv_model = {
            "vs":[],
            "vp":[],
            "rho":[],
            "thick":[]
        }
        self.inv_process ={
            "iter_vs":[],
            "iter_thick":[],
            "loss":[]
        }
        self._run()
    
    def _run(self):
        # multiple initial model 【station,vs/vp/thick/rho】
        init_vs = self.init_model.init_model["vs"]
        init_vp = self.init_model.init_model["vp"]
        init_thick = self.init_model.init_model["thick"]
        init_rho = self.init_model.init_model["rho"]
        b = numpy2tensor(list2numpy(init_vs)).to(self.device)
        a = numpy2tensor(list2numpy(init_vp)).to(self.device)
        rho = numpy2tensor(list2numpy(init_rho)).to(self.device)
        d = numpy2tensor(list2numpy(init_thick)).to(self.device)
        
        # observed dispersion data [[t,v],[t,v]] 【station,pvs_num,[t,v]]】
        pvs_obs = list2numpy(self.pvs_obs)
        tlist = numpy2tensor(pvs_obs[:,:,0]).to(self.device)
        vlist = numpy2tensor(pvs_obs[:,:,1]).to(self.device)
        
        # initialized the calculator
        damp_vertical   = self.inv_param.damp_vertical
        damp_horizontal = self.inv_param.damp_horizontal
        clist = self.model_param.clist
        Clist = torch.ones((vlist.shape[0],clist.shape[0]))*numpy2tensor(clist)
        Clist = Clist.to(self.device)
        
        compress = self.inv_param.compress
        normalized = self.inv_param.normalized
        compress_method = self.inv_param.compress_method
        inversion_method = self.inv_param.inversion_method
        layering_method = self.model_param.layering_method
        initialize_method = self.model_param.initialize_method
        vp_vs_ratio = self.model_param.vp_vs_ratio
        
        # the range of inversion thickness 
        # if layering_method == "LR" or layering_method =="LN":
        layer_mindepth = np.ones_like(init_vs)*self.init_model.init_model["layer_mindepth"]
        layer_maxdepth = np.ones_like(init_vs)*self.init_model.init_model["layer_maxdepth"]
        
        calculator = multi_cal_grad(d,a,b,rho,Clist,damp_vertical,damp_horizontal,compress,normalized,compress_method,inversion_method,initialize_method,vp_vs_ratio,self.AK135_data,device=self.device)
        calculator = calculator.to(self.device)
        
        # optimizer
        lr = self.inv_param.lr
        iteration = self.inv_param.iteration
        step_size = self.inv_param.step_size
        gamma = self.inv_param.gamma
        if self.inv_param.optimizer == "Adam":
            optimizer = torch.optim.Adam(calculator.parameters(),lr = lr)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
        else:
            raise NameError("The input optimizer {} can not find in Pytorch".format(self.inv_param.optimizer))
        
        # intermediate process data
        pbar = tqdm(range(iteration))
        iter_vs = torch.zeros((iteration,init_vs.shape[0],init_vs.shape[1])).to(self.device)
        iter_thick = torch.zeros((iteration,init_vs.shape[0],init_vs.shape[1])).to(self.device)
        loss_list = (torch.ones((iteration,init_vs.shape[0]))*10).to(self.device)
        
        # inversion boundary
        if self.vsrange_sign =="mul":
            lower_b = self.vsrange[0]*b
            upper_b = self.vsrange[1]*b
        else:
            lower_b = b[0]*torch.ones_like(b) + self.vsrange[0]
            upper_b = b[0]*torch.ones_like(b) + self.vsrange[1]
            lower_b[:,-1] = 4.5

        # early stoopint param
        patience = 20
        eps = 1e-4
        trigger_times = 0
        
        ###############################
        # single station inversion
        ###############################
        for j in pbar:
            loss = calculator(vlist,tlist)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss),retain_graph=True)
            if inversion_method.lower()=="vs-and-thick":
                for k,para in enumerate(calculator.parameters()):
                    # constrain the velocity
                    if k == 0:
                        para.data.clip_(min=lower_b,max=upper_b)
                        # velocity of last layer > the second layer from bottom
                        para.data[:,-1].clip_(min=(para.data[:,-2]).detach())
                        iter_vs[j] = para.data.detach()
                    # constrain the depth
                    elif k ==1:
                        with torch.no_grad():
                            if self.device=="cpu":
                                depth_temp = np.cumsum(para.data.detach().numpy())
                            else:
                                depth_temp = np.cumsum(para.data.cpu().detach().numpy(),axis=1)
                            if layering_method == "LR":
                                depth_temp = np.clip(depth_temp,a_min=layer_mindepth,a_max=layer_maxdepth)
                            elif layering_method == "LN":
                                for ind in range(depth_temp.shape[-1]):
                                    if ind == 0:
                                        depth_temp[:,ind] = np.clip(depth_temp[:,ind],a_min=layer_mindepth[:,ind],a_max=layer_maxdepth[:,ind])
                                    else:
                                        depth_temp[:,ind] = np.clip(depth_temp[:,ind],a_min=(depth_temp[:,ind-1]+layer_mindepth[:,ind]),a_max=layer_maxdepth[:,ind])
                            depth_temp = np.insert(depth_temp,0,0,axis=1)
                            thick_temp = np.diff(depth_temp,axis=1)
                            # depth at least wmin/3
                            thick_temp = np.clip(thick_temp,a_min=layer_mindepth[0,0],a_max=None)
                            para.data = numpy2tensor(thick_temp).to(self.device)
                            iter_thick[j] = para.data.detach()
            else:
                for para in calculator.parameters():
                    if torch.isnan(para.data).any():
                        nan_mask = torch.isnan(para.data)
                        para.data[nan_mask] = numpy2tensor(list2numpy(init_vs)).to(self.device)[nan_mask]
                        # print("Bad vel model ")
                        
                    para.data.clip_(min=lower_b,max=upper_b)
                    # velocity of last layer > the second layer from bottom
                    # para.data[:,-1].clip_(min=(para.data[:,-2]).detach())
                    # para.data[0].clip_(min=lower_b)
                    # para.data[-1].clip_(min=(para.data[-2]).detach()) 
                    iter_vs[j] = para.data.detach()
                    
            optimizer.step()
            scheduler.step()
            loss_list[j]=loss.detach()

        loss_mask = torch.isnan(loss_list)
        loss_list[loss_mask] = 10
        ###############################
        # get the best inversoin result
        ###############################
        if self.device=="cpu":
            loss_list  = tensor2numpy(loss_list.detach().numpy())
            iter_thick = tensor2numpy(iter_thick.detach())
            iter_vs    = tensor2numpy(iter_vs.detach())
        else:
            loss_list = tensor2numpy(loss_list.cpu().detach().numpy())
            iter_thick = tensor2numpy(iter_thick.cpu().detach())
            iter_vs = tensor2numpy(iter_vs.cpu().detach())
        best_num = np.argmin(loss_list,axis=0)
        inv_thick = list2numpy(init_thick)
        if inversion_method.lower()=="vs-and-thick":
            for i in range(inv_thick.shape[0]):
                inv_thick[i] = iter_thick[best_num[i],i,:]
        else:
            inv_thick = list2numpy(init_thick)
        inv_vs  = np.zeros_like(inv_thick)
        inv_vp  = np.zeros_like(inv_thick)
        inv_rho = np.zeros_like(inv_thick)
        if initialize_method == "Brocher":
            for i in range(inv_thick.shape[0]):
                _,inv_vp_temp,inv_vs_temp,inv_rho_temp = gen_model(thick=inv_thick[i],vs=iter_vs[best_num[i],i],area=True)
                inv_vs[i] = inv_vs_temp
                inv_vp[i] = inv_vp_temp
                inv_rho[i] = inv_rho_temp
        elif initialize_method == "Constant":
            rho = self.model_param.rho
            for i in range(inv_thick.shape[0]):
                _,inv_vp_temp,inv_vs_temp,inv_rho_temp = gen_model1(thick=inv_thick[i],vs=iter_vs[best_num[i],i],vp_vs_ratio=vp_vs_ratio,rho=rho,area=True)
                inv_vs[i] = inv_vs_temp
                inv_vp[i] = inv_vp_temp
                inv_rho[i] = inv_rho_temp
        elif initialize_method == "USArray":
            for k in range(inv_thick.shape[0]):
                _,inv_vp_single,inv_vs_single,inv_rho_single = gen_model(thick=inv_thick[k],vs=iter_vs[k][np.argmin(loss_list[k])],area=True)
                mask = inv_vs_single>4.6
                mask_vp = []
                mask_rho = []
                for j in range(len(inv_vp_single[mask])):
                    mask_vp.append(self.AK135_data[np.argmin(np.abs(self.AK135_data[:,2]-inv_vp_single[mask][j]))][2])
                    mask_rho.append(self.AK135_data[np.argmin(np.abs(self.AK135_data[:,1]-inv_rho_single[mask][j]))][1])
                inv_vp_single[mask] = mask_vp
                inv_rho_single[mask] = mask_rho
                inv_vs[k] = inv_vs_single
                inv_vp[k] = inv_vp_single
                inv_rho[k] = inv_rho_single
        else:
            raise NameError("The initialize method can only select from [Brocher,Constant]")
        
        self.inv_model = {
            "thick":inv_thick.tolist(),
            "vp":inv_vp.tolist(),
            "vs":inv_vs.tolist(),
            "rho":inv_rho.tolist()
        }
        self.inv_process ={
            "iter_vs":iter_vs.tolist(),
            "iter_thick":iter_thick.tolist(),
            "loss":loss_list.tolist()
        }