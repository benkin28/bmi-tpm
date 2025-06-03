import numpy as np
import torch
from utils.nnArchitectures.nnCoarseGrained import nnPANIS, nnmPANIS
import json
import matplotlib.pyplot as plt




class probabModel:
    def __init__(self, pde, stdInit=2, display_plots=False, lr=0.001, sigma_r=0, yFMode=True, randResBatchSize=None, reducedDim=None, dataset=None):  # It was stdInit=8
        self.pde = pde
        self.sigma_r = sigma_r
        self.yFMode = yFMode
        self.nele = pde.nele
        self.mean_px = pde.mean_px
        self.sigma_px = pde.sigma_px
        self.Nx_samp = pde.Nx_samp
        self.Nx_samp_phi = self.Nx_samp
        self.x = torch.zeros(self.Nx_samp_phi, 1)
        self.y = torch.zeros(self.Nx_samp_phi, self.nele)
        self.Nx_counter = 0
        self.guideExecCounter = 0
        self.modelExecCounter = 0
        self.dataDriven = True
        self.dataDrivenFlag = False

        self.lowUnif = -1.
        self.highUnif = 1.
        self.readData = 0
        self.xPoolSize = 1
        if self.readData == 1:
            print("Pool of input data x was read from file.")
        else:
            print("Pool of input data x generated for this run.")
            self.data_x = torch.normal(self.mean_px, self.sigma_px, size=(self.xPoolSize,))
            self.data_x = 4 * self.sigma_px * torch.rand(size=(self.xPoolSize,)) - 2 * self.sigma_px
            print(self.data_x)
            print(torch.exp(-self.data_x))
            self.data_x = torch.linspace(-1., 1., self.xPoolSize)


        self.constant_phi_max = torch.ones((pde.NofShFuncs, pde.NofShFuncs))
        self.phi_max = torch.rand((pde.NofShFuncs, 1), requires_grad=True) * 0.01 + torch.ones(pde.NofShFuncs, 1)
        self.phi_max = self.phi_max / torch.linalg.norm(self.phi_max)

        phi_max_leaf = self.phi_max.clone().detach().requires_grad_(True)
        self.phi_max = phi_max_leaf
        self.phiBase = torch.reshape(self.phi_max, [1, -1])

        self.phi_max_history = np.zeros((pde.NofShFuncs, 1))
        self.sigma_history = np.zeros((pde.NofShFuncs, 1))
        self.temp_res = []
        self.full_temp_res = []
        self.model_time = 0
        self.guide_time = 0
        self.sample_time = 0
        self.stdInit = stdInit
        self.randResBatchSize = randResBatchSize
        self.compToKeepInPCA = 4
        self.residualCorrector = torch.sqrt(torch.tensor(self.Nx_samp))

        self.validationIndex = 0


        self.reducedDim = reducedDim
        xx, yy = torch.meshgrid(torch.linspace(0, 1, self.reducedDim), torch.linspace(0, 1, self.reducedDim), indexing='ij')
        xxx, yyy = torch.meshgrid(torch.linspace(0, 1, self.pde.intPoints), torch.linspace(0, 1, self.pde.intPoints), indexing='ij')
        self.xtoXCnn = None
        self.neuralNetCG = None
        self.NTraining = 99
        self.I = torch.ones(self.pde.NofShFuncs)*(stdInit) 
        self.I.requires_grad_(True) 
        self.vDim = 10
        if yFMode:
            self.V = torch.ones((self.reducedDim**2, self.vDim))*(-3) +torch.randn(self.reducedDim**2, self.vDim)/10 #lower rank matrix from paper (track gradient) --> L
        else:
            self.V = torch.ones((self.pde.NofShFuncs, self.vDim))*(-3) +torch.randn(self.pde.NofShFuncs, self.vDim)/10
        self.V.requires_grad_(True)

        self.globalSigma = torch.tensor([float(stdInit)]) #sigma from paper
        print("Global Sigma: ", self.globalSigma)

        self.globalSigma.requires_grad_(True)
        if yFMode: 
            self.neuralNet = nnmPANIS(reducedDim=self.reducedDim, cgCnn=self.neuralNetCG, xtoXCnn=self.xtoXCnn, pde=pde, extraParams=[self.V, self.globalSigma, yFMode])
        else:
            #always this is relevant (yFMode=False)
            self.neuralNet = nnPANIS(reducedDim=self.reducedDim, cgCnn=self.neuralNetCG, xtoXCnn=self.xtoXCnn, pde=pde, extraParams=[self.V, self.globalSigma, yFMode])

        
        numOfPars = self.neuralNet.count_parameters()
        print("Number of NN Parameters Used: ", numOfPars)

        
        

        self.losss = torch.nn.MSELoss(reduction='mean')
        

        if True:
            self.xConst = torch.reshape(torch.randn(self.NTraining+1, self.pde.numOfInputs),
                                       [self.NTraining+1, 1, self.pde.numOfInputs])

        else:
            self.xConst = pde.sampX.view(pde.sampX.size(0), 1, pde.sampX.size(-1))

        if self.yFMode:
            self.yF = (torch.ones(self.pde.shapeFunc.aInvBCO.size(-1)) + 0.1 * torch.rand(self.pde.shapeFunc.aInvBCO.size(-1))).repeat(self.NTraining+1, 1).requires_grad_(True)
            
        if not self.yFMode:
            #Adam optimizer is used
            self.manualOptimizer = torch.optim.Adam(params=list(self.neuralNet.parameters())+ [self.V] + [self.globalSigma], lr=lr, maximize=True, amsgrad=False)
            
        else:
            
            self.manualOptimizer = torch.optim.Adam(params=list(self.neuralNet.parameters())+ [self.neuralNet.V] + [self.neuralNet.globalSigma] + [self.yF], lr=lr, maximize=True, amsgrad=False)
           
        self.ELBOLossHistory = torch.zeros(1, 1000)
       
        




    def sampleFromQxyMVN(self, Nx, useTheRealNN=True):
        
        if not self.dataDrivenFlag:
            x = torch.reshape(torch.randn(Nx, self.pde.numOfInputs),
                                       [Nx, 1, self.pde.numOfInputs]) #example top draw random x
            #x is drawn (line 6)

            if self.yFMode:
                indices = torch.randperm(self.xConst.size(0))[:Nx]
                x = self.xConst[indices]
                #x are the function inputs and are simply assumed to follow a standard normal distribution. Why
                y = self.neuralNet.forward(x) + torch.einsum('...ij,...j->...i', self.pde.shapeFunc.aInvBCO, 1 * self.yF[indices]).view(x.size(0), 1, -1)
                #More info on this. What object are we multiplying with what
            else:
                y = self.neuralNet.forward(x) + torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(Nx, 1, self.vDim)) + torch.pow(10, self.globalSigma/2.) * torch.randn(Nx, 1, self.pde.NofShFuncs)
                # why is all of thus to the power using 10 as a base? i thought this was equ 17
            
            y = y 
        return x, y
    

    
 
    
    
    def entropyMvnShortAndStable(self, V, sigma):
        manualEntropyStable = 0.5 * torch.logdet(torch.einsum('ij,kj->ik', V, V)+ sigma**2 * torch.eye(self.pde.NofShFuncs))
        return manualEntropyStable
    
    def entropyMvnShortAndStableNN(self):
        manualEntropyStable = 0.5 * torch.logdet(torch.einsum('ij,kj->ik', torch.pow(10, self.neuralNet.V), torch.pow(10, self.neuralNet.V)) \
                                                 + torch.pow(10, self.neuralNet.globalSigma/2)**2 * torch.eye(self.reducedDim**2))

        return manualEntropyStable
    
    
    
    
    def logProbMvnUninformative(self, x):
        #2nd term in appendix c
        manualLogProbShortest = - 0.5 * torch.einsum('...i,...i->...', x, x) * 10**(-8)
        # why is this multiplied by 10**(-8) and not 10**(-16)? Is R simply assumed
        return manualLogProbShortest
    

    
    def sviStep(self):
        MultipleActiveClasses = False
        # What is MultipleActiveClasses????? --> irrelevant for now
        self.manualOptimizer.zero_grad()
        #gradient set to zero from prev step
        
        x, y = self.sampleFromQxyMVN(self.Nx_samp)
        indices1 = torch.randperm((self.pde.shapeFuncsDim+1)**2)[:self.randResBatchSize]
        #random indeces from Monte carlo approx (categprical dist)
        phi = torch.eye(((self.pde.shapeFuncsDim+1)**2)).unsqueeze(0).expand(x.size(0), -1, -1)[:, indices1, :]
        #coeff of weight functions
        if MultipleActiveClasses:
            indices2 = torch.randperm(49)[:10]
            phi2 = torch.eye(49).unsqueeze(0).expand(x.size(0), -1, -1)[:, indices2, :]
        self.phi = phi
        

        if self.yFMode:
            entropy = self.entropyMvnShortAndStableNN()
        else:
            entropy = self.entropyMvnShortAndStable(torch.pow(10, self.V), torch.pow(10, self.globalSigma/2)) 
        logProb = torch.mean(self.logProbMvnUninformative(y))
        res = self.pde.calcSingleResGeneralParallel(x, y, phi)
        # is this only from a single x,y pair or from multiple pairs?
        # if MultipleActiveClasses:
        #     res2 = self.pde.calcSingleResGeneralParallel2(x, y, phi2)
        #     res = torch.cat((res, res2), dim=1)

        absRes = torch.abs(res)
        likelihood = - 1./2./self.sigma_r * torch.sum(torch.mean(absRes, dim=0), dim=0) #Monte Carlo approx
        #first term of ELBO
        yField = self.pde.shapeFunc.cTrialSolutionParallel(y)

        # # Convert yField to a list and save to a JSON file
        # yField_list = yField.detach().cpu().numpy().tolist()
        # with open('y_solution.json', 'w') as json_file:
        #     json.dump(yField_list, json_file)

        # # Plot yField and save the plot
        # plt.plot(np.array(yField_list).flatten())
        # plt.xlabel('Index')
        # plt.ylabel('Value')
        # plt.title('yField Plot')
        # plt.savefig('yField_plot.png')
        # plt.close()


        loss = likelihood + logProb + entropy
        #lines 9-10
        loss.backward(retain_graph=True)
        self.manualOptimizer.step()
        if res.size(0) > 1 or self.Nx_samp == 1:
            self.temp_res.append(torch.mean(torch.abs(res)))

        self.neuralNet.iter = self.neuralNet.iter + 1

        return loss.clone().detach()



    def removeSamples(self):
        self.temp_res = []
        self.full_temp_res = []




    
    def samplePosteriorMvn(self, x, Nx=1, Navg=1): #almost all of algo 2
        
        xReshaped = torch.reshape(x, [x.size(0), 1, -1])

        dimOut = self.pde.NofShFuncs

        if Nx <= 10:
            if self.yFMode:
                mean = self.neuralNet.forwardMultiple(xReshaped, Navg=Navg)
            else:
                mean = self.neuralNet.forward(xReshaped).view(Nx, 1, -1).repeat(1, Navg, 1)
    
            
       
        NN = int(Nx/10)
        mean = torch.zeros(Nx, Navg, dimOut)
        for i in range(0, NN):
            if self.yFMode:
                mean[10*i:10*i+10] = self.neuralNet.forwardMultiple(xReshaped[10*i:10*i+10], Navg=Navg)
            else:
                mean[10*i:10*i+10] = self.neuralNet.forward(xReshaped[10*i:10*i+10]).view(10, 1, -1).repeat(1, Navg, 1)
            
        if self.yFMode:
            ySamples = mean
        
        Sigma = torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(mean.size(0), Navg, self.vDim)) + \
            torch.pow(10, self.globalSigma/2.) * torch.randn(mean.size(0), Navg, mean.size(-1))
        ySamples = mean + Sigma

        
        return ySamples, mean
    def samplePosteriorMvn_fromX(self, X, Nx=1, Navg=1):
        """
        Sample from the posterior distribution of Y given PDE coefficients X.
        This version assumes X is already the PDE coefficients.

        Nx = number of PDE coefficient vectors
        Navg = how many posterior draws per X (if you have a distribution)
        """
        # X is [Nx, <maybe some dimension>] 
        # We reshape if needed:
        X_reshaped = X.view(Nx, 1, -1)  # e.g., so it's [Nx, 1, NofShFuncs]

        dimOut = self.pde.NofShFuncs

        # 1) Forward pass to get PDE solution "mean" (or the main forward)
        #    We chunk if Nx is large. Example:
        if Nx <= 10:
            # direct call
            mean = self.neuralNet.forward_fromX(X_reshaped)  # <--- use our new method
            # shape = [Nx, 1, dimOut]
            mean = mean.repeat(1, Navg, 1)  # replicate along the second dim if we want multiple draws
        else:
            # chunk in sets of 10
            mean_list = []
            NN = Nx // 10
            for i in range(NN):
                chunk = X_reshaped[10*i : 10*(i+1)]
                chunk_out = self.neuralNet.forward_fromX(chunk)  # shape [10, 1, dimOut]
                chunk_out = chunk_out.repeat(1, Navg, 1)
                mean_list.append(chunk_out)
            mean = torch.cat(mean_list, dim=0)
            # If Nx is not multiple of 10, handle remainder, etc.

        # 2) If self.yFMode, that probably means the forward call already includes random sampling?
        #    Or we do it ourselves like in your original snippet.  Suppose you want the same logic:
        if self.yFMode:
            # If forward_fromX() already adds random draws, 'mean' might already be random each call.
            # You might just rename it 'ySamples' directly:
            ySamples = mean
        else:
            # same logic as your original code
            # "mean" is shape [Nx, Navg, dimOut]
            # plus random noise from V, globalSigma, etc.:
            Sigma = (
                torch.einsum('ij,...j->...i', torch.pow(10, self.V), torch.randn(mean.size(0), Navg, self.vDim))
                + torch.pow(10, self.globalSigma/2.0) * torch.randn(mean.size(0), Navg, dimOut)
            )
            ySamples = mean + Sigma

        return ySamples, mean

    def calcCovarianceMatrix_III(self):
        V = torch.pow(10, self.V)
        globalSigma = torch.pow(10, self.globalSigma)
        cov = V @ V.t() + torch.diag(globalSigma)
        return cov

    def samplePosteriorMean(self, x):
        if self.pde.numOfInputs == 1:
            x = torch.reshape(x, [1, 1, -1])
        elif self.pde.numOfInputs > 1:
            x = torch.ones(self.pde.numOfInputs) * x
            x = torch.reshape(x, [1, 1, -1])
        res = self.neuralNet.forward(x)
        if self.doPCA:
            res = self.pde.meanSampYCoeff + torch.einsum('ji,i->j', self.usedEigVecs,
                                                         res)
        return res






