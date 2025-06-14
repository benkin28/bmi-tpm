### Importing Libraries ###
import numpy as np
import torch
import os
import time
### Importing Manual Modules ###
from model.ProbModels.cgCnnMvn import probabModel
from model.pde.pdeForm2D import pdeForm
from utils.PostProcessing import postProcessing
from utils.tempData import storingData
from input import *
from utils.variousFunctions import calcRSquared, calcEpsilon, makeCGProjection, setupDevice, createFolderIfNotExists, memoryOfTensor, list_tensors
from utils.saveEvaluationDataset import saveDatasetAll, importDatasetAll
from model.pde.pdeTrueSolFenics import solve_pde
import warnings
import json
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt



### Device Selection ###
device = setupDevice(cudaIndex, device, dataType)

### Constructing Post Processing Instance ###
post = postProcessing(path='./results/data/', displayPlots=display_plots)

# Print internals of postProcessing instance
print(f"PostProcessing Path: {post.path}")
print(f"Display Plots: {post.displayPlots}")

# Save some data and print confirmation
post.save([['numOfTestSamples', np.array(numOfTestSamples)]])
print("Saved number of test samples.")

### Reading Existing Dataset (if it exists) ###
if (not createNewCondField) and not importDatasetOption:
    sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = post.readTestSample(numOfTestSamples)
    print("Read existing test samples.")
    torch.save(sampX, './model/pde/RefSolutions/condFields/' + 'sampX.dat')
    print("Saved sampX to file.")

### Importing Dataset ###
if importDatasetOption and not createNewCondField:
    if os.path.exists(saveDatasetName):
        print("Dataset:" + saveDatasetName + " exists. Importing...")
        sampCond, sampSolFenics, sampX, sampYCoeff, sampSol, gpEigVals, gpEigVecs = importDatasetAll(datapath=saveDatasetName, device=device)
        createFolderIfNotExists('./model/pde/RefSolutions/condFields/')
        torch.save(sampX, './model/pde/RefSolutions/condFields/' + 'sampX.dat')
        torch.save(gpEigVals, './model/pde/RefSolutions/condFields/' + 'gpEigVals.dat')
        torch.save(gpEigVecs, './model/pde/RefSolutions/condFields/' + 'gpEigVecs.dat')
        print("Imported and saved dataset.")
    else:
        raise ValueError("Dataset:" + saveDatasetName + " does not exist. Ignoring the command")

### Definition and Form of the PDE ###
pde = pdeForm(nele, shapeFuncsDim, mean_px, sigma_px, sigma_r, Nx_samp, createNewCondField, device, post, rhs=rhs, reducedDim=reducedDim, options=options)

# Print some internals of the pdeForm instance
print(f"Shape Functions Dimension: {pde.shapeFuncsDim}")
print(f"Number of Elements: {pde.nele}")
print(f"Grid Shape: {pde.grid.shape}")
print(f"Grid first 10: {pde.grid}")
print(f"GridW Shape: {pde.gridW.shape}")
print(f"GridW2 Shape: {pde.gridW2.shape}")
print(f"Number of Shape Functions: {pde.NofShFuncs}")
print(f"Number of Shape Functions W: {pde.NofShFuncsW}")
print(f"Node Coordinates: {pde.node_corrs.shape}")
print(f"Options: {pde.options}")
print(f"Integration Points: {pde.intPoints}")
print(f"Mean px: {pde.mean_px}")
print(f"Sigma px: {pde.sigma_px}")
print(f"Sigma r: {pde.sigma_r}")
print(f"Number of Samples: {pde.Nx_samp}")
print(f"RHS: {pde.rhs}")
print(f"Boundary Condition: {pde.uBc}")
print(f"Length Scale: {pde.lengthScale}")
print(f"GP Sigma: {pde.gpSigma}")
print(f"Fraction: {pde.fraction}")
print(f"Phase High: {pde.phaseHigh}")
print(f"Phase Low: {pde.phaseLow}")


### Calculation of the Coarse-Grained Projection ###
ProjectionIsTheTrueSolution = True
if ProjectionIsTheTrueSolution and (not createNewCondField) and compareWithCGProjection:
    xTest, yTest, yProj, yProjTotal = makeCGProjection(pde, ProjectionIsTheTrueSolution, createNewCondField, compareWithCGProjection, sampX, sampSolFenics, sampYCoeff)

### Save the Reference Solution Data ###
if importDatasetOption and not createNewCondField:
    pde.saveFields(sampSol, sampSolFenics, sampYCoeff, sampCond, sampX)



### Initialization of Variables ###
tempData = storingData()
residual_history = []
sigmaHistory = []
movAvgResHist = []
residuals_history = []
hist_elbo = []
hist_iter = []
sigmaEvolution = []
varNormEvolution = []
RSquaredHistory = torch.zeros(100)
progress_perc = 0
t = time.time()


### Creating instance of the Probabilistic Model ###
samples = probabModel(pde, stdInit=stdInit, lr=lr, sigma_r=sigma_r, yFMode=yFMode, randResBatchSize=randResBatchSize, reducedDim=reducedDim)
samples.neuralNet.train()



### SVI Optimization Loop ###
for i in range(IterSvi):


    elbo = samples.sviStep() 

    
    ### Storing Current Convergence Data ###
    current_residual = samples.temp_res[0].item()
    residual_history.append(current_residual)
    sigmaHistory.append(samples.I.detach().clone())
    hist_elbo.append(elbo)
    hist_iter.append(torch.tensor(float(i)))
    samples.removeSamples() # Freeing up memory from lists of samples


    ### Printing Progress and Storing Convergence Data ###
    if (i + 1) % (IterSvi * 0.01) == 0:

        
        samples.neuralNet.eval() #Evaluation --> more research on those different modes

        progress_perc = progress_perc + 1

        ### Tempering of Non-Linear PDE parameters
        pchange = 70
        if not pde.Linear:
            if progress_perc < pchange:
                pde.alphaPDE = options['alpha'] * ((progress_perc-1)/ pchange)**10
                pde.shapeFunc.alphaPDE = options['alpha'] * ((progress_perc-1)/ pchange)**10
            else:
                pde.alphaPDE = options['alpha']
                pde.shapeFunc.alphaPDE = options['alpha']
            if pde.alphaPDE != options['alpha'] and progress_perc > (pchange+1):
                raise ValueError("Incorrect Tempering of Non-Linear Parameters")
            
        ## RSquared Calculation
        if True and createNewCondField == False:

            if compareWithCGProjection:
                yTrueT = yProj
            else:
                yTrueT = sampSolFenics
            yyS, yyMean = samples.samplePosteriorMvn(pde.sampX, Nx=numOfTestSamples, Navg=Navg)
            yPredT, yPredStd = pde.createMeanPredictionsParallel(yyS.clone().detach(), yyMean.clone().detach())[1:]
            yPredT = yPredT.clone().detach()
            RSquared = calcRSquared(yTrueT, yPredT)
            RSquaredHistory[progress_perc-1] += RSquared
            relativeL2Error = calcEpsilon(yTrueT, yPredT)
            tempData.appendToFile(torch.reshape(RSquared, [-1]).detach().cpu(), 'RSquared.dat')
            print("RSquared is: ", RSquared)
            print("Relative L2 Error: ", relativeL2Error)
        
        

        ### Printing and Saving Data ###
        if progress_perc > 1:
            print("Progress: ", progress_perc, "/", 100, " ELBO", "{:.2e}".format(elbo), "Residual",
                    "{:.2e}".format(min(residual_history[-int(1 / 100):]), "sigma_r", sigma_r))
            print("Moving Avg. of Residual (Last 50 Values)", "{:.2e}".format(sum(residual_history[-50:]) / 50.))
            samples.V = samples.neuralNet.V
            samples.globalSigma = samples.neuralNet.globalSigma
            covv = samples.calcCovarianceMatrix_III()
            sigmaEvolution.append(torch.pow(10,samples.globalSigma.clone().detach()))
            varNormEvolution.append(torch.mean(torch.abs(covv.clone().detach())))

        samples.neuralNet.train() #Training (wooooooooooo)
            

samples.neuralNet.eval()

print("Training Finished.")

if saveModelOption:
    torch.save(samples.neuralNet.state_dict(), './utils/trainedNNs/trainedCG'+'dim'+str(pde.numOfInputs)+options['contrastRatio']+options['volumeFraction']+'D'+str(int(options['boundaryCondition']))+'.pth')
    torch.save(samples.V, './utils/trainedNNs/trainedCG'+'V_dim'+str(pde.numOfInputs)+options['contrastRatio']+options['volumeFraction']+'D'+str(int(options['boundaryCondition']))+'.pth')
    torch.save(samples.globalSigma, './utils/trainedNNs/trainedCG'+'globalSigma_dim'+str(pde.numOfInputs)+options['contrastRatio']+options['volumeFraction']+'D'+str(int(options['boundaryCondition']))+'.pth')
    print("Saving NN parameters")
elapsed = time.time() - t
print("Total time: ", elapsed)





### Post Processing ###

### Out of Distribution Prediction / Loading Pre-trained NNs ###
if outOfDistributionPrediction:
    samples.neuralNet.load_state_dict(torch.load('./utils/trainedNNs/trainedCGdim1024CR10FR50D0.pth'))
    samples.neuralNet.pde = pde
    Navg = 100
    samples.globalSigma = torch.load('./utils/trainedNNs/trainedCGglobalSigma_dim1024CR10FR50D0.pth').to(device)
    samples.V = torch.load('./utils/trainedNNs/trainedCGV_dim1024CR10FR50D0.pth').to(device)
    samples.neuralNet.V = samples.V
    samples.neuralNet.globalSigma = samples.globalSigma
    FR = options['volumeFractionOutOfDistribution']
    xTest, yTest, sampX = importDatasetAll(datapath=saveDatasetNameOutOfDist, device=device)[:3]
    pde.gpEigVals, pde.gpEigVecs = importDatasetAll(datapath=saveDatasetNameOutOfDist, device=device)[-2:]
    samples.neuralNet.eval()
post.save([['outOfDistributionPrediction', outOfDistributionPrediction]])


### Post Processing for Plotting Coarse Grained Fields ###
CGPlotting = True
if CGPlotting and not createNewCondField:
    #instead of sampX, draw from normal distribution
    #mean=0, std=I
    xKLE = pde.sampX.view(pde.sampX.size(0), 1, pde.sampX.size(1)) #original small x
    print("xKLE size: ", xKLE.size())
    x = pde.gpExpansionExponentialParallel(xKLE) #add noise (sigma + random draw from standard normal distribution)
    xKLE = torch.reshape(xKLE, [xKLE.size(dim=0), 1, -1]).to('cuda:0')

    if numOfTestSamples <= 10:
        y = samples.neuralNet.forward(xKLE.to(device)).detach().cpu()
    else:
        NN = int(numOfTestSamples/10)
        dimOut = pde.NofShFuncs
        y = torch.zeros([numOfTestSamples, 1,  dimOut])
        for i in range(0, NN):
            y[10*i:10*i+10] = samples.neuralNet.forward(xKLE.to(device)[10*i:10*i+10])

    y = pde.shapeFunc.cTrialSolutionParallel(y.to(device)).cpu()
    y = torch.reshape(y, [y.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()
    print("y size: ", y.size())
    
    x = x.squeeze(1)

    XCG = samples.neuralNet.xtoXCnn(xKLE).squeeze(1) #capital X
    data = {
        "x": x.tolist(),
        "XCG": XCG.tolist()
    }

    with open("data.json", "w") as f:
        json.dump(data, f)

    print("Saved x and XCG to data.json.")
    if options['boundaryCondition'] == 'sinx':
        sinX = True
    else:
        sinX = False
    YCGFenics = pde.shapeFunc.solveCGPDE(XCG[:10], f=pde.rhs, uBc=pde.uBc)
    YCG = YCGFenics.detach().cpu()
    YCGFenics = YCG.detach().cpu()
    yFenics = solve_pde(XCG[0].clone().detach().cpu(), rhs=-100., uBc=pde.uBc, sinX=sinX, options=options)
    yCG = y.detach().cpu()
    
    xCG = x.detach().cpu()
    

    randomShapeFuncs = torch.sum(pde.shapeFunc.shapeFuncW[torch.randperm(pde.shapeFunc.shapeFuncW.size(0))[:randResBatchSize]], dim=0)
    post.save([
    ['randShapeFuncs', randomShapeFuncs.detach().cpu()],
    ['xCG', xCG.detach().cpu()],
    ['yCG', yCG.detach().cpu()],
    ['XCG', XCG.detach().cpu()],
    ['YCG', YCG.detach().cpu()],
    ['YCGFenics', YCG.detach().cpu()]])
    if compareWithCGProjection:
        post.save([['yProjT', yProj.detach().cpu()]])
    else:
        post.save([['yProjT', sampSolFenics.detach().cpu()]])



### Saving Data for Post Processing ###
if createNewCondField:
    writingList = [['intGrid', pde.sgrid],
                   ['rbfGrid', pde.grid],
                   ['rbfGridW', pde.gridW],
                   ['gpEigVals', pde.gpEigVals],
                   ['gpEigVecs', pde.gpEigVecs]]
    post.save(writingList)
    sampSol, sampSolFenics, sampCond, sampX, sampYCoeff = pde.produceTestSample(Nx=1, post=post)
    sampX=sampX.repeat(numOfTestSamples, 1)
    x_testing_samples = sampX
    xKLE = x_testing_samples.view(numOfTestSamples, 1, 24) #original small x
    x = pde.gpExpansionExponentialParallel(xKLE) #add noise (sigma + random draw from standard normal distribution)
    # Reshape xKLE and repeat for the specified number of test samples
    xKLE = torch.reshape(xKLE, [xKLE.size(dim=0), 1, -1]).to('cuda:0')

    # Remove unnecessary dimensions from x
    x = x.squeeze(1)

    # Generate XCG using the neural network's xtoXCnn function
    XCG = samples.neuralNet.xtoXCnn(xKLE).squeeze(1)

    # Forward pass through the neural network
    y = samples.neuralNet.forward(xKLE.to(device)).detach().cpu()

    # Apply the trial solution transformation
    y = pde.shapeFunc.cTrialSolutionParallel(y.to(device)).cpu()

    # Reshape y to match the desired dimensions
    y = torch.reshape(y, [y.size(dim=0), pde.sgrid.size(dim=1), pde.sgrid.size(dim=2)]).detach().cpu()
    print("y size: ", y.size())
    # Plotting the grid
    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(x[0].cpu().numpy(), cmap='viridis')
    # plt.title('Original Grid')
    # plt.colorbar()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 2)
    plt.imshow(y[0].cpu().numpy(), cmap='viridis')
    plt.title('Predicted Grid')
    plt.colorbar()

    plt.savefig("Z.png")    # data = {
    # Plotting the grid
    #     "XCG": XCG.tolist(),
    #     "y": y.tolist()
    # }

    # with open("samedata.json", "w") as f:
    #     json.dump(data, f)

else:
    post.save([['xKLE', pde.sampX], ['RSquared', [RSquared]],])

if saveDatasetOption:
    saveDatasetAll(sampCond, sampSolFenics, sampX, sampYCoeff, sampSol, pde.gpEigVals, pde.gpEigVecs, createNewCondField, path=saveDatasetName)

yyS, yyMean = samples.samplePosteriorMvn(sampX, Nx=numOfTestSamples, Navg=Navg)
sampSamplesMean, sampSamplesStd = pde.createMeanPredictionsParallel(yyS.clone().detach(), yyMean.clone().detach())[1:]
post.save([['sampSamplesMean', sampSamplesMean],
        ['sampSamplesStd', sampSamplesStd],
        ['xTest', sampCond]])

writingList = [ ['createNewCondField',  createNewCondField],
                ['intGrid',  pde.sgrid],
                ['rbfGrid',  pde.grid],
                ['residualEvolution', residual_history],
                ['elboEvolution', torch.stack(hist_elbo).detach().cpu().numpy()],
                ['iterEvolution', torch.stack(hist_iter).detach().cpu().numpy()],
                ['gpEigVals', pde.gpEigVals],
                ['gpEigVecs', pde.gpEigVecs],
                ['ELBOLossHistory', samples.ELBOLossHistory],
                ['RSquaredHistory', RSquaredHistory],
                ['yFENICSTrue', sampSolFenics.clone().detach().cpu()]
                ]
post.save(writingList)
if compareWithCGProjection:
    post.save([['yTest', yProj.detach().cpu()]])
else:
    post.save([['yTest', sampSolFenics.detach().cpu()]])



print("\n", "Post Processing Begins: \n")
# # Ensure the dataPairs directory exists
# os.makedirs('./dataPairs', exist_ok=True)
    
#     # Read data from data.json
# with open('data.json', 'r') as file:
#         data = json.load(file)
    
# fineGrids = data['x']
# coarseGrids = data['XCG']
    
#     # Plot each grid and save the plots
# for i, (fineGrid, coarseGrid) in enumerate(zip(fineGrids, coarseGrids)):
#         plot_grid(fineGrid, coarseGrid, i)

# post.producePlots()

