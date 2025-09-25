# KANG2P:Predicting phenotypes from genotypes and predicted gene expression levels  using Kolmogorov-Arnold networks

## **Abstract**
Genomic prediction has emerged as a powerful tool with applications spanning human disease risk assessment and plant/animal breeding. Kolmogorov–Arnold networks (KANs) have risen as a potential substitute for multilayer perceptrons within dense fully connected networks. Substantial efforts have been dedicated to incorporating KANs into diverse deep learning frameworks in computer vision and natural language processing. However, the integration of KANs into deep learning models for genomic prediction remains unexplored. Here, we present KANG2P, a novel method designed to enhance genomic prediction accuracy. We extensively evaluated KANG2P on diverse datasets, including ALS and Parkinson's disease, as well as maize and rice datasets. In classification and regression tasks, KANG2P demonstrated superior performance compared to existing methods. By integrating both genotype and predicted expression data, KANG2P leverages a unique architecture to better capture complex genetic patterns. We also propose an explainable approach for spline functions to identify disease-related functional genes. Our results highlight KANG2P's potential to advance research and applications in both biomedical and agricultural fields.

## **workflow of KANG2P**
![image](https://github.com/RayLing88/KANG2P_test/blob/master/img/KANG2P_workflow.png)

## **Installation and dependencies**
- Linux OS; GPU hardware support
- Python >= v3.7
- PyTorch v1.5.0 (GPU)
- sklearn v0.22.2

No need to install the source code. Dependencies can be installed with a few minutes.

## Imputing gene expression levels
- For Diseases, run_PrediXcan.sh could be used to predict gene expression levels. In the run_PrediXcan.sh shell script, [MetaXcan](https://github.com/hakyimlab/MetaXcan) was called upon.
- For plant, SNP2Expression_v1.0.R could be used to build gene expression levels prediction models for each gene.
- VisualizeSNP2ExpressionAcc.R script was designed to visualize of the accuracy for gene-expression prediction models.

## Feature Selection
- MergeAndWilcox.R script was used to perform differential expression analysis.
- MRMR.py scirpt ,the maximum relevance minimum redundancy (MRMR) algorithm was employed to identify the optimal gene subset.

## Train KANG2P
- For Diseases, Within the DiseaseKANG2P folder, the DualOmicsModel.py can be used to train the KANG2P model.
- For Plant regression task, Within the PlantKANG2P folder, the DualOmicsModel.py can be used to train the KANG2P model.

## Predict
- For Diseases, Within the DiseaseKANG2P folder, the Predict.py script can be used to make prediction.
- For Plant, Within the PlantKANG2P folder, the Predict.py script can be used to make prediction.

## Explain KANG2P
- Explain_KAN_spline.py script was used to calculate the contribution of each latent variable to the classification labels.
- PCCMultSpline.R script was used to calculate the contribution of each gene to the classification labels.
- Mask_Gene.py was used to evaluate the classification performance of the 913 crucial genes while masking all remaining genes on the independent test set.
- Random_Mask_Gene.py was to assess the classification performance of 913 randomly sampled genes across 1000 repetitions.

## Traditional GS mdoel
- runG2P_optparse.R contains 9 state-of-the-art GS models ("BayesC", "BL", "BRR", "RKHS", "RRBLUP","LASSO", "SPLS", "SVR", "RFR") for regression task.
- basicML.py script contains 4 models ("LR","Random Forest","SVC","Adaboost") for classification task.

## Raw data 
- the ALS dataset from project [Mine](https://www.projectmine.com), dbGaP Study Accession: phs003146.v1.p1,
- the PD dataset, dbGaP Study Accession:phs000918.v1.p1; 
- the rice 18K dataset was available at [figshare](https://figshare.com/articles/dataset/NAM_variations/19166475)
- the rice1495 hybrid line dataset was available at [CropGS](https://iagr.genomics.cn/CropGS)
- the maize CUBIC dataset was available at [G2P-env](https://github.com/G2P-env/G2P).

## Questions and errors
If you have a question, error, bug, or problem, please use the [Github issue page](https://github.com/RayLing88/KANG2P_test/issues).

## Contact  
  - **Lei Ling**: [linglei@hnu.edu.cn](mailto:[linglei@hnu.edu.cn)   
  College of Biology, Hunan University, Changsha
