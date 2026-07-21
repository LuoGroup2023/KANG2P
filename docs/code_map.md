# Source-to-release migration map

The release candidate was assembled without modifying the audited source
trees. This table records provenance.

| Release workflow | Audited source | Release change |
|---|---|---|
| Genotype QC | `ALS_raw_data/DiseaseCapsule/data_preprocessing/run_paper_qc.py` | portable defaults and CLI launcher |
| Nested GWAS/Gene-PCA | `ALS_raw_data/DiseaseCapsule/data_preprocessing/run_nested_gwas_gene_pca.py` | fixed rsID recovery and training-only missing-value imputation |
| Human predicted expression | public `Impute_gene_expression/run_PrediXcan.sh` and `MergeAndWilcox.R` | parameterized MetaXcan runner, chromosome merge, fold-specific selection |
| Predicted proteome | existing PP matrix consumers in `Disease/{ML,Capsule,KAN}` | added alignment/QC/standardization utility; upstream scorer still missing |
| Crop predicted expression | `Gene_Exp_Prediction/{Rice18K,Maize1404}` R scripts and launchers | consolidated crop-agnostic R model, final refit, parallel and merge utilities |
| Comparison / DiseaseCapsule | duplicated `Disease/Capsule/{4.23,AD5.9}/.../capsnet_nested_cv.py` | consolidated G/PE/PP loader and fold-safe inner preprocessing |
| Comparison / traditional disease ML | duplicated `Disease/ML/{4.23,AD5.9}/...` scripts | one multimodal CLI with preprocessing inside `GridSearchCV` |
| Comparison / traditional plant GS | `Plant/Script/runG2P_nestedCV.R` and `runTreeGS_NestedCV.py` | renamed and grouped under the comparison workflow |
| DualKAN gated fusion | `DualKAN/DualOmicsModel01.py`, `KAN.py`, launcher, aggregator | portable names, paths, and Fourier/spline head selection |
| Human gradients | `Disease/KAN/G2P/als_g*_interpretability.py` | grouped dependencies and portable defaults |

The former public GitHub layout is preserved under `legacy/`.
