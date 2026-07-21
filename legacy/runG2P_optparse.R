
library(optparse)
suppressPackageStartupMessages(library(G2P))
suppressPackageStartupMessages(library(data.table))

option_list <- list(
  
  make_option(c("-x", "--X_file"), type = "character", default = NULL,
              help = "X file", metavar = "character"),
  make_option(c("-y", "--Y_file"), type = "character", default = NULL,
              help = "Y file", metavar = "character"),
  make_option(c("-t", "--train_ids_file"), type = "character", default = NULL,
              help = "Train IDs file", metavar = "character"),
  make_option(c("-e", "--test_ids_file"), type = "character", default = NULL,
              help = "Test IDs file", metavar = "character"),
  
  make_option(c("-o", "--output_dir"), type = "character", default = NULL,
              help = "Output directory", metavar = "character")
 
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

if (!dir.exists(opt$output_dir)){
  dir.create(opt$output_dir,recursive = T)
}

X <- fread(opt$X_file)
X <- data.frame(X,stringsAsFactors = F,check.names = F)


Y <- read.delim(opt$Y_file, header = T, check.names = F, stringsAsFactors = F, as.is = T)
Y$ID <- as.character(Y$ID)
trait <- colnames(Y)[2]
print(trait)
train_s <- read.delim(opt$train_ids_file, header = F, stringsAsFactors = F, as.is = T)
test_s <- read.delim(opt$test_ids_file, header = F, stringsAsFactors = F, as.is = T)
rownames(X) <- X$ID
X <- X[,-1]
train_s$V1 <- as.character(train_s$V1)
test_s$V1 <- as.character(test_s$V1)
X_train <- X[train_s$V1,]
X_test <- X[test_s$V1,]
Y_train <- Y[match(train_s$V1,Y$ID),2]
Y_test <- Y[match(test_s$V1,Y$ID),2]

trainIdx <- match(train_s$V1,Y$ID)
testIdx <- match(test_s$V1,Y$ID)

X <- as.matrix(X)


# 对训练矩阵质控，过滤掉全为常值的 -----------------------------------------------------------------
sds = apply(X_train,2,sd)
rm_index <- which(sds ==0)
if (length(rm_index)>=1){
  rm_snp = names(rm_index)
  X <- X[,-match(rm_snp,colnames(X))]
}


t1 <- Sys.time()
C2Pres <- G2P(markers = X,
              data = Y,
              trait = trait,
              modelMethods = c("BayesC", "BL", "BRR", 
                               "RKHS", "RRBLUP","LASSO", "SPLS", "SVR", "RFR"
                               ),trainIdx = trainIdx,
              predIdx = testIdx)
t2 <- Sys.time()


evalres <- G2PEvaluation(realScores = C2Pres[,1], predScores = C2Pres[,2:ncol(C2Pres)], 
                         evalMethod = c("pearson", "meanNDCG",
                                        "MSE","R2"),topAlpha = 1:90, probIndex = 5)

save(C2Pres,evalres,file = file.path(opt$output_dir,'G2P.Rdata'))
plot_mat <- evalres$corMethods
print(plot_mat)