

library(progress)
library(optparse)
library(data.table)
library(plink2R)
library(glue)
library(glmnet)


option_list <- list(
  make_option(c("-p", "--plink_prefix"), type="character", default=NULL,
              help="PLINK genotype file prefix", metavar="character"),
  make_option(c("-c", "--chr"), type="numeric", default=12,
              help="chr", metavar="character"),
  make_option(c("-o", "--output_dir"), type="character", default='output_dir',
              help="Output file name", metavar="character")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

if (!dir.exists(opt$output_dir)){
  dir.create(opt$output_dir)
}


gff_data <- fread("/home/work/leiling/04_PlantG2P/00_DataSet/Rice4K/osa1_r7.all_models.gff3",
                  sep = "\t", header = FALSE, stringsAsFactors = FALSE)
gff_data <- gff_data[which(gff_data$V3 =='gene'),]

id <- sapply(gff_data$V9,function(x){
  tmp = unlist(strsplit(x,';'))[1]
  tmp <- gsub('ID=','',tmp)
})
names(id) <- NULL
gff_data$V9 <- id


chr = opt$chr

sub <- gff_data[which(gff_data$V1 ==paste0('Chr',chr)),]
sub <- data.frame(sub,stringsAsFactors = F)
bim <- fread(paste0(opt$plink_prefix,'.bim'),
             sep = "\t", header = FALSE, stringsAsFactors = FALSE)
fam <- fread(paste0(opt$plink_prefix,'.fam'),
             sep = " ", header = FALSE, stringsAsFactors = FALSE)

fam1 <- read.table('/home/work/leiling/04_PlantG2P/00_DataSet/Rice1495/1495Hybrid_MSUv7.fam',
                   header = F,stringsAsFactors = F)

GeneExp <- fread('/home/work/leiling/NetGP/HeadingDate/all_Rpkm_samplename_match.txt',
                 sep = "\t", header = TRUE, stringsAsFactors = FALSE,check.names = F)
GeneExp <- data.frame(GeneExp,stringsAsFactors = F)
rownames(GeneExp) <- GeneExp[,1]
GeneExp <- GeneExp[,-1] #55986 * 277
s <- colnames(GeneExp)

overlap <- intersect(fam$V1,s)

predict_s <- as.character(fam1$V1)
#277

GeneExp <- t(GeneExp)
overlapGeneExp <- GeneExp[overlap,]

SelectHighExpr <- function(overlapGeneExp,cutoff = 0.5){
  tmp = apply(overlapGeneExp, 2, function(idx){
    length(which(idx>=1))
  })
  index = which(tmp/nrow(overlapGeneExp) > cutoff)
  return(overlapGeneExp[,index])
  
}

HighExpr <- SelectHighExpr(overlapGeneExp)
HighExpr <- log2(HighExpr + 1)

gene = intersect(colnames(HighExpr),sub$V9)
HighExpr <- HighExpr[,gene]
sub <- sub[match(gene,sub$V9),]

pb <- progress_bar$new(total = nrow(sub))
TrainModel <- function(sub,bim,chr = 5){
  
  re <- NULL
  rho_df <- NULL
  for (i in 1:nrow(sub)){
    curGene <- sub[i,9]
    l <- sub[i,4]
    r <- sub[i,5]
    index <- which(bim$V4 > l-5e4 & bim$V4 < r+5e4)
    
    if (length(index) >1){
      snps <- bim[index,2]
     
      write.table(snps[,1],file.path(opt$output_dir,paste0(chr,'_topk_snp_list.txt')),
                  quote = F,row.names = F,col.names = F)
      
      A = file.path(opt$output_dir,paste0(chr,'_topk_snp_list.txt'))
      B = file.path(opt$output_dir,paste0(chr,'_temp_extracted'))
      cmd = glue("/home/leiling/software/plink --bfile {opt$plink_prefix} --extract {A} --make-bed --out {B}",
                 A=A,B = B)
      
      system(cmd,ignore.stdout = T,ignore.stderr = T)
      
      geno_path <- file.path(opt$output_dir,paste0(chr,'_temp_extracted'))
      # 
      extracted_plink_data <- read_plink(geno_path)
      snps <- extracted_plink_data$bed
      
      newid = sapply(rownames(snps),function(x){
        unlist(strsplit(x,':'))[1]
      })
      names(newid) <- NULL
      rownames(snps) <- newid
      
      X <- snps[overlap,]
      Y <- HighExpr[overlap,curGene]
      names(Y) <- rownames(X)
      
      PredictX <- snps[predict_s,]
 
      set.seed(123)
      trainS <- sample(rownames(X),round(0.8*nrow(X)))
      testS <- setdiff(rownames(X),trainS)
      
      X_train <- X[trainS,]
      X_test <- X[testS,]
      Y_train <- Y[trainS]
      Y_test <- Y[testS]
      

      cv_fit <- cv.glmnet(x = X_train, y = Y_train, alpha = 0.5)
      
      # 
      lambda_optimal <- cv_fit$lambda.min
      
      # 5. 
      final_fit <- glmnet(x = X_train, y = Y_train, alpha = 0.5, lambda = lambda_optimal)

      # 7. 
      new_y_pred <- predict(final_fit, newx = X_test)
      rho = cor(new_y_pred[,1],Y_test,method = 'spearman')
      
      rho_df <- rbind(rho_df,c(curGene,rho,ncol(X_train)))
      
      predict_exp <- predict(final_fit, newx = PredictX)
      colnames(predict_exp) <- curGene
      re <- cbind(re,predict_exp)
      C = file.path(opt$output_dir,paste0(chr,'_temp_extracted.*'))
      cmd2 = glue('rm {C}',C = C)
      system(cmd2)
      
    }
    
    pb$tick()
   
  }
  
  rho_df <- data.frame(rho_df,stringsAsFactors = F)
  colnames(rho_df) <- c('Gene','Spearman','Nsnps')
  return(list(re = re,rho_df = rho_df))
}

t1 <- Sys.time()

tt = TrainModel(sub = sub,bim,chr = chr)


outdf <- data.frame('ID' = rownames(tt$re),tt$re,stringsAsFactors = F)
fwrite(outdf, file.path(opt$output_dir,'Predict_Expre.tsv'), 
       sep = "\t", col.names = T, row.names = F)

fwrite(tt$rho_df, file.path(opt$output_dir,'Predict_Spearman_acc.tsv'), 
       sep = "\t", col.names = T, row.names = F)
t2 <- Sys.time()
print(t2 - t1)