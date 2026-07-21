
suppressPackageStartupMessages(library(ggplot2))

library(optparse)

suppressPackageStartupMessages(library(data.table))

option_list <- list(
  
  make_option(c("-x", "--InputDir"), type = "character", default = NULL,
              help = "InputDir", metavar = "character")
  

)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

InputDir = opt$InputDir
re <- vector('list',length = 22)
for (i in 1:22){
  cur_file <- file.path(InputDir,paste0('en_chr',i,'_prediction.txt'))
  tmp <- fread(cur_file,header = T)
  tmp <- data.frame(tmp,stringsAsFactors = F)
  print(ncol(tmp)-2)
  re[[i]] <- tmp[,3:ncol(tmp)]
}
re <- do.call(cbind,re)
label <- read.csv('/home/work/xluo/prj/g2p/disease/als/labels.csv',
                  header = T)
rownames(re) <- label$FID

test_idx <- read.table('/home/work/leiling/05_HumanG2P/ALS_Data/test.idx',header = F)
test_idx <- test_idx$V1 +1
test_sample <- label[test_idx,]
table(test_sample$Pheno)
train_sample <- setdiff(label$FID,test_sample$FID)


train_a <- re[train_sample,]
train_label <- label[match(train_sample,label$FID),]


CalWilcoxP <- function(train_a,train_label){
  re <- NULL
  for (i in 1:ncol(train_a)){
    id <- colnames(train_a)[i]
    curx <- train_a[which(train_label$Pheno ==1),i]
    cury <- train_a[which(train_label$Pheno ==2),i]
    p = wilcox.test(curx,cury)$p.value
    re <- rbind(re,c(i,p,id))
  }
  return(re)
}

Pvalue <- CalWilcoxP(train_a = train_a,train_label = train_label)
Pvalue <- data.frame(Pvalue,stringsAsFactors = F,check.names = F)
colnames(Pvalue) <- c('RIdx','Pvalue','ID')
Pvalue[,2] <- as.numeric(Pvalue[,2])

Pvalue$FDR <- p.adjust(Pvalue$Pvalue,method = 'fdr')
str(Pvalue)

sig_mat <- re[,which(Pvalue$Pvalue < 0.05)]
fwrite(sig_mat,file.path(InputDir,'Exp.txt'),
       quote = F,row.names = F,sep = '\t')

CountDF <- data.frame('Type' = c('ALL_gene','Sig_Diff_gene'),
                      'Number' = c(ncol(re),ncol(sig_mat)),
                      stringsAsFactors = F)
print(CountDF)
write.table(CountDF,file.path(InputDir,'Sig_Diff_gene_count.txt'),
            quote = F,row.names = F,
            sep = '\t')
write.table(Pvalue,file.path(InputDir,'wilcox_test_result.txt'),quote = F,row.names = F,
            sep = '\t')

df <- Pvalue
df$log10_p <- -log10(df$Pvalue)

# 
df$is_significant <- df$Pvalue < 0.05
num_significant <- sum(df$is_significant,na.rm = T)

# 
p = ggplot(df, aes(x = log10_p, fill = is_significant)) +
  geom_histogram(bins = 30, color = "black") +
  scale_fill_manual(values = c("skyblue", "red"), 
                    labels = c("Non - significant", "Significant")) +
  labs(title = "Histogram of -log10(Pvalue)",
       x = "-log10(Pvalue)",
       y = "Frequency") +
  guides(fill = guide_legend(title = "Significance")) +
  annotate("text", 
           x = max(df$log10_p,na.rm = T), 
           y = max(ggplot_build(ggplot(df, aes(x = log10_p)) + 
                                  geom_histogram(bins = 30))$data[[1]]$count), 
           label = paste("Significant: ", num_significant), 
           hjust = 1, vjust = 1, color = "red") +
  geom_vline(xintercept = -log10(0.05), linetype = "dashed", color = "black")
ggsave(file.path(InputDir,'pvalue_distribution.png'),
       plot = p,height = 5,width = 5,dpi = 600)
