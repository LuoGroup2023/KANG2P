#

re = lapply(1:12,function(x){
  cur_path = file.path(paste0('chr',x,'_impute'),'Predict_Spearman_acc.tsv')
  tmp = read.delim(cur_path,header = T,check.names = F,stringsAsFactors = F)
  
  tmp$Chr <- x
  
  return(tmp)
})
re <- do.call(rbind,re)
re$Chr <- paste0('chr',re$Chr)

re$Chr <- factor(re$Chr,levels = unique(re$Chr))
library(dplyr)
median_data <- re %>%
  group_by(Chr) %>%
  summarize(median_Nsnps = round(median(Nsnps)))


re_with_median <- left_join(re, median_data, by = "Chr")


p <- ggplot(re_with_median, aes(x = Nsnps)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap( ~ Chr, nrow = 2) +
  theme_classic() + 
  theme(axis.text.x = element_text(face = "bold", size = 14, color = 'black',
                                   angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(face = "bold", size = 14, color = 'black'),
        axis.title  = element_text(face = "bold", size = 16, color = 'black'),
        strip.text = element_text(face = "bold", size = 16, color = 'black')) +
  labs(x = "Number of cis SNPs", y = "Count") +
  geom_vline(data = re_with_median,
             aes(xintercept = median_Nsnps), color = "red", linetype = "dashed", size = 1.2) +
  geom_text(aes(x = Inf, y = Inf, label = paste("Median:", median_Nsnps)), 
            hjust = 1.1, vjust = 1.1, size = 4, color = "black")
ggsave('Hist_cis_SNPs.png',plot = p,height = 8,width = 12,dpi = 600)
ggsave('Hist_cis_SNPs.pdf',plot = p,height = 8,width = 12,dpi = 600)


#  ----------------------------------------------------------------


quantile_data <- re %>%
  group_by(Chr) %>%
  summarize(quantile_70_Spearman = quantile(Spearman, 0.9,na.rm = T))

# 
df_with_quantile <- left_join(re, quantile_data, by = "Chr")

# 
p <- ggplot(df_with_quantile, aes(x = Spearman)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  facet_wrap( ~ Chr, nrow = 2) +
  theme_classic() + 
  theme(axis.text.x = element_text(face = "bold", size = 14, color = 'black',
                                   angle = 90, hjust = 1, vjust = 0.5),
        axis.text.y = element_text(face = "bold", size = 14, color = 'black'),
        axis.title  = element_text(face = "bold", size = 16, color = 'black'),
        strip.text = element_text(face = "bold", size = 16, color = 'black'))  +
  labs(x = "Spearman Correlation", y = "Count") +
  geom_vline(aes(xintercept = quantile_70_Spearman), color = "red", linetype = "dashed", size = 1.1) +
  geom_text(aes(x = Inf, y = Inf, 
                label = sprintf("90%% Quantile: %.2f", quantile_70_Spearman)), hjust = 1.1, vjust = 1.1, size = 4, color = "black")


print(p)
ggsave('Hist_Spearman_Acc.png',plot = p,height = 8,width = 12,dpi = 600)


# output Top10% gene ----------------------------------------------------------

re1 = lapply(1:12,function(x){
  cur_path = file.path(paste0('chr',x,'_impute'),'Predict_Spearman_acc.tsv')
  tmp = read.delim(cur_path,header = T,check.names = F,stringsAsFactors = F)
 
  tmp$Chr <- x
 
  cutoff <- quantile(tmp$Spearman,0.9,na.rm = T)
  index <- which(tmp$Spearman > cutoff)
  return(tmp[index,])
})
re1 <- do.call(rbind,re1)

Expr = lapply(1:12,function(x){
  cur_path = file.path(paste0('chr',x,'_impute'),'Predict_Expre.tsv')
  tmp = read.delim(cur_path,header = T,check.names = F,stringsAsFactors = F)
  
  return(tmp)
})

id <- Expr[[1]]$ID

re <- lapply(1:12, function(idx){
  Expr[[idx]][,-1]
})
re <- do.call(cbind,re)
re <- re[,re1$Gene]
re <- data.frame('ID' = id,re,stringsAsFactors = F,check.names = F)
fwrite(re,'High_predict_acc_Exp.txt',quote = F,sep = '\t',row.names = F)


