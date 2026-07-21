#setwd('/home/work/leiling/05_HumanG2P/ALS_Data/')
library(data.table)

g <- read.delim("z1_contribution_kan.txt",
                header = F)
g <- as.vector(g$V1)
g <- g*1e6
print(g)

g <- matrix(g,nrow = length(g),ncol = 1)

X  <- fread("omics1_test_output.txt",
            header = T)
X <- as.matrix(X)

Y <- fread("z1_latent_vectors.txt",
           header = T)
Y <- as.matrix(Y)
cor_mat <- cor(X,Y)
cor_mat <- abs(cor_mat)

re <- cor_mat %*% g

rownames(re) <- rownames(cor_mat)
re <- as.vector(re)
names(re) <- rownames(cor_mat)

re <- sort(re,decreasing = T)
print(re[1:10])
write.table(re,'Feature_rank_Spline.txt',quote = F,col.names = F)



pcc <- read.table('Feature_rank_Spline.txt',header = F)
pcc$gene <- sapply(pcc$V1,function(idx){
  unlist(strsplit(idx,':'))[2]
})


# 
file_path <- "DiseaseCaps_922_genes.txt"  
# 
lines <- readLines(con = file_path)

# 
gene_list <- unlist(strsplit(lines, split = " "))


print(gene_list)

Spline_list <- read.delim('top913_genes_list_based_on_Spline.txt',
                          header = F)
Spline_list <- Spline_list$V1

library(openxlsx)
ref_map <- read.xlsx('1-s2.0-S0896627321010369-mmc3.xlsx',
                     sheet = 1)
ref_map_list <- ref_map$gene

curated <- read.xlsx('1-s2.0-S0896627321010369-mmc3.xlsx',
                     sheet = 2)

x <- list('DiseaseCapsule' = gene_list,
          'KANG2P' = Spline_list,
          'RefMap' = ref_map_list,
          'Curated' = curated$Gene.Symbol)

# x <- list('DiseaseCapsule' = gene_list,
#           'Gradient' = Gradient_list,
#           'Spline' = Spline_list)

library(VennDiagram
        )

display_venn <- function(x, ...){
  library(VennDiagram)
  grid.newpage()
  venn_object <- venn.diagram(x, filename = NULL, ...)
  grid.draw(venn_object)
}
p = display_venn(
  x,
  
  fill = c("#0073C2FF", "#EFC000FF", "#CD534CFF",'#F7B7D2')
)


marker <- read.delim('/home/leiling/software/Translatomer/ALS/ALS_phasing_marker.bed',
                     header = F)

gene_pcc <- pcc[!duplicated(pcc$gene),]


intersect(gene_pcc$gene[1:913],marker$V5)

tmp = gene_pcc[match(marker$V5,gene_pcc$gene),]
gene_pcc$rank <- 1:nrow(gene_pcc)
gene_pcc$to_label <-  FALSE
index = match(c("THBS2","BCL2","HNRNPA2B1", "ELP3","ARHGEF28", "ITPR2"),gene_pcc$gene)
gene_pcc[index,'to_label'] = TRUE

library(ggrepel)

p = ggplot(gene_pcc, aes(x = rank, y = V2)) +
  # 
  geom_point(alpha = 0.6, size = 2, color = "gray40") +
  # 
  geom_point(data = subset(gene_pcc, to_label),
             aes(color = gene), size = 3) +
  # 
  geom_text_repel(data = subset(gene_pcc, to_label),
                  aes(label = gene, color = gene),
                  size = 4, fontface = "bold",
                  box.padding = unit(1, "lines"),
                  point.padding = unit(0.5, "lines"),
                  segment.size = 1,
  ) +
  # 
  scale_color_manual(values = c("HNRNPA2B1" = "#E41A1C", "ELP3" = "#E41A1C",
                                "THBS2" = "#377EB8","BCL2" = "#E41A1C",
                                "ARHGEF28" = "#E41A1C", 
                                "ITPR2" = "#E41A1C")) +#   # 添加标题和轴标签
  labs(title = "",
       x = "Rank",
       y = "Feature Importance",
  ) +
  # 
  theme_classic() +
  geom_vline(xintercept = 913, linetype = "dashed", color = "gray50", size = 1) +
  theme(
    
    axis.title = element_text(size = 14, color = "black", face = "bold"),
    # 
    axis.text = element_text(size = 14, color = "black"),
    legend.position = 'none'
    
  )
ggsave(filename = 'Feature_importance_scatter_plot_spline.pdf',plot = p,
       height = 5,width = 5,dpi = 600)

write.table(gene_pcc$gene[1:913],
            'top913_genes_list_based_on_Spline.txt',
            quote = F,row.names = F,col.names = F)
