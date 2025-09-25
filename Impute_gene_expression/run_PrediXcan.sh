

#!/bin/bash

# brain list
brain_regions=(
    "Brain_Substantia_nigra"
    "Brain_Spinal_cord_cervical_c-1"
    "Brain_Putamen_basal_ganglia"
    "Brain_Nucleus_accumbens_basal_ganglia"
    "Brain_Hypothalamus"
    "Brain_Hippocampus"
    "Brain_Frontal_Cortex_BA9"
    "Brain_Cortex"
    "Brain_Cerebellum"
    "Brain_Cerebellar_Hemisphere"
    "Brain_Caudate_basal_ganglia"
    "Brain_Anterior_cingulate_cortex_BA24"
    "Brain_Amygdala"
)

model_db_path = '/home/leiling/software/PrediXcan/elastic_net_models/'
# 
for brain_region in "${brain_regions[@]}"; do
    # 
    mkdir -p "$brain_region"
    
    # 
    for i in {1..22}; do
        # call Predict.py
        python3 ~/software/PrediXcan/MetaXcan-master/software/Predict.py \
            --model_db_path /home/leiling/software/PrediXcan/elastic_net_models/en_${brain_region}.db \
            --model_db_snp_key varID \
            --vcf_genotypes "/home/work/leiling/05_HumanG2P/ALS/GWAS2019_NL/02_Filter_batch_Data/0_chr_split/chr${i}.phased.vcf.gz" \
            --vcf_mode genotyped \
            --liftover hg19ToHg38.over.chain.gz \
            --on_the_fly_mapping METADATA "chr{}_{}_{}_{}_b38" \
            --prediction_output "${brain_region}/en_chr${i}_prediction.txt" \
            --prediction_summary_output "${brain_region}/en_chr${i}_summary.txt" \
            --verbosity 9 \
            --throw
        
        
        echo "Processed ${brain_region} - Chromosome ${i} ------------------"
    done
done

