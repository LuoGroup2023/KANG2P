import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 
data = pd.read_csv("/home/work/leiling/05_HumanG2P/ALS_Data/result/G+E_KANG2P/random_masking_accuracies.csv", header=0)

# 
plt.figure(figsize=(5, 5))

# 
sns.histplot(data["Accuracy"], bins=int((data["Accuracy"].max() - data["Accuracy"].min()) / 0.005), 
            color="lightblue", edgecolor="black", stat="density")

# 
sns.kdeplot(data["Accuracy"], color="blue", linewidth=1.5)

# 
plt.axvline(x=0.754, color="red", linestyle="--", linewidth=1)

# 
plt.xlim(0.62, 0.76)
plt.xticks(ticks=np.arange(0.62, 0.77, 0.03))
plt.ylim(0, 30)
plt.yticks(ticks=np.arange(0, 31, 5))

# 
plt.xlabel("Test accuracy (random genes)", fontsize=14, fontweight='bold')
plt.ylabel("Density", fontsize=14, fontweight='bold')
plt.title("", fontsize=14)

#
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')


plt.savefig('/home/work/leiling/05_HumanG2P/ALS_Data/result/G+E_KANG2P/Density_plot_spline.pdf', format='pdf', dpi=600, bbox_inches='tight')

# 
plt.tight_layout()
plt.show()