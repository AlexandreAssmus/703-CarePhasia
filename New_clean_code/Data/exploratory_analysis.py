import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import mannwhitneyu, shapiro
import numpy as np

data = pd.read_csv(r'New_clean_code\Data\thresholds_per_file.csv')

## boxplot ##
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.boxplot(data=data, x='diagnosis', y='average_tree_depth')
plt.subplot(2,2,2)
sns.boxplot(data=data, x='diagnosis', y='average_lexical_density')
plt.subplot(2,2,3)
sns.boxplot(data=data,x='diagnosis', y='word_stutter_count')
plt.subplot(2,2,4)
sns.boxplot(data=data, x='diagnosis', y='syllable_stutter_ratio')
plt.tight_layout()
plt.savefig('boxplot_per_metric.jpg')


