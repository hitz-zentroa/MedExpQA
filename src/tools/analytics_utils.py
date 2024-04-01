from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def build_linear_bins(min_val:int, max_val:int, num_bins:int):

	return [((max_val-min_val)/num_bins)*(i+1) for i in range(num_bins)]

def plot_size_dist(token_lengths:List, title:str=None, num_bins:int=50, bins:List = None):
	# sns.displot(size_df.loc[size_df['total'] < 1e6], x="total", bins=20)
	if bins is None:
		bins = build_linear_bins(min(token_lengths), max(token_lengths), num_bins)
	#sizes = [random.uniform(0, 100) for i in range(100)]
	fg = sns.displot(token_lengths, stat='percent', bins=bins)
	#plt.xscale('log')
	for ax in fg.axes.ravel():

		# add annotations
		for c in ax.containers:
			# custom label calculates percent and add an empty string so 0 value bars don't have a number
			labels = [rf'{i+1}: {w:0.1f}%' if (w := v.get_height()) > 0 else '' for i,v in enumerate(c)]

			ax.bar_label(c, labels=labels, label_type='edge', fontsize=8, rotation=90, padding=2)

		ax.margins(y=0.2)
	if title is not None:
		plt.title(title)
	plt.show()

def multiple_dist(data:dict, num_bins:int=50, bins:List = None):
	df = pd.DataFrame.from_dict(data)
	sns.displot(data=df, x="docs", stat='percent', hue="length") # "layer", "stack", "fill"
	plt.show()