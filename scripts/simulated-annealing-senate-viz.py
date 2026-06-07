import pandas as pd
import os 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")

# read in all the csvs in throughput/simulated_annealing_senate_bills and concatenate


dfs = []
for file in os.listdir("throughput/simulated_annealing/senate_bills/metrics"):
    
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join("throughput/simulated_annealing/senate_bills/metrics", file))
        df["job_id"] = file.split(".")[0]
        dfs.append(df)
    
    
df = pd.concat(dfs, ignore_index=True)

fig, axarr = plt.subplots(1, 2, figsize=(12, 6))
sns.lineplot(data=df, x="step_num", y="likelihood", ax=axarr[0], hue="job_id", legend=False)
sns.scatterplot(data=df, x="step_num", y="likelihood", ax=axarr[0], hue="job_id", legend=False)
axarr[0].set(xlabel="Step Number", ylabel="Log-likelihood (only increases shown)")



sns.lineplot(data=df, x="step_num", y="ari", ax=axarr[1], hue="job_id", legend=False)
sns.scatterplot(data=df, x="step_num", y="ari", ax=axarr[1], hue="job_id", legend=False)
axarr[1].set(xlabel="Step Number", ylabel="ARI")

plt.tight_layout()
plt.savefig("fig/senate-bills-simulated-annealing.png")