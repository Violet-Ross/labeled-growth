import pandas as pd
import os 
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# read in all the csvs in throughput/simulated_annealing_senate_bills and concatenate


dfs = []
for file in os.listdir("throughput/simulated_annealing_senate_bills"):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join("throughput/simulated_annealing_senate_bills", file))
        df["job_id"] = file.split(".csv")[0]
        dfs.append(df)
df = pd.concat(dfs, ignore_index=True)

fig, axarr = plt.subplots(1, 2, figsize=(12, 6))
sns.lineplot(data=df, x="step_num", y="likelihood", ax=axarr[0], hue="job_id", legend=False)
sns.lineplot(data=df, x="step_num", y="ari", ax=axarr[1], hue="job_id", legend=False)
axarr[0].set_title("Likelihood over time (only increases shown)")
axarr[1].set_title("ARI over time")
plt.tight_layout()
plt.savefig("fig/senate-bills-simulated-annealing.png")