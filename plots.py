import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# plt.style.use(['seaborn-white', 'seaborn-paper'])
matplotlib.rc("font", family="monospace")
# import seaborn as sns

# column_names = ["n", "σ", "λ/μ", "1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"]
column_names = ["n", "σ", "λ/μ", "1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"]

df = pd.read_excel("../vanilla_sigma_10_seeds.xlsx")
problem = "Ackley function"

fig = plt.figure(figsize=(10, 5))

grouped = pd.melt(df, id_vars=["n"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["n"] == 1]
plt.subplot(2,3,1)
plt.title('n = 1')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["n"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["n"] == 5]
plt.subplot(2,3,2)
plt.title('n = 5')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["n"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["n"] == 10]
plt.subplot(2,3,3)
plt.title('n = 10')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["n"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["n"] == 50]
plt.subplot(2,3,4)
plt.title('n = 50')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")


grouped = pd.melt(df, id_vars=["n"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["n"] == 100]
plt.subplot(2,3,5)
plt.title('n = 100')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of n size on ES for the " + problem)
plt.show()



fig = plt.figure(figsize=(10, 5))

grouped = pd.melt(df, id_vars=["σ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["σ"] == 0.1]
print(grouped)
plt.subplot(2,2,1)
plt.title('σ = 0.1')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["σ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["σ"] == 1]
plt.subplot(2,2,2)
plt.title('σ = 1')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["σ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["σ"] == 10]
plt.subplot(2,2,3)
plt.title('σ = 10')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["σ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["σ"] == 100]
plt.subplot(2,2,4)
plt.title('σ = 100')
# plt.yscale('log')
boxplot = grouped.boxplot(column="value")


plt.suptitle("Impact of σ on ES for the " + problem)
plt.show()


fig = plt.figure(figsize=(10, 5))

grouped = pd.melt(df, id_vars=["λ/μ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["λ/μ"] == 1]
plt.subplot(2,3,1)
plt.title('λ/μ = 1')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["λ/μ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["λ/μ"] == 2]
plt.subplot(2,3,2)
plt.title('λ/μ = 2')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["λ/μ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["λ/μ"] == 4]
plt.subplot(2,3,3)
plt.title('λ/μ = 4')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["λ/μ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["λ/μ"] == 6]
plt.subplot(2,3,4)
plt.title('λ/μ = 6')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["λ/μ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["λ/μ"] == 12]
plt.subplot(2,3,5)
plt.title('λ/μ = 12')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

grouped = pd.melt(df, id_vars=["λ/μ"],value_vars=["1821", "97", "1940", "1924", "1250", "776", "600", "430", "445", "336"])
grouped = grouped[grouped["λ/μ"] == 20]
plt.subplot(2,3,6)
plt.title('λ/μ = 20')
plt.yscale('log')
boxplot = grouped.boxplot(column="value")

plt.suptitle("Impact of λ/μ ratio on ES for the " + problem)
plt.show()

