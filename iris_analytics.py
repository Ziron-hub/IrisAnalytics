#!/usr/bin/env python
# coding: utf-8

# # a) Tabela descritiva para cada espécie

# In[6]:


import pandas as pd

df = pd.read_csv('iris3.csv')


# In[7]:


df[["Sepal.Length", "Sepal.Width","Petal.Length","Petal.Width"]] = df[["Sepal.Length", "Sepal.Width","Petal.Length","Petal.Width"]].apply(pd.to_numeric)


# In[8]:


df.groupby('Species')['Sepal.Width',].describe()


# In[9]:


df.groupby('Species')['Sepal.Length',].describe()


# In[10]:


df.groupby('Species')['Petal.Width',].describe()


# In[11]:


df.groupby('Species')['Petal.Length',].describe()


# # b) Gráficos para cada uma das medições realizadas para a média

# In[12]:


mediaPetalLength = df.groupby('Species')['Petal.Length',].mean()
mediaPetalWidth = df.groupby('Species')['Petal.Width',].mean()
mediaSepalWidth = df.groupby('Species')['Sepal.Width',].mean()
mediaSepalLength = df.groupby('Species')['Sepal.Length',].mean()


# In[13]:


import matplotlib.pyplot as plt

x = df['Species'].unique()
y = mediaPetalLength['Petal.Length']

plt.bar(x,y)
plt.grid(True)
plt.ylabel('Média do comprimento da Pétala')
plt.xlabel("Plantas")


# In[14]:


x = df['Species'].unique()
y = mediaPetalWidth['Petal.Width']

plt.bar(x,y)
plt.grid(True)
plt.ylabel('Média da largura da Pétala ')
plt.xlabel("Plantas")


# In[15]:


x = df['Species'].unique()
y = mediaSepalWidth['Sepal.Width']

plt.bar(x,y)
plt.grid(True)
plt.ylabel('Média da largura da Sépala ')
plt.xlabel("Plantas")


# In[16]:


x = df['Species'].unique()
y = mediaSepalLength['Sepal.Length']

plt.bar(x,y)
plt.grid(True)
plt.ylabel('Média do comprimento da Sépala ')
plt.xlabel("Plantas")


# # c) Gráfico para avaliação de associação entre as variáveios comprimento e largura
# 

# In[17]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:,2 :4]  # we only take the last two features.
y = iris.target

splits = np.array_split(X, 3)

setosa = splits[0]
versicolor = splits[1]
virginica = splits[2]

plt.figure(2, figsize=(8, 6)) #aumentar o tamanho do gráfico
plt.clf()

# Plot the training points

plt.scatter(setosa[:, 0], setosa[:, 1], cmap=plt.cm.Set1, edgecolor="black", color="red")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

plt.show()

#calculando a correlação de pearson

from scipy.stats import pearsonr

print("A relação entre as variáveis largura e comprimento da setosa usando a correlação de Pearson é: ", pearsonr(setosa[:, 0], setosa[:, 1])[0])


# In[18]:


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:,2 :4]  # we only take the last two features.
y = iris.target

splits = np.array_split(X, 3)

setosa = splits[0]
versicolor = splits[1]
virginica = splits[2]

plt.figure(2, figsize=(8, 6)) #aumentar o tamanho do gráfico
plt.clf()

# Plot the training points

plt.scatter(versicolor[:, 0], versicolor[:, 1], cmap=plt.cm.Set1, edgecolor="black", color="orange")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

plt.show()

#calculando a correlação de pearson

from scipy.stats import pearsonr

print("A relação entre as variáveis largura e comprimento da versicolor usando a correlação de Pearson é: ", pearsonr(versicolor[:, 0], versicolor[:, 1])[0])


# In[19]:


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:,2 :4]  # we only take the last two features.
y = iris.target

splits = np.array_split(X, 3)

setosa = splits[0]
versicolor = splits[1]
virginica = splits[2]

plt.figure(2, figsize=(8, 6)) #aumentar o tamanho do gráfico
plt.clf()

# Plot the training points

plt.scatter(virginica[:, 0], virginica[:, 1], cmap=plt.cm.Set1, edgecolor="black", color="grey")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

plt.show()

#calculando a correlação de pearson

from scipy.stats import pearsonr

print("A relação entre as variáveis largura e comprimento da virginica usando a correlação de Pearson é: ", pearsonr(virginica[:, 0], virginica[:, 1])[0])


# In[21]:


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:,2 :4]  # we only take the last two features.
y = iris.target

splits = np.array_split(X, 3)

setosa = splits[0]
versicolor = splits[1]
virginica = splits[2]

plt.figure(2, figsize=(8, 6)) #aumentar o tamanho do gráfico
plt.clf()

# Plot the training points

colors = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor="k")
plt.xlabel("Petal length")
plt.ylabel("Petal width")

classes = ['Setosa', 'Versicolor', 'Sirginica']

plt.legend(handles=colors.legend_elements()[0], labels=classes)

plt.show()


#calculando a correlação de pearson

from scipy.stats import pearsonr

print("A relação entre as variáveis largura e comprimento de todas usando a correlação de Pearson é: ", pearsonr(X[:, 0], X[:, 1])[0])


# In[ ]:




