# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:48:48 2020

"""


# Imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr
from statsmodels.stats.diagnostic import normal_ad
from scipy.optimize import minimize
import emcee
import mplcursors
import math
import seaborn as sns
sns.set(style="ticks")


# Reducing and cleaning up the data
data = pd.read_csv('co-emissions-per-capita-vs-gdp-per-capita-international-.csv',
                    usecols=[0,1,2,3,4,5],
                    header=0,
                    names=['Country', 'Code' , 'Year', 'CO2_emissions', 'GDP', 'Population'],
                    na_values='--')
cont_name = pd.read_csv('country-and-continent-codes-list.csv',
                    usecols=[0,1,5],
                    header=0,
                    names=['Continent', 'Color_code','Code'])
cont_name = cont_name.drop_duplicates('Code')

data_2015 = data.loc[data['Year'] == '2015']
data_2015 = data_2015.dropna()
data_2015 = data_2015.reset_index()
del data_2015['index']
data_2015 = data_2015.merge(cont_name, on='Code')
data_2015_simple = data_2015[['Country','Continent','Color_code','CO2_emissions','GDP','Population']]
N = data_2015_simple.shape[0]

data_2015_simple['log_pop'] = np.log10(data_2015_simple['Population'])
data_2015_simple['log_gdp'] = np.log10(data_2015_simple['GDP'])
data_2015_simple['log_co2'] = np.log10(data_2015_simple['CO2_emissions'])
data_2015_simple = data_2015_simple.drop(columns = ['Color_code','Population', 'GDP','CO2_emissions'])
data_2015_simple.to_csv('./CO2_dataframe.csv',index=False)
co2 = pd.read_csv("CO2_dataframe.csv")
size = np.sqrt(np.sqrt(data_2015_simple['log_pop']))

#pd.plotting.scatter_matrix(co2,diagonal='kde',alpha=0.5)

g = sns.pairplot(co2, hue="Continent", kind="reg",palette="Set2", diag_kind="kde",height=2.5)

j = sns.lmplot(x='log_gdp',y='log_co2', hue="Continent", data=co2);
# ax.annotate('hi',(3,2))
# plt.show()

# Preliminary scatter plot of data - Dif colors for each point, size of point corresponds to population

# js - Well done visualization!
# np.random.seed(521)
# #colors = np.random.rand(N)
# x = data_2015_simple.GDP
# y = round(data_2015_simple.CO2_emissions,2)
# size = np.sqrt(np.sqrt(data_2015_simple['Population']))
# names = data_2015_simple.Country
# cont = data_2015_simple.Continent
# colors = data_2015_simple.Color_code

# fig, ax = plt.subplots()
# plt.figure(1)
# plt.xlabel('GDP per Capita')
# plt.ylabel('CO2 Emissions per Capita (tons)')
# plt.title('CO2 Emissions Based on 2015 GDP')
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(300,400000)
# plt.ylim(.01,100)
#sc = plt.scatter(x,y,s=size,c=colors,cmap='viridis',alpha=0.5)

#%%
handles, labels = sc.legend_elements(prop='colors')
labels = ['Africa','Asia','Europe','North America','Oceania','South America']
leg1 = plt.legend(handles,labels,title='Continent',loc="lower right",shadow=True)
for area in [31,56,99.9,177.8]:  #make dummy data to specify legend entries
    plt.scatter([],[],c="k",s=area, label=str(math.ceil(area**4/1000000)))
plt.legend(scatterpoints=1,frameon=True,labelspacing=1,title='Population\n(millions)',shadow=True)
ax.add_artist(leg1) #add color legend back on into figure

# annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
#                     bbox=dict(boxstyle="round", fc="w"),
#                     arrowprops=dict(arrowstyle="->"))
# annot.set_visible(False)

annot1 = mplcursors.cursor(hover=True)
@annot1.connect("add")
def _(sel):
    sel.annotation.set(text=f'{names[sel.target.index]}\n{y[sel.target.index]}:{int(x[sel.target.index])}')
    sel.annotation.get_bbox_patch().set(fc="white")
    sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=.5)

plt.show()


# js - This is so cool, btw!
# def update_annot(ind):
#     pos = sc.get_offsets()[ind["ind"][0]]
#     annot.xy = pos
#     text = "{}, {}".format("".join(str([y[n] for n in ind["ind"]])),
#                           " ".join([names[n] for n in ind["ind"]]))
#     annot.set_text(text)
#     annot.get_bbox_patch().set_facecolor('g')
#     annot.get_bbox_patch().set_alpha(0.4)

# def hover(event):
#     vis = annot.get_visible()
#     if event.inaxes == ax:
#         cont, ind = sc.contains(event)
#         if cont:
#             update_annot(ind)
#             annot.set_visible(True)
#             fig.canvas.draw_idle()
#         else:
#             if vis:
#                 annot.set_visible(False)
#                 fig.canvas.draw_idle()

# fig.canvas.mpl_connect("motion_notify_event", hover)

plt.savefig('Scatter_CO2_Emissions_vs_GDP')

# Preliminary fitting a line 
fit = np.polyfit(x, y, 1, full=False,cov=True )
fitparams = fit[0]
slope = fitparams[0]
intercept = fitparams[1]


# js - the errors would be the square root of the covariance elements!
cov = fit[1]
param_error = np.sqrt(np.diagonal(cov))
slope_error = param_error[0]
intercept_error = param_error[1]

xfit = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit = intercept + slope*xfit
pf = plt.plot(xfit,yfit,'r--',label='Polyfit line') # Looks pretty good ngl
annot2 = mplcursors.cursor(pf)
@annot2.connect("add")
def _(sel):
    sel.annotation.set(text='Polyfit line',position=(2100,30))
    sel.annotation.get_bbox_patch().set(fc="0.9",ec="gray")
    sel.annotation.arrow_patch.set(arrowstyle="->", fc="white", alpha=.5)
plt.savefig('Preliminary_Line_Fit')
#%%
# Basix data values
std = np.std(data_2015_simple.CO2_emissions)
mean = np.mean(data_2015_simple.CO2_emissions)
print(std, mean)

# Seeing how closely related the data is to a Gaussian
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

# Histogram of CO2 Emissions
plt.figure(2)
hist = plt.hist(data_2015_simple.CO2_emissions,bins=50,density=True)

x_gauss,y_gauss = gauss(sig=std,x0=mean)
plt.figure(3)
plt.plot(x_gauss,y_gauss,'r--')
#plt.savefig('Attempted_Gaussian_Fit')

ad, p = normal_ad(data_2015_simple.CO2_emissions) # A-D test
print(p) # Basically 0

# KDE Estimation
kde = gaussian_kde(data_2015_simple.CO2_emissions)
xvals = np.linspace(0, 50, 10000)

plt.figure(4)
plt.plot(xvals,kde.pdf(xvals),'r--')
plt.xlim(0,50)
plt.hist(data_2015_simple.CO2_emissions,bins=np.arange(50),density=True)
plt.ylabel('Frequency',fontsize=12)
plt.xlabel('CO2 Emissions Per Capita',fontsize=12)
plt.title('KDE Plot')
#plt.savefig('CO2_Emissions_KDE.png')

# js - Nice KDE

# Log Likelihood for Linear Fit
def log_likelihood(theta, x, y, yerr=1):
    m, b = theta
    model = m * x + b
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

# Well done likelihood function.

np.random.seed(42) # js - the answer to the universe??

# js - did you steal this from DFM?... looks like Dan. 
nll = lambda *args: -log_likelihood(*args)
initial = np.array([slope, intercept]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial, args=(x, y))
m_ml, b_ml = soln.x

xfit_nll = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit_nll = b_ml + m_ml*xfit
plt.figure(5)
plt.plot(xfit_nll,yfit_nll,'g--',label='Log Likelihood')
#%%
# Played around with emcee, it didn't do much as expected
pos = soln.x + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(x, y))
sampler.run_mcmc(pos, 1000, progress=True);

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")


'''
js comments
-----------
 - Well done Karina!

 - Your analyses are solid, but you could have done more with the interpretation
   of what is going on and why. Remember these analyses are for you to come to 
   a deeper understanding.

'''
