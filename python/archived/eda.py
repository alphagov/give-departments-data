


import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler #for matplotlib colors
import seaborn as sns
from sklearn import preprocessing
from sqlalchemy import create_engine


# # Exploratory data analyses
# 1. Date of metric by fact, by document type/primary org (e.g., page views over date-dimension stratified by document type)
# 2. content-specific performance metric (e.g. reading age) by user-related performance metric (e.g., page views) over a defined time-period (min-max in date dimension)

# ### Read in data

DATADIR = os.getenv('DATADIR')


engine = create_engine('postgresql://ellieking@localhost:5432/givedata')


# In[4]:


facts_metrics = pd.read_sql_query('select * from "facts_metrics"',con=engine)


# In[5]:


facts_metrics.head()


# In[6]:


dates = pd.read_sql_query('select * from "dimensions_dates"',con=engine)
dates = dates.drop_duplicates()


# In[7]:


dates


# In[8]:


items = pd.read_sql_query('select * from "dimensions_items"',con=engine)


# In[9]:


items.columns


# ### Distribution of metrics

# ### Join facts_metrics to specific item variables

# In[10]:


content_performance_bytime = pd.merge(
    left=facts_metrics,
    right=items,
    left_on='dimensions_item_id', # which taxon is the content item tagged to
    right_on='id', # what is the id of that taxon
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)


# In[11]:


content_performance_bytime.groupby('_merge').size()


# In[12]:


content_performance_bytime[content_performance_bytime['_merge']=='right_only']


# ### Dates to index for plots

# In[13]:


content_performance_bytime['date'] = pd.to_datetime(content_performance_bytime['dimensions_date_id'])
content_performance_bytime.index = content_performance_bytime['date']


# ## Metric over time, by doc type

# In[14]:


content_performance_bytime.groupby(pd.Grouper(key='date', freq='D', sort=True))['pageviews'].sum()


# In[15]:


content_performance_bytime.groupby(pd.Grouper(key='date', sort=True))['pageviews'].sum()


# In[16]:


def plot_time_metric(df, metric):
    grouped = df.groupby([df.index, pd.Grouper(freq='D')])[metric].sum() #resample operation for each day in datime index, sum the metric
    grouped.index = grouped.index.droplevel()
    ax = grouped.plot()
    ax.set_ylabel(metric)
    ax.set_xlabel('Date')

    return ax
    


# In[17]:


plot_time_metric(content_performance_bytime, 'pageviews')


# In[18]:


plot_time_metric(content_performance_bytime, 'unique_pageviews')


# In[19]:


plot_time_metric(content_performance_bytime, 'feedex_comments')


# ### trying to get weekday onto plot. 
# Aborted for now. think ax.table might work

# In[20]:


grouped = content_performance_bytime.groupby([content_performance_bytime.index, pd.Grouper(freq='D')])['pageviews'].sum()
grouped.index = grouped.index.droplevel()


# In[21]:


grouped = grouped.to_frame()


# In[22]:


grouped['day'] = grouped.index.weekday_name


# In[23]:


grouped


# In[24]:


ax = grouped.plot()
ax.set_ylabel('Pageviews (100M)')
ax.set_xlabel('Date')
plt.show()


# In[25]:


list(zip(grouped.index, grouped.index.weekday_name))


# That's a bit odd. Assumed 12/13th would be weekend but they're Thurs/Friday. 

# In[26]:


grouped.index.weekday_name


# ### Stratify by categorical variables

# In[27]:


def plot_time_metric_byvar(df, metric, byvar):
    grouped = df.groupby([byvar, pd.Grouper(freq='D')])[metric].sum()
    by_day = grouped.unstack(byvar, fill_value=0)
    top = by_day.iloc[:, by_day.columns.isin(by_day.min().sort_values(ascending=False)[:10].index)]
    bottom = by_day.iloc[:, by_day.columns.isin(by_day.min().sort_values()[:10].index)]
    
    ax = top.plot()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel(metric)
    ax.set_xlabel('Date')
    ax.set_title('Top 10 {}s for {}'.format(byvar, metric))
    
    ay = bottom.plot()
    ay.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ay.set_ylabel(metric)
    ay.set_xlabel('Date')
    ay.set_title('Bottom 10 {}s for {}'.format(byvar, metric))

    return ax, ay
    


# In[28]:


plot_time_metric_byvar(df=content_performance_bytime, metric='pageviews', byvar='document_type')


# In[29]:


plot_time_metric_byvar(df=content_performance_bytime, metric='unique_pageviews', byvar='document_type')


# In[30]:


plot_time_metric_byvar(df=content_performance_bytime, metric='feedex_comments', byvar='document_type')


# Document types with no page views in date range

# In[31]:


#over the whole month (includes entire data range) total number of page views 

x = content_performance_bytime.groupby(['document_type', pd.Grouper(freq='M')])['pageviews'].sum()


# In[32]:


#which documen types had no page views
x[x==0]


# In[33]:


plot_time_metric_byvar(df=content_performance_bytime, metric='pageviews', byvar='primary_organisation_title')


# In[34]:


plot_time_metric_byvar(df=content_performance_bytime, metric='unique_pageviews', byvar='primary_organisation_title')


# In[35]:


plot_time_metric_byvar(df=content_performance_bytime, metric='feedex_comments', byvar='primary_organisation_title')


# In[36]:


plot_time_metric_byvar(df=content_performance_bytime, metric='pageviews', byvar='content_purpose_document_supertype')


# In[37]:


plot_time_metric_byvar(df=content_performance_bytime, metric='unique_pageviews', byvar='content_purpose_document_supertype')


# In[38]:


plot_time_metric_byvar(df=content_performance_bytime, metric='feedex_comments', byvar='content_purpose_document_supertype')


# # Intrinsic content performance metrics
# Explore those metrics generated by characteristics of the content itself and their correlation with metrics relating to user interaction with content.

# In[39]:


metrics_time_independent = facts_metrics.groupby('dimensions_item_id').sum()


# In[40]:


metrics_time_independent.shape


# In[41]:


metrics_time_independent = metrics_time_independent.drop(['id'], axis=1)


# In[42]:


content_performance = pd.merge(
    left=metrics_time_independent,
    right=items,
    left_index=True, # dimensions_items_id
    right_on='id', # database specific key
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)


# In[43]:


#content_performance[content_performance.duplicated(subset='content_id', keep=False)].sort_values(by='content_id')


# In[44]:


content_performance.content_id.nunique()


# In[45]:


content_performance.shape


# <span style="color:red">There are multiple ids per content_id reflecting a change to the content e.g., re-written.</span>
# 
# **Need to think about left/right censoring for these items when considering date ranges**

# In[46]:


content_performance.columns


# In[47]:


def scatter_byvar(df, x, y, byvar, log=True):
    groups = df.groupby(byvar)

    # Plot
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(groups)))

    fig, ax = plt.subplots()
    ax.set_prop_cycle(cycler('color', colors))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    
    ax.set_xlabel(x)
    ax.set_title('{} and {} by {}'.format(x, y, byvar))
    
    if log:
        for name, group in groups:
            ax.plot(group[x], np.log(group[y]), marker='o', linestyle='',  label=name, alpha=0.5)
            ax.set_ylabel('log({})'.format(y))
    else:
        for name, group in groups:
            ax.plot(group[x], group[y], marker='o', linestyle='',  label=name, alpha=0.5 )
            ax.set_ylabel(y)
            
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return ax


# #### string length

# In[48]:


content_performance.string_length.describe()


# In[49]:


content_performance[content_performance['string_length']==0].shape


# In[50]:


print('{} out of {} content items ({}%) have a recorded string_length of 0'.format(
    content_performance[content_performance['string_length']==0].shape[0], 
    content_performance.content_id.nunique(),
    round(content_performance[content_performance['string_length']==0].shape[0]/content_performance.content_id.nunique()*100, 2)))


# In[51]:


content_performance.string_length.hist(bins='auto', range=(0, 7000))


# In[52]:


scatter_byvar(df=content_performance, x='string_length', y='unique_pageviews', byvar='content_purpose_document_supertype')


# #### readability score
# "Check readability"?!

# In[53]:


content_performance.readability_score.hist(bins='auto', range=(-750, 124))


# In[54]:


content_performance.readability_score.describe()


# In[55]:


scatter_byvar(df=content_performance, x='readability_score', y='unique_pageviews', byvar='content_purpose_document_supertype')


# #### Number of pdfs

# In[56]:


content_performance.number_of_pdfs.describe()


# In[57]:


content_performance.number_of_pdfs.hist(bins=100, range=(0, 50))


# In[58]:


scatter_byvar(df=content_performance, x='number_of_pdfs', y='unique_pageviews', byvar='content_purpose_document_supertype')


# #### Number of word files

# In[59]:


content_performance.number_of_word_files.describe()


# In[60]:


content_performance.number_of_word_files.hist(bins=100, range=(0, 50))


# In[61]:


scatter_byvar(df=content_performance, x='number_of_word_files', y='unique_pageviews', byvar='content_purpose_document_supertype')


# #### retractions
# Check apostrophe use in contractions. 
# This should feed into a general ‘spelling/grammar errors’ metric.

# In[62]:


scatter_byvar(df=content_performance, x='contractions_count', y='unique_pageviews', byvar='content_purpose_document_supertype')


# #### retext-equality 
# - Warn about possible insensitive, inconsiderate language. This has some interesting changes (‘commit suicide’ to ‘die by suicide’, for example). Not sure how publishers would respond to this if they had lots of changes, but inclusive language should be important for us.

# In[63]:


scatter_byvar(df=content_performance, x='equality_count', y='unique_pageviews', byvar='content_purpose_document_supertype')


# #### retext-indefinite-article
# Check if indefinite articles (a, an) are used correctly. This should feed into a general ‘spelling/grammar errors’ metric.
# 

# In[64]:


scatter_byvar(df=content_performance, x='indefinite_article_count', y='unique_pageviews', byvar='content_purpose_document_supertype')


# In[65]:


#scatter_byvar(df=content_performance, x='readability_score', y='unique_pageviews', byvar='document_type')


# In[66]:


scatter_byvar(df=content_performance, x='readability_score', y='unique_pageviews', byvar='content_purpose_document_supertype')


# ## metric by content age

# In[67]:


content_performance.columns


# In[68]:


first_published = content_performance[['first_published_at', 'pageviews', 'unique_pageviews', 'document_type', 'content_purpose_document_supertype']].copy()


# In[69]:


first_published['first_published_at'] = pd.to_datetime(first_published['first_published_at']).copy()
first_published.index = first_published['first_published_at']


# In[70]:


first_published.plot(x='first_published_at', y='pageviews')


# Old items are generally not being viewed

# In[71]:


ax = first_published.plot(x='first_published_at', y='pageviews')
ax.set_xlim(pd.Timestamp('2009-12-31'), pd.Timestamp('2017-12-31'))


# In[72]:


ay = first_published.plot(x='first_published_at', y='pageviews')
ay.set_xlim(pd.Timestamp('1945-12-31'), pd.Timestamp('2009-12-31'))
ay.set_ylim(0, 1000)


# In[73]:


ax = first_published[first_published['document_type']=='guidance'].plot(x='first_published_at', y='pageviews', color='DarkBlue', label='Group 1', style=".")
first_published[first_published['document_type']=='news_story'].plot(x='first_published_at', y='pageviews', color='LightGreen', label='Group 2', style=".", ax=ax)
first_published[first_published['document_type']=='world_news_story'].plot(x='first_published_at', y='pageviews', color='DarkGreen', label='Group 3', style=".", ax=ax)


# In[74]:


groups = first_published.groupby('content_purpose_document_supertype')


# In[75]:


first_published.groupby('content_purpose_document_supertype').describe()


# In[76]:


first_published.groupby('content_purpose_document_supertype').groups['other'].min()


# In[77]:


for group in groups:
    print(min(group[1].first_published_at))


# In[78]:


first_published.index.min()


# In[79]:


# Plot
cmap = plt.get_cmap('nipy_spectral')
colors = cmap(np.linspace(0, 1.0, len(groups)))

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', colors))

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    
    ax.plot(group['first_published_at'], group['pageviews'], marker='o', linestyle='', ms=5, label=name)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


plt.show()


# ALERT! Cannot understand why this 1686 data is coming from. It is not showing up when I print the timestamps. how is it getting into plot ?
# 
# 

# In[80]:


# Plot
cmap = plt.get_cmap('nipy_spectral')
colors = cmap(np.linspace(0, 1.0, len(groups)))

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', colors))

ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    
    ax.plot(group['first_published_at'], group['pageviews'], marker='o', linestyle='', ms=5, label=name)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim(pd.Timestamp('1940-12-31'), pd.Timestamp('2018-12-31'))

plt.show()


# In[81]:


# Plot
cmap = plt.get_cmap('nipy_spectral')
colors = cmap(np.linspace(0, 1.0, len(groups)))

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', colors))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    
    ax.plot(group['first_published_at'], group['pageviews'], marker='o', linestyle='', ms=5, label=name)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim(pd.Timestamp('2009-12-31'), pd.Timestamp('2018-12-31'))

plt.show()


# In[82]:


# Plot
cmap = plt.get_cmap('nipy_spectral')
colors = cmap(np.linspace(0, 1.0, len(groups)))

fig, ax = plt.subplots()
ax.set_prop_cycle(cycler('color', colors))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    
    ax.plot(group['first_published_at'], group['pageviews'], marker='o', linestyle='', ms=5, label=name, alpha=0.8)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_xlim(pd.Timestamp('1945-12-31'), pd.Timestamp('2009-12-31'))
ax.set_ylim(0, 1000)
plt.show()


# In[83]:


ax = first_published[first_published['content_purpose_document_supertype']=='guidance'].plot(x='first_published_at', y='pageviews', color='DarkBlue', label='Guidance', style=".")
first_published[first_published['content_purpose_document_supertype']=='navigation'].plot(x='first_published_at', y='pageviews', color='Red', label='navigation', style=".", ax=ax)
first_published[first_published['content_purpose_document_supertype']=='transactions'].plot(x='first_published_at', y='pageviews', color='LightGreen', label='transactions', style=".", alpha=0.5, ax=ax)

