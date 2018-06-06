---

author: "Ellie king"
date: "5/23/2018"

---

This repo contains the notebooks behind the exploratory data analysis (EDA) which was conducted on GOV.UK warehouse data (extracted 15MAY18) for the data-informed-content team. The presentation can be found here: 

https://docs.google.com/presentation/d/18pYJY0gpOJHVZtUJ-mQzUiuIlWO3XZfjcIYFpLdd3sg/edit#slide=id.g3bbbc926d1_5_0

In order to reproduce or update these analyses, follow the instructions below:

Clone this repo and mkdir DATA outside of repo

## Get a recent download of the warehouse data

https://trello.com/c/I1LA2tB2/319-05-science-time-where-is-my-data

## Setting up local PostgreSQL

Download and unzip the CPM-*.dump.gz and
move it to DATA directory

Create the local database:  
`createdb DATABASE_NAME`  
Import data into database:  
`psql -d DATABASE_NAME -f DATA/CPM-*.dump`  


Set environment variables:

```
export DATADIR="../DATA"
export LOGGING_CONFIG="$PWD/python/logging.conf"
export ENGINE="postgresql://USERNAME@localhost:5432/DATABASE_NAME"
```

## Preparing the data for exploratory data analyses

Run the script to prepare and save the data (this takes ages to save)  
`python3 prepare_data_for_eda.py`

### Setting up a Jupyter kernel
Run jupyter lab or jupyter notebook to use exploratory data analyses notebooks  

If using virtualenv:  

Install the ipython kernel module into your virtualenv

`workon my-virtualenv-name`  # activate your virtualenv  

if you haven't already  
`pip install ipykernel`

Now run the kernel "self-install" script:  

`python -m ipykernel install --user --name=my-virtualenv-name`  
Replacing the --name parameter as appropriate.

You should now be able to see your kernel in the IPython notebook menu: Kernel -> Change kernel and be able so switch to it (you may need to refresh the page before it appears in the list). IPython will remember which kernel to use for that notebook from then on.

## Exploratory data analysis notebooks
The exploratory data analysis is split over several jupyter notebooks so that it's easier to find things and quicker to run them. They're listed in here in a reasonable order if browsing the whole EDA. 

| Notebook        | contents                                                                             | 
| ------------- |-------------| 
| EDA_data_quality      |describes missing data, duplicates and basic counts | 
| EDA_distributions     |describes univariate distributions of warehouse variables      |  
| EDA_over_time|describes trends over time      |  
| EDA_bivariate|describes correlations between variables      |
| EDA_usefulness_survey|describes response rate and usfulness rating, citizens advice Backlogger with GOV.UK data |
| exploratory_models|baseline models to predict usefulness rating (based on everything we know) and explain usefulness using all variables, variables that can be improved thorugh editing, and variables that we can measure the first time content is published|

