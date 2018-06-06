import os
import pandas as pd
import logging.config
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler #for matplotlib colors
import seaborn as sns
from sklearn import preprocessing
from sqlalchemy import create_engine
import functions_to_derive_vars

LOGGING_CONFIG = os.getenv('LOGGING_CONFIG')
logging.config.fileConfig(LOGGING_CONFIG)
logger = logging.getLogger('prepare_data_for_eda')

DATADIR = os.getenv('DATADIR')
logger.info("using DATADIR {}".format(DATADIR))
ENGINE = os.getenv('ENGINE')
logger.info("using data from {}".format(ENGINE))
# ### Read in data



# ### Read in data
engine = create_engine(ENGINE)

logger.info('Importing facts_metrics')
facts_metrics = pd.read_sql_query('select dimensions_date_id, dimensions_item_id, pageviews, unique_pageviews, feedex_comments, is_this_useful_yes, is_this_useful_no,\
number_of_internal_searches, exits, entrances, bounce_rate,\
avg_time_on_page from "facts_metrics"',con=engine)

logger.info('Importing dates')
dates = pd.read_sql_query('select * from "dimensions_dates"',con=engine)
logger.info('Dropping {} dates duplicates'.format(sum(dates.duplicated())))
dates = dates.drop_duplicates()

logger.info('Importing items')
items = pd.read_sql_query('select id, \
content_id, title, base_path, description, number_of_pdfs, document_type, content_purpose_document_supertype,\
first_published_at, public_updated_at, number_of_word_files, status,  \
readability_score, contractions_count, equality_count, indefinite_article_count, \
passive_count, profanities_count, redundant_acronyms_count, repeated_words_count, \
simplify_count, spell_count, string_length, sentence_count, word_count, \
primary_organisation_title, primary_organisation_content_id, primary_organisation_withdrawn,\
content_hash, locale, publishing_api_payload_version from "dimensions_items"',con=engine)
logger.info('Finished importing items')

logger.info("create lists of component variables")
spelling_grammar_vars = ['contractions_count',
                         'indefinite_article_count',
                         'redundant_acronyms_count',
                         'repeated_words_count',
                         'spell_count'
                         ]

style_vars = ['readability_score',
              'equality_count',
              'passive_count',
              'simplify_count'
              ]

error_vars = ['profanities_count', 'spell_count']



# ### Join facts_metrics to specific item variables

logger.info("joining facts_metrics to items")

content_performance_bytime = pd.merge(
    left=facts_metrics,
    right=items,
    left_on='dimensions_item_id', #
    right_on='id', # 
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)

logger.info("{} items(right) matches in facts_metrics(left)".format(
    content_performance_bytime.groupby('_merge').size()
    )
)


# ### Dates to index for plots
logger.info("dates to index")
content_performance_bytime['date'] = pd.to_datetime(content_performance_bytime['dimensions_date_id'])
content_performance_bytime.index = content_performance_bytime['date']

content_performance_bytime = functions_to_derive_vars.derive_variables(content_performance_bytime, spelling_grammar_vars, style_vars, error_vars, logger)

metrics_time_independent_sum = facts_metrics[['dimensions_item_id', 'pageviews', 'unique_pageviews', 'feedex_comments',
       'is_this_useful_yes', 'is_this_useful_no',
       'number_of_internal_searches', 'exits', 'entrances']].groupby('dimensions_item_id').sum()

metrics_time_independent_ave = facts_metrics[['dimensions_item_id', 'bounce_rate',
       'avg_time_on_page']].groupby('dimensions_item_id').mean()

metrics_time_independent = pd.concat([metrics_time_independent_sum,metrics_time_independent_ave], axis=1)

logger.info("joining metrics_time_independent to items")

content_performance = pd.merge(
    left=metrics_time_independent,
    right=items,
    left_index=True, # dimensions_items_id
    right_on='id', # database specific key
    how='outer', # keep everything for checking merge
    indicator=True # so we can filter by match type
)

logger.info("{} items(right) matches in metrics_time_independent(left)".format(
    content_performance.groupby('_merge').size()
    )
)
logger.info("content_performance.content_id.nunique()={}".format(content_performance.content_id.nunique()))
logger.info("content_performance.id.nunique()={}".format(content_performance.id.nunique()))
logger.info("content_performance.shape={}".format(content_performance.shape))

content_performance = functions_to_derive_vars.derive_variables(content_performance, spelling_grammar_vars, style_vars, error_vars, logger)

logger.info("create content_management vars")
conditions = [
    (content_performance.base_path.str.startswith('/government', na=False)), 
    (content_performance.base_path.str.startswith('/guidance', na=False))]
     
choices = ['whitehall',  'guides_manuals']
content_performance['content_management'] = np.select(conditions, choices, default='mostly_mainstream')

conditions = [
    (content_performance_bytime.base_path.str.startswith('/government', na=False)), 
    (content_performance_bytime.base_path.str.startswith('/guidance', na=False))]
     
choices = ['whitehall',  'guides_manuals']
content_performance_bytime['content_management'] = np.select(conditions, choices, default='mostly_mainstream')

content_performance = content_performance[content_performance['_merge']=='both'].copy()
content_performance_bytime = content_performance_bytime[content_performance_bytime['_merge']=='both'].copy()

logger.info("content_performance.content_id.nunique()={}".format(content_performance.content_id.nunique()))
logger.info("content_performance.id.nunique()={}".format(content_performance.id.nunique()))
logger.info("content_performance.shape={}".format(content_performance.shape))

logger.info("content_performance_bytime.content_id.nunique()={}".format(content_performance_bytime.content_id.nunique()))
logger.info("content_performance_bytime.id.nunique()={}".format(content_performance_bytime.id.nunique()))
logger.info("content_performance_bytime.shape={}".format(content_performance_bytime.shape))

dtypes_bytime = dict(zip(list(content_performance_bytime),[content_performance_bytimef[x].dtype.name for x in content_performance_bytime]))

logger.info("writing content_performance to csv")
#content_performance.to_pickle(os.path.join(DATADIR, 'content_performance.pkl.compress'),  compression='xz')
content_performance.to_csv(os.path.join(DATADIR, 'content_performance.csv.gz'),  compression='gzip')

logger.info("writing content_performance_by_time to csv")
#content_performance_bytime.to_pickle(os.path.join(DATADIR, 'content_performance_bytime.pkl.compress'),  compression='xz')
content_performance_bytime.to_csv(os.path.join(DATADIR, 'content_performance_bytime.csv.gz'),  compression='gzip')
logger.info("finished writing content_performance to csv")

