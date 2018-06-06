'''
Functions used to derive variables in the eda.ipynb.
'''

import pandas as pd
import numpy as np
from sklearn import preprocessing

# # Derived variable functions


# - **ratio of page_views:unique_pageviews** if someone accessed the same page 8 times, it is either very useful and constantly referred to or not quick to understand, requiring many attempts
# - **ratio of feedex_comments:unique_pageviews** of all the people visiting the page, what proportion have enough oomph to write a comment. Beware seleciton bias, different page types may attract different people who are more/less likely to repsond/respond in a particular way
# - **composite grammar** can all spelling grammar metrics be simply combined?
# - **days since published** might need a function to do days since published rather than working with first_published_at. Could be useful for other age objects too, if any modelling done
def create_count_survey_responses(df, logger):
    logger.info("create count_survey_responses")

    df['count_survey_responses'] = df['is_this_useful_yes'] + df['is_this_useful_no']
    return df['count_survey_responses']

def create_response_per_pageview(df, logger):
    
    logger.info("create response_per_pageview")

    df['response_per_pageview'] = df['count_survey_responses'] / df['pageviews']
    return df['response_per_pageview']
 
def create_response_per_unique_pageview(df, logger):
    logger.info("create response_per_unique_pageview")

    df['response_per_unique_pageview'] = df['count_survey_responses'] / df['unique_pageviews'] 
    return df['response_per_unique_pageview']

def create_useful_per_responses(df, logger):
    logger.info("create useful_per_responses")

    df['useful_per_responses'] = df['is_this_useful_yes'] / df['count_survey_responses']
    return df['useful_per_responses']

def create_total_to_unique_pageviews(df, logger):
    logger.info("create mean_views_per_session")

    df['total_to_unique_pageviews'] = df['pageviews'] / df['unique_pageviews']
    return df['total_to_unique_pageviews']

def create_feedex_per_unique(df, logger):
    logger.info("create feedex_per_unique")

    df['feedex_per_unique_1000'] = df['feedex_comments'] / df['unique_pageviews'] * 1000
    return df['feedex_per_unique_1000']

def create_searches_per_pageview(df, logger):
    logger.info("create searches_per_pageview")

    df['searches_per_pageview_1000'] = df['number_of_internal_searches'] / df['pageviews'] * 1000
    return df['searches_per_pageview_1000']

def create_consolidated_format(df, logger):
    logger.info("create consolidated_format")

    df['consolidated_format'] = df['document_type'].astype('category')
    df['consolidated_format'].cat.rename_categories({1: 'x', 2: 'y', 3: 'z'})
    
    return df['searches_per_pageview_1000']

def derive_variables(df, spelling_grammar_vars, style_vars, error_vars, logger):
    """call individual derived variable functions and loop through spelling/grammar/error 
    lists to created scaled vars and then composite vars"""
    
    df['count_survey_responses'] = create_count_survey_responses(df, logger)
    df['response_per_pageview'] = create_response_per_pageview(df, logger)
    df['response_per_unique_pageview'] = create_response_per_unique_pageview(df, logger)
    df['useful_per_responses'] = create_useful_per_responses(df, logger)
    df['create_total_to_unique_pageviews'] = create_total_to_unique_pageviews(df, logger)
    df['feedex_per_unique_1000'] = create_feedex_per_unique(df, logger)
    df['searches_per_pageview_1000'] = create_searches_per_pageview(df, logger)

    logger.info("Loop through and min-max scale component vars")

    for var in spelling_grammar_vars + style_vars + error_vars:
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        min_max_scaler = preprocessing.MinMaxScaler()

        if var != 'readability_score':
            var_imputed = imp.fit_transform(df[var].values.reshape(-1, 1))
            var_scaled = min_max_scaler.fit_transform(var_imputed)

        else:
            neg_var = -df[var].values.reshape(-1, 1) # because a low score is bad and we need it in same direction as counts of bad things
            var_imputed = imp.fit_transform(neg_var)
            var_scaled = min_max_scaler.fit_transform(var_imputed)

        df[str(var) + '_scaled'] = var_scaled

    logger.info("Sum component vars for each composite var")

    df['spelling_grammar'] = (df['contractions_count_scaled'] +
                                df['indefinite_article_count_scaled'] +
                                df['redundant_acronyms_count_scaled'] +
                                df['repeated_words_count_scaled'] +
                                df['spell_count_scaled']
                             )

    logger.info("spelling_grammar: {}".format(df['spelling_grammar'].describe()))

    df['style'] = (df['readability_score_scaled'] +
                                           df['equality_count_scaled'] +
                                           df['passive_count_scaled'] +
                                           df['simplify_count_scaled']
                                           )
    df['style'].describe()

    logger.info("style: {}".format(df['style'].describe()))


    df['errors'] = (df['profanities_count_scaled'] + df['spell_count_scaled'])

    logger.info("errors: {}".format(df['errors'].describe()))
    
    return df


