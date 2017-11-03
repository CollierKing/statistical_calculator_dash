import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from scipy.stats import norm
from sklearn.neighbors import KernelDensity

#additional:
import numpy as np,pandas as pd
import numpy.random as npr

## sample size
from scipy.stats import norm, zscore
## chisquared
from statsmodels.stats.proportion import proportions_chisquare
## z-score converter
from scipy.stats import norm
## t test
from scipy.stats import ttest_ind, ttest_ind_from_stats
## poisson test
from scipy.stats import binom_test
from scipy.stats import beta
from scipy.stats import poisson
# survival
from lifelines.statistics import logrank_test
# price elasticity
import statsmodels.api as sm
# r-squared
# from sklearn.metrics import pearson_correl_output
from sklearn.metrics import r2_score

print(dcc.__version__) # 0.6.0 or above is required

app = dash.Dash()

# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.supress_callback_exceptions = True

# HOME
##############################################################################
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Br(),
])

markdown_intro = '''
    This website is a collection of calculators for solving certain statistical questions.
    '''

index_page = html.Div([
    html.H1('Calculators'),
    
    dcc.Markdown(children=markdown_intro),
    html.Br(),

    html.H2('Statistical:'),
    dcc.Link('Sample Size Calculator', href='/sample_size_calculator'),
    html.Br(),
    dcc.Link('Chi Squared Test', href='/chi_squared_test'),
    html.Br(),
    dcc.Link('2 Sample T-Test',href='/2_sample_ttest'),
    html.Br(),
    dcc.Link('2 Sample Survival Times',href='/2_sample_survival'),
    html.Br(),
    dcc.Link('Count Data',href='/count_data'),
    html.Br(),
    dcc.Link('Pearson Correlation',href='/pearson_correl'),
    html.Br(),
    html.H2('Financial:'),
    dcc.Link('Credit Value at Risk',href='/cvar'),
    html.Br(),
    html.H2('Economic:'),
    dcc.Link('Price Elasticity of Demand - Values',href='/pe'),
    html.Br(),
    dcc.Link('Price Elasticity of Demand - Raw Data',href='/pe_raw')

])

# PAGE 1 - SAMPLE SIZE CALCULATOR
##############################################################################
page_1_layout = html.Div([
        
    html.H1('Sample Size Calculator'),
    dcc.Link('Go back to home', href='/'),

    html.Div(
        [
            # INPUT BOX 1 - CONV RATE
            html.Label('Baseline Rate'),
            dcc.Input(
                id='baseline_input',
                placeholder='0.3',
                type='text',
                value='0.3'
            ),
            html.Br(),

            # INPUT BOX 2 - EFFECT SIZE
            html.Label('Effect Size'),
            dcc.Input(
                id='effect_size_input',
                placeholder='0.05',
                type='text',
                value='0.35'
            ),
            html.Br(),

            # INPUT BOX 3 - POWER LEVEL
            html.Label('Power (1-False Negative Rate)'),
            dcc.Input(
                id='statistical_power_input',
                placeholder='0.8',
                type='text',
                value='0.8'
            ),
            html.Br(),

            # INPUT BOX 4 - SIGNIF LEVEL
            html.Label('Significance Level'),
            dcc.Input(
                id='significance_level_input',
                placeholder='0.05',
                type='text',
                value='0.05'
            ),
            html.Br(),

            html.Label('Sample Size:'),
            html.Div(id='sample_size_content'),
            html.Br(),
            html.Br(),
        ],
    ),
    html.Br(),
    html.Br(),
    html.Div(
        [
            dcc.Graph(id='example-graph')
        ],
    ),
    html.Br(),
    html.Br()
    
],style={'columnCount': 2})

#BAR GRAPH - SAMPLE SIZE
@app.callback(dash.dependencies.Output('example-graph','figure'),
            [dash.dependencies.Input('baseline_input','value'),
            dash.dependencies.Input('effect_size_input','value')
            ])
def update_graph(baseline_input,effect_size_input):
    err_bar = float(effect_size_input)-float(baseline_input)
    return {
            'data': [
                {'x': [1], 
                'y': [baseline_input], 
                'error_y':{
                    "array":[err_bar]},
                'type': 'bar', 'name': 'SF'},
                ],
                'layout': {
                    'title': 'Baseline with Effect'
                }
            }

#SAMPLE SIZE
@app.callback(dash.dependencies.Output('sample_size_content', 'children'),
              [dash.dependencies.Input('baseline_input', 'value'),
               dash.dependencies.Input('effect_size_input', 'value'),
               dash.dependencies.Input('significance_level_input','value'),
               dash.dependencies.Input('statistical_power_input', 'value')
              ])
def sample_size_content(baseline_input, effect_size_input,significance_level_input, statistical_power_input):
    z = norm.isf([float(significance_level_input)/2]) #two-sided t test
    zp = -1 * norm.isf([float(statistical_power_input)])
    d = (float(baseline_input)-float(effect_size_input))
    s = 2*((float(baseline_input)+float(effect_size_input)) /2)*\
    (1-((float(baseline_input)+float(effect_size_input)) /2))
    n = s * ((zp + z)**2) / (d**2)
    n = int(round(n[0]))
    # value = "lol noob"
    return 'The Sample Size Is: "{}"'.format(n)


#PAGE 2 - CHI-SQUARED TESTS
##############################################################################
##############################################################################
#Text
markdown_text_chi_squared = '''
    A Chi-Squared-Test is...
'''

page_2_layout = html.Div([
    html.H1('Chi Squared Test'),
    dcc.Markdown(children=markdown_text_chi_squared),
    dcc.Link('Go back to home', href='/'),
    html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Sample1 Successes
            html.Label('Sample 1: # successes'),
            dcc.Input(
                id='sample1_successes',
                placeholder='150',
                type='text',
                value='150'
            ),

            html.Br(),

            # INPUT BOX 2 - Sample1 Trials
            html.Label('Sample 1: # trials'),
            dcc.Input(
                id='sample1_trials',
                placeholder='1000',
                type='text',
                value='1000'
            ),

            html.Br(),

            # INPUT BOX 3 - Sample2 Successes
            html.Label('Sample 2: # successes'),
            dcc.Input(
                id='sample2_successes',
                placeholder='180',
                type='text',
                value='180'
            ),

            html.Br(),

            # INPUT BOX 4 - Sample2 Trials
            html.Label('Sample 2: # trials'),
            dcc.Input(
                id='sample2_trials',
                placeholder='1000',
                type='text',
                value='1000'
            ),
            html.Br(),

            # INPUT BOX 5 - Confidence Level
            html.Label('Confidence Level:'),
            dcc.Input(
                id='chisq_confidence_level',
                placeholder='0.95',
                type='text',
                value='0.95'
            ),
            html.Br(),
            html.Br(),

            html.Label('Result p-value:'),
            html.Div(id='chisq_content'),
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    html.Div(
        [
            dcc.Graph(id='chisq-graph')

        ]),


    html.Br(),
    html.Br()

],style={'columnCount': 2})

#BAR GRAPH - CHISQ - SAMPLE PROPORTIONS
@app.callback(dash.dependencies.Output('chisq-graph','figure'),
            [dash.dependencies.Input('sample1_trials','value'),
            dash.dependencies.Input('sample1_successes','value'),
            dash.dependencies.Input('sample2_trials','value'),
            dash.dependencies.Input('sample2_successes','value'),
            dash.dependencies.Input('chisq_confidence_level','value')
            ])
def update_chisqgraph(sample1_trials,sample1_successes,
                        sample2_trials,sample2_successes,
                        chisq_confidence_level):
    #calculate sample proportions
    sample1_prop = int(sample1_successes)/int(sample1_trials)
    sample2_prop = int(sample2_successes)/int(sample2_trials)
    #calculate confidence intervals
    z_score = norm.ppf(1-(1-float(chisq_confidence_level))/2)

    err1 = (float(z_score))* \
            np.sqrt( (sample1_prop * ( 1-sample1_prop))/int(sample1_trials))
    err2 = (float(z_score))* \
            np.sqrt( (sample2_prop * ( 1-sample2_prop))/int(sample2_trials))
    return {
            'data': [
                {'x': [1,2], 
                'y': [sample1_prop,sample2_prop], 
                'error_y':{
                    "array":[err1,err2]},
                'type': 'bar', 'name': 'SF2'},
                ],
                'layout': {
                    'title': 'Sample Proportions'
                }
            }
# ChiSq
@app.callback(dash.dependencies.Output('chisq_content', 'children'),
              [dash.dependencies.Input('sample1_successes', 'value'),
               dash.dependencies.Input('sample1_trials', 'value'),
               dash.dependencies.Input('sample2_successes','value'),
               dash.dependencies.Input('sample2_trials', 'value')
              ])
def chi_squared_result(sample1_successes, sample1_trials,sample2_successes, sample2_trials):
    successes = np.array([int(sample1_successes),int(sample2_successes)])
    trials = np.array([int(sample1_trials),int(sample2_trials)])
    result = proportions_chisquare(successes,trials)
    p = result[1]
    return 'The p-value is: "{}"'.format(p)


#PAGE 3 - 2 SAMPLE T-TESTS
##############################################################################
##############################################################################
page_3_layout = html.Div([

    html.H1('Two Sample T-Test'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.Label('Sample 1 Values:'),
    dcc.Textarea(
        id='sample1_dat_ttest',
        placeholder="Paste Samples Here",
            value="54,231,554,322,553,111",
        style={'width': '20%',
        'height':'50%'},
    ),
    html.Br(),
    html.Label('Sample 2 Values:'),
    dcc.Textarea(
        id='sample2_dat_ttest',
        placeholder='Paste Samples Here',
        value='54,231,554,322,553,111',
        style={'width': '20%'}
    ),
    html.Br(),
    html.Div(id='sample1_dat_summ'),
    # html.Div([
    #         dcc.Graph(id='ttest-graph')

    #     ]),
    # html.Br(),
    html.Label('T Test P-Value:'),
    html.Div(id='ttest_result'),
    html.Br(),
    # dcc.Link('Go to Page 1', href='/sample_size_calculator'),
    html.Br(),

    ])

#TTEST - RESULT
@app.callback(dash.dependencies.Output('ttest_result', 'children'),
              [dash.dependencies.Input('sample1_dat_ttest', 'value'),
              dash.dependencies.Input('sample2_dat_ttest', 'value')
              ])
def sample1_dat_summ(sample1_dat_ttest,sample2_dat_ttest):
    #sample 1 input
    #if comma separated values
    if "," in sample1_dat_ttest:
        p1 = np.asarray(sample1_dat_ttest.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample1_dat_ttest:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p1 = np.asarray(sample1_dat_ttest.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p1 = np.asarray(sample1_dat_ttest.split(','),dtype=np.float32)
    #sample 2 input
    #if comma separated values
    if "," in sample2_dat_ttest:
        p2 = np.asarray(sample2_dat_ttest.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample2_dat_ttest:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p2 = np.asarray(sample2_dat_ttest.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample2_dat_ttest = sample2_dat_ttest.replace('\n', ',')
        p2 = np.asarray(sample2_dat_ttest.split(','),dtype=np.float32)

    t, p = ttest_ind(p1, p2, equal_var=False)
    # return p
    return 'The p-value is: "{}"'.format(p)




# KERNEL DENSITY - TTEST
# @app.callback(dash.dependencies.Output('ttest-graph','figure'),
#             [dash.dependencies.Input('sample1_dat_ttest','value')])
# def update_ttestgraph(sample1_dat_ttest):
#     #if comma separated values
#     if "," in sample1_dat_ttest:
#         p = np.asarray(sample1_dat_ttest.split(','),dtype=np.float32)
#         # p = sum(p)
#     # #if space separated values
#     elif " " in sample1_dat_ttest:
#         # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
#         p = np.asarray(sample1_dat_ttest.split(' '),dtype=np.float32)
#         # p = sum(p)
#     # #if line delimited values
#     else:
#         sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
#         p = np.asarray(sample1_dat_ttest.split(','),dtype=np.float32)
#         # p = sum(p)
#     # np.random.seed(1)
#     # N = 20
#     # X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
#     #                     np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
#     # X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#     # bins = 10
#     # X = eval('[' + sample1_dat_ttest + ']')
#     # sample1_dat_ttest = sample1_dat_ttest.split(',')
#     # p = np.asarray(p)
    
#     # np.random.seed(1)
#     # N = 20
#     # X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
#     #                     np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
#     # X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
#     # bins = 10


#     # kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
#     # log_dens = kde.score_samples(X_plot)

#     return {
#         # go.Figure(
#         'data':[
#             # go.Scatter(x=X, y=np.exp(log_dens), 
#             #                 mode='lines', fill='tozeroy',
#             #                 line=dict(color='#AAAAFF', width=2)),
#             # {'x':[X],
#             # 'y':[np.exp(log_dens)], 
#             # 'mode':'lines', 
#             # 'fill':'tozeroy',
#             # 'line':dict(color='#AAAAFF', width=2)},

#             trace0 = go.Scatter(
#                 x = random_x,
#                 y = random_y0,
#                 mode = 'lines',
#                 name = 'lines'
#             )

#             go.Bar(
#                 x=[1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
#                    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012],
#                 y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,
#                    350, 430, 474, 526, 488, 537, 500, 439],
#                 name='Rest of world',
#                 marker=go.Marker(
#                     color='rgb(55, 83, 109)'
#                 )
#             )
#             ]
#             # 'data': [
#             #     {'x': [1,2], 
#             #     'y': [sample1_prop,sample2_prop], 
#             #     'error_y':{
#             #         "array":[err1,err2]},
#             #     'type': 'bar', 'name': 'SF2'},
#             #     ],
#             #     'layout': {
#             #         'title': 'Dash Data Visualization'
#             #     }
#         }



#PAGE 4 - 2 SAMPLE SURVIVAL TIMES
##############################################################################
##############################################################################
page_4_layout = html.Div([

    html.H1('Survival Times'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    # INPUT BOX 1 - Sample 1
    html.Label('Sample 1:'),
    dcc.Textarea(
        id='sample1_dat_survival',
        placeholder="Paste Samples Here",
            value="1,1,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23",
        style={'width': '20%',
        'height':'50%'},
    ),
    html.Br(),
    # INPUT BOX 2 - Sample 2
    html.Label('Sample 2:'),
    dcc.Textarea(
        id='sample2_dat_survival',
        placeholder='Paste Samples Here',
        value='6,6,7,9,10,11,13,15,16,19,20,22,23,32,6,10,17,19,24,25,25,28,28,32,33,34,35,39',
        style={'width': '20%'}
    ),
    html.Br(),
    # INPUT BOX 3 - Confidence Level
    html.Label('Confidence Level:'),
    dcc.Input(
        id='survival_confidence_level',
        placeholder='0.95',
        type='text',
        value='0.95'
    ),
    html.Label('Log Rank P-Value:'),
    html.Div(id='survival_output'),
    html.Br(),
        html.Div([
            dcc.Graph(id='survival-graph')

        ]),

    ])

#RESULT- SURVIVAL 
@app.callback(
    dash.dependencies.Output('survival_output','children'),
            [dash.dependencies.Input('sample1_dat_survival','value'),
            dash.dependencies.Input('sample2_dat_survival','value'),
            dash.dependencies.Input('survival_confidence_level','value')
            ])
def update_survival(sample1_dat_survival,sample2_dat_survival,
                    survival_confidence_level):
    #sample 1 input
    #if comma separated values
    if "," in sample1_dat_survival:
        p1 = np.asarray(sample1_dat_survival.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample1_dat_survival:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p1 = np.asarray(sample1_dat_survival.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample1_dat_survival = sample1_dat_survival.replace('\n', ',')
        p1 = np.asarray(sample1_dat_survival.split(','),dtype=np.float32)
    #sample 2 input
    #if comma separated values
    if "," in sample2_dat_survival:
        p2 = np.asarray(sample2_dat_survival.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample2_dat_survival:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p2 = np.asarray(sample2_dat_survival.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample2_dat_survival = sample2_dat_survival.replace('\n', ',')
        p2 = np.asarray(sample2_dat_survival.split(','),dtype=np.float32)
    
    x = logrank_test(p1, p2, alpha=float(survival_confidence_level))

    p_result = float(x.p_value)

    return 'The p-value is: "{}"'.format(p_result)
    # return p2

#LINE GRAPH - SURVIVAL
@app.callback(dash.dependencies.Output('survival-graph','figure'),
            [dash.dependencies.Input('sample1_dat_survival','value'),
            dash.dependencies.Input('sample2_dat_survival','value'),
            dash.dependencies.Input('survival_confidence_level','value')
            ])
def update_survivalgraph(sample1_dat_survival,sample2_dat_survival,
                        survival_confidence_level):
    #sample 1 input
    #if comma separated values
    if "," in sample1_dat_survival:
        p1 = np.asarray(sample1_dat_survival.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample1_dat_survival:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p1 = np.asarray(sample1_dat_survival.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample1_dat_survival = sample1_dat_survival.replace('\n', ',')
        p1 = np.asarray(sample1_dat_survival.split(','),dtype=np.float32)
    #sample 2 input
    #if comma separated values
    if "," in sample2_dat_survival:
        p2 = np.asarray(sample2_dat_survival.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample2_dat_survival:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p2 = np.asarray(sample2_dat_survival.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample2_dat_survival = sample2_dat_survival.replace('\n', ',')
        p2 = np.asarray(sample2_dat_survival.split(','),dtype=np.float32)

    y1 = []
    for i in np.sort(p1):
        x = 1 - np.compress((i > p1), p1).size/len(p1)
        y1.append(x)

    x1 = np.arange(0,len(p1),1).tolist()

    y2 = []
    for i in np.sort(p2):
        x = 1 - np.compress((i > p2), p2).size/len(p2)
        y2.append(x)

    x2 = np.arange(0,len(p2),1).tolist()

    trace1 = go.Scatter(
                x=x1,
                y=y1,
                mode = 'lines',
                name = 'Sample 1'
            )
    trace2 = go.Scatter(
                x=x2,
                y=y2,
                mode = 'lines',
                name = 'Sample 2'
            )

    return {
       
        'data':[trace1,trace2]
       
    }

#PAGE 5 - Count Data
##############################################################################
##############################################################################
#Text
markdown_text_poisson = '''
    A Poisson test is...
'''

page_5_layout = html.Div([
    html.H1('Count Data'),
    dcc.Markdown(children=markdown_text_poisson),
    dcc.Link('Go back to home', href='/'),
    html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Sample 1 # of Events
            html.Label('Sample 1: # of Events:'),
            dcc.Input(
                id='sample1_events',
                placeholder='20',
                type='text',
                value='20'
            ),
            # html.Div(id='sample1_successes_out'), #WIP

            # INPUT BOX 2 - Sample 1 # of Days
            html.Label('Sample 1: # of Days:'),
            dcc.Input(
                id='sample1_days',
                placeholder='1',
                type='text',
                value='1'
            ),

            # INPUT BOX 3 - Sample 2 # of Events
            html.Label('Sample 2: # of Events:'),
            dcc.Input(
                id='sample2_events',
                placeholder='25',
                type='text',
                value='25'
            ),

            # INPUT BOX 4 - Sample 2 # of Days
            html.Label('Sample 2: # of Days:'),
            dcc.Input(
                id='sample2_days',
                placeholder='1',
                type='text',
                value='1'
            ),

            # INPUT BOX 4 - Confidence Level
            html.Label('Confidence Level:'),
            dcc.Input(
                id='poisson_confidence_level',
                placeholder='0.95',
                type='text',
                value='0.95'
            ),
            
            html.Label('Poisson P-Value:'),
            html.Div(id='poisson_output'),
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    html.Div(
        [
            dcc.Graph(id='poisson-graph')

        ]),

    html.Br(),
    html.Br(),
    html.Br(),

],style={'columnCount': 2})

#RESULT- POISSON 
@app.callback(
    dash.dependencies.Output('poisson_output','children'),
            [dash.dependencies.Input('sample1_events','value'),
            dash.dependencies.Input('sample1_days','value'),
            dash.dependencies.Input('sample2_events','value'),
            dash.dependencies.Input('sample2_days','value')
            ])
def update_poisson(sample1_events,sample1_days,
                        sample2_events,sample2_days):
    
    p_result = binom_test(np.array([float(sample1_events)/float(sample1_days),
    float(sample2_events)/float(sample2_days)]), 
    float(sample1_events)+float(sample1_events))

    # return 'The CVaR is: "{}"'.format(CVaR)
    # return p_result
    return 'The p-value is: "{}"'.format(p_result)

#BAR GRAPH - POISSON
@app.callback(dash.dependencies.Output('poisson-graph','figure'),
            [dash.dependencies.Input('sample1_events','value'),
            dash.dependencies.Input('sample1_days','value'),
            dash.dependencies.Input('sample2_events','value'),
            dash.dependencies.Input('sample2_days','value'),
            dash.dependencies.Input('poisson_confidence_level','value')
            ])
def update_poissongraph(sample1_events,sample1_days,
                        sample2_events,sample2_days,
                        poisson_confidence_level):
    
    #calculate confidence intervals
    lower, upper = poisson.interval(float(poisson_confidence_level),
    [float(sample1_events),float(sample2_events)])

    err1 = (upper[0] - lower[0])/2
    err2 = (upper[1] - lower[1])/2

    return {
            'data': [
                {'x': [1,2], 
                'y': [sample1_events,sample2_events], 
                'error_y':{
                    "array":[err1,err2]},
                'type': 'bar', 'name': 'SF2'},
                ],
                'layout': {
                    'title': 'Poisson Confidence Interval'
                }
            }

#PAGE 6 - Value at Risk
##############################################################################
##############################################################################
#Text
markdown_text_chi_squared = '''
    Value at Risk is...
'''

page_6_layout = html.Div([
    html.H1('Credit Value at Risk'),
    dcc.Markdown(children=markdown_text_chi_squared),
    dcc.Link('Go back to home', href='/'),
    html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Asset Value at Time 0
            html.Label('Asset Value at Time 0:'),
            dcc.Input(
                id='s0',
                placeholder='1000',
                type='text',
                value='1000'
            ),

            # INPUT BOX 2 - Constant riskless short rate
            html.Label('Constant riskless short rate:'),
            dcc.Input(
                id='riskless_short_rate',
                placeholder='0.05',
                type='text',
                value='0.05'
            ),

            # INPUT BOX 3 - Constant volatility
            html.Label('Constant volatility'),
            dcc.Input(
                id='constant_volatility',
                placeholder='0.25',
                type='text',
                value='0.25'
            ),

            # INPUT BOX 4 - Random Draw Size
            html.Label('Number of Random Draws:'),
            dcc.Input(
                id='random_iter',
                placeholder='100000',
                type='text',
                value='100000'
            ),

            # INPUT BOX 5 - Number of Years
            html.Label('Time Horizon:'),
            dcc.Input(
                id='time_horizon',
                placeholder='1',
                type='text',
                value='1'
            ),

            # INPUT BOX 6 - Loss %
            html.Label('Loss %:'),
            dcc.Input(
                id='loss_pct',
                placeholder='0.05',
                type='text',
                value='0.05'
            ),

            # INPUT BOX 7 - Default Probability
            html.Label('Default Probability:'),
            dcc.Input(
                id='default_prob',
                placeholder='0.05',
                type='text',
                value='0.05'
            ),

            html.Label('Result VaR:'),
            html.Div(id='cvar_output'),
        ]),
        html.Br(),
    html.Div(
        [
            dcc.Graph(id='cvar-graph')

        ]),

    html.Br(),

])

#BAR GRAPH - CVaR
@app.callback(
    dash.dependencies.Output('cvar_output','children'),
            [dash.dependencies.Input('s0','value'),
            dash.dependencies.Input('riskless_short_rate','value'),
            dash.dependencies.Input('constant_volatility','value'),
            dash.dependencies.Input('random_iter','value'),
            dash.dependencies.Input('time_horizon','value'),
            dash.dependencies.Input('loss_pct','value'),
            dash.dependencies.Input('default_prob','value')
            ])
def update_cvartext(s0,riskless_short_rate,
                        constant_volatility,random_iter,
                        time_horizon,loss_pct,default_prob):
    s0 = float(s0)
    r = float(riskless_short_rate)
    sigma = float(constant_volatility)
    T = int(time_horizon)
    I = int(random_iter)
    #calculate 
    ST = s0 * np.exp((r-0.5*sigma**2)*T + sigma * np.sqrt(T) * npr.standard_normal(I))
    L = float(loss_pct)
    p = float(default_prob)
    #Poisson dfault scenarios
    D = npr.poisson(p*T,I)
    D = np.where(D>1,1,D)

    #calculate CVaR
    CVaR = np.exp(-r * T) * 1/I * np.sum(L*D*ST)

    return 'The CVaR is: "{}"'.format(CVaR)
    # return CVaR

#WIP
#HEATMAP- CVAR
@app.callback(
    dash.dependencies.Output('cvar-graph','figure'),
            [dash.dependencies.Input('s0','value'),
            dash.dependencies.Input('riskless_short_rate','value'),
            dash.dependencies.Input('constant_volatility','value'),
            dash.dependencies.Input('random_iter','value'),
            dash.dependencies.Input('time_horizon','value'),
            dash.dependencies.Input('loss_pct','value'),
            dash.dependencies.Input('default_prob','value')
            ])
def update_cvargraph(s0,riskless_short_rate,
                        constant_volatility,random_iter,
                        time_horizon,loss_pct,default_prob):
    s0 = float(s0)
    r = float(riskless_short_rate)
    sigma = float(constant_volatility)
    T = int(time_horizon)
    I = int(random_iter)
    #calculate 
    ST = s0 * np.exp((r-0.5*sigma**2)*T + sigma * np.sqrt(T) * npr.standard_normal(I))
    L = float(loss_pct)
    p = float(default_prob)
    #Poisson dfault scenarios
    D = npr.poisson(p*T,I)
    D = np.where(D>1,1,D)

    loss_rates = np.arange(0,1.05,0.05) #L
    default_rates = np.arange(0,1.05,0.05) #p
    cvar_df = pd.DataFrame(index=loss_rates,columns=default_rates)

    for col in cvar_df:
        for idx, row in cvar_df.iterrows():
            L = idx
            p = col
            #using Poisson distribution to create default scenarios
            D = npr.poisson(p*T,I)
            D = np.where(D>1,1,D)
            CVaR = np.exp(-r * T) * 1/I * np.sum(L*D*ST)
            cvar_df.set_value(idx,col,CVaR)

# WIP

    return {
        'data': [{
            'z': cvar_df.values.tolist(),
            'x': cvar_df.index.tolist(),
            'y': cvar_df.columns.tolist(),
             
            # 'text': cvar_df.values.tolist()
            # , 
            # 'customdata':cvar_df.values.tolist()
            # ,
            'type': 'heatmap', 'name': 'SF2'},
            ],
            'layout': {
                'title': 'CVaR Sensitivity',
                'xaxis' : {'title': 'Default Probability'},
                'yaxis' : {'title': 'Loss %'},
            }
        }

#PAGE 7 - PRICE ELASTICITY - VALUES
##############################################################################
##############################################################################
#Text
markdown_text_chi_squared = '''
    Price Elasticity of Demand is...
'''

page_7_layout = html.Div([
    html.H1('Price Elasticity of Demand - Values'),
    dcc.Markdown(children=markdown_text_chi_squared),
    dcc.Link('Go back to home', href='/'),
    html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Original Quantity
            html.Label('Original Quantity:'),
            dcc.Input(
                id='original_quantity',
                placeholder='2000',
                type='text',
                value='2000'
            ),

            # INPUT BOX 2 - New Quantity
            html.Label('New Quantity:'),
            dcc.Input(
                id='new_quantity',
                placeholder='4000',
                type='text',
                value='4000'
            ),

            # INPUT BOX 3 - Original Price
            html.Label('Original Price:'),
            dcc.Input(
                id='original_price',
                placeholder='1.50',
                type='text',
                value='1.50'
            ),

            # INPUT BOX 4 - New Price
            html.Label('New Price:'),
            dcc.Input(
                id='new_price',
                placeholder='1.00',
                type='text',
                value='1.00'
            ),


            html.Label('Result PE:'),
            html.Div(id='pe_output'),
        ]),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    html.Div(
        [
            # dcc.Graph(id='chisq-graph')

        ]),

    html.Br(),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])

#BAR GRAPH - PE
@app.callback(
    dash.dependencies.Output('pe_output','children'),
            [dash.dependencies.Input('original_quantity','value'),
            dash.dependencies.Input('new_quantity','value'),
            dash.dependencies.Input('original_price','value'),
            dash.dependencies.Input('new_price','value'),
            ])
def update_pe(original_quantity,new_quantity,
                        original_price,new_price):
    
    pe = ( (float(new_quantity)-float(original_quantity) ) / (float(new_quantity)+float(original_quantity) ) ) /\
     ( (float(new_price) - float(original_price ) ) / (float(new_price) + float(original_price)))

    # return 'The PE: "{}"'.format(pe)
    return 'An increase in price by  1% changes the quantity demanded by: "{}"%'.format(pe)


#PAGE 8 - PRICE ELASTICITY (RAW DATA)
##############################################################################
##############################################################################
#Text
markdown_text_chi_squared = '''
    Price Elasticity of Demand is...
'''

page_8_layout = html.Div([
    html.H1('Price Elasticity of Demand - Raw Data'),
    dcc.Markdown(children=markdown_text_chi_squared),
    dcc.Link('Go back to home', href='/'),
    html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Sample 1
            html.Label('Quantities:'),
            dcc.Textarea(
                id='pe_quantities',
                placeholder="Paste Samples Here",
                    value="14.300000190734901,15.1000003814697,15.300000190734901,15.199999809265099,15.800000190734901,14.699999809265099,16.799999237060501,17.600000381469702,19.299999237060501,19.799999237060501,19.200000762939499,20.600000381469702,20.600000381469702,21.100000381469702,21.299999237060501,22.899999618530302,24.5,25.100000381469698,25.200000762939499,26.299999237060501,27.399999618530302,27.399999618530302,28.299999237060501,27.100000381469698,27.0,26.399999618530302,28.5,29.0,30.399999618530302,32.799999237060497,32.700000762939503,33.700000762939503,33.900001525878899,34.0,35.299999237060497,36.400001525878899,37.200000762939503,39.400001525878899,39.599998474121101,40.900001525878899,42.400001525878899,44.099998474121101,46.5,48.200000762939503,48.799999237060497,48.200000762939503,48.799999237060497,49.5,49.799999237060497,52.900001525878899,53.200000762939503,53.900001525878899",
                style={'width': '20%',
                'height':'50%'},
            ),
            html.Br(),
            # INPUT BOX 2 - Sample 2
            html.Label('Prices:'),
            dcc.Textarea(
                id='pe_prices',
                placeholder='Paste Samples Here',
                value="2.8838200569152797,2.8038499355316202,2.7584900856018102,2.67040991783142,2.3940498828887899,2.5,2.1617600917816202,2.0391499996185298,1.95848000049591,1.7732000350952102,1.77026998996735,1.5852799415588399,1.6556299924850499,1.6111099720001201,1.5548399686813401,1.58095002174377,1.6265399456024199,1.4610799551010099,1.4712599515914899,1.47411000728607,1.35052001476288,1.3061699867248502,1.27751004695892,1.73649001121521,1.4665299654007,1.5130100250244098,1.3514900207519498,1.2755800485611,1.3128800392150901,1.2010999917983998,1.14563000202179,1.0616099834442101,0.98237997293472301,0.96687000989913907,1.0490900278091402,0.97118997573852506,1.0529199838638299,0.99735999107360795,1.0574799776077299,1.1056499481201201,1.0321300029754601,0.96696001291274991,0.940129995346069,0.95502001047134411,0.94533997774124101,0.93307000398635898,0.97259002923965499,0.98754000663757291,0.97913998365402188,0.97118997573852506,0.94599002599716198,0.94862002134323087",
                style={'width': '20%',
                'height':'50%'},
            ),

            html.Label('Result PE:'),
            html.Div(id='pe_data_output'),
        ]),
        html.Br(),
 
    html.Div(
        [
            dcc.Graph(id='pe_raw-graph')

        ]),

    html.Br(),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])

#BAR GRAPH - PE
@app.callback(
        dash.dependencies.Output('pe_data_output','children'),
            [dash.dependencies.Input('pe_prices','value'),
            dash.dependencies.Input('pe_quantities','value'),
            ])
def update_pe(pe_prices,pe_quantities):
     #sample 1 input
    #if comma separated values
    if "," in pe_prices:
        p1 = np.asarray(pe_prices.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in pe_prices:
        # pe_prices = pe_prices.replace('\n', ',')
        p1 = np.asarray(pe_prices.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        pe_prices = pe_prices.replace('\n', ',')
        p1 = np.asarray(pe_prices.split(','),dtype=np.float32)
    
    #sample 2 input
    #if comma separated values
    if "," in pe_quantities:
        p2 = np.asarray(pe_quantities.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in pe_quantities:
        # pe_quantities = pe_quantities.replace('\n', ',')
        p2 = np.asarray(pe_quantities.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        pe_quantities = pe_quantities.replace('\n', ',')
        p2 = np.asarray(pe_quantities.split(','),dtype=np.float32)

    # pd.DataFrame()

    est = sm.OLS(np.log(p2), sm.add_constant(np.log(p1))).fit()
    pe = est.params[1]

    return 'An increase in price by  1% changes the quantity demanded by: "{}"%'.format(pe)

#LINE GRAPH - PRICE ELASTICITY
@app.callback(dash.dependencies.Output('pe_raw-graph','figure'),
            [dash.dependencies.Input('pe_prices','value'),
            dash.dependencies.Input('pe_quantities','value'),
            ])
def update_survivalgraph(pe_prices,pe_quantities):
    #sample 1 input
    #if comma separated values
    if "," in pe_prices:
        p1 = np.asarray(pe_prices.split(','),dtype=np.float32)
        # p1 = list(pe_prices)
    # #if space separated values
    elif " " in pe_prices:
        # sample1_dat_ttest = sample1_dat_ttest.replace('\n', ',')
        p1 = np.asarray(pe_prices.split(' '),dtype=np.float32)
        # p1 = list(pe_prices)
    # #if line delimited values
    else:
        pe_prices = pe_prices.replace('\n', ',')
        # p1 = list(pe_prices)
        p1 = np.asarray(pe_prices.split(','),dtype=np.float32)
    #sample 2 input
    #if comma separated values
    if "," in pe_quantities:
        # p2 = list(pe_quantities)
        p2 = np.asarray(pe_quantities.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in pe_quantities:
        pe_quantities = pe_quantities.replace('\n', ',')
        # p2 = list(pe_quantities)
        p2 = np.asarray(pe_quantities.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        pe_quantities = pe_quantities.replace('\n', ',')
        # p2 = list(pe_quantities)
        p2 = np.asarray(pe_quantities.split(','),dtype=np.float32)

    df = pd.DataFrame(
    {'quantity': p2,
     'price': p1})

    df['demand'] = ""
    for idx,row in df.iterrows():
        x = np.sum(df[df['price']>=row['price']]['quantity'])
        df.set_value(idx,'demand',x)

    DEMAND = df['demand']
    # QUANT = df['quantity']
    PRICE = df['price']

    trace1 = go.Scatter(
                x=DEMAND,
                y=PRICE,
                mode = 'lines',
                name = 'Quantity Demanded'
            )

    return {
       
        'data':[trace1],
        'layout': {
            'title': 'Demand as a Function of Price',
            'xaxis' : {'title': 'Quantity Demand'},
            'yaxis' : {'title': 'Price'},
        }
       
    }


#PAGE 9 - PEARSON CORRELATION
##############################################################################
##############################################################################
#Text
markdown_text_pearson_correl= '''
    Pearson's Correlation...
'''

page_9_layout = html.Div([
    html.H1('Pearson Correlation'),
    dcc.Markdown(children=markdown_text_pearson_correl),
    dcc.Link('Go back to home', href='/'),
    html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Sample 1
            html.Label('Sample 1:'),
            dcc.Textarea(
                id='sample1_dat',
                placeholder="Paste Samples Here",
                    value="14.300000190734901,15.1000003814697,15.300000190734901,15.199999809265099,15.800000190734901,14.699999809265099,16.799999237060501,17.600000381469702,19.299999237060501,19.799999237060501,19.200000762939499,20.600000381469702,20.600000381469702,21.100000381469702,21.299999237060501,22.899999618530302,24.5,25.100000381469698,25.200000762939499,26.299999237060501,27.399999618530302,27.399999618530302,28.299999237060501,27.100000381469698,27.0,26.399999618530302,28.5,29.0,30.399999618530302,32.799999237060497,32.700000762939503,33.700000762939503,33.900001525878899,34.0,35.299999237060497,36.400001525878899,37.200000762939503,39.400001525878899,39.599998474121101,40.900001525878899,42.400001525878899,44.099998474121101,46.5,48.200000762939503,48.799999237060497,48.200000762939503,48.799999237060497,49.5,49.799999237060497,52.900001525878899,53.200000762939503,53.900001525878899",
                style={'width': '20%',
                'height':'50%'},
            ),
            html.Br(),
            # INPUT BOX 2 - Sample 2
            html.Label('Sample 2:'),
            dcc.Textarea(
                id='sample2_dat',
                placeholder='Paste Samples Here',
                value="2.8838200569152797,2.8038499355316202,2.7584900856018102,2.67040991783142,2.3940498828887899,2.5,2.1617600917816202,2.0391499996185298,1.95848000049591,1.7732000350952102,1.77026998996735,1.5852799415588399,1.6556299924850499,1.6111099720001201,1.5548399686813401,1.58095002174377,1.6265399456024199,1.4610799551010099,1.4712599515914899,1.47411000728607,1.35052001476288,1.3061699867248502,1.27751004695892,1.73649001121521,1.4665299654007,1.5130100250244098,1.3514900207519498,1.2755800485611,1.3128800392150901,1.2010999917983998,1.14563000202179,1.0616099834442101,0.98237997293472301,0.96687000989913907,1.0490900278091402,0.97118997573852506,1.0529199838638299,0.99735999107360795,1.0574799776077299,1.1056499481201201,1.0321300029754601,0.96696001291274991,0.940129995346069,0.95502001047134411,0.94533997774124101,0.93307000398635898,0.97259002923965499,0.98754000663757291,0.97913998365402188,0.97118997573852506,0.94599002599716198,0.94862002134323087",
                style={'width': '20%',
                'height':'50%'},
            ),

            html.Label('Result Correlation:'),
            html.Div(id='pearson_correl_output'),
        ]),
        html.Br(),
 
    html.Div(
        [
            dcc.Graph(id='pearson_correl-graph')

        ]),

    html.Br(),
    html.Br(),

])

#BAR GRAPH - PE
@app.callback(
        dash.dependencies.Output('pearson_correl_output','children'),
            [dash.dependencies.Input('sample1_dat','value'),
            dash.dependencies.Input('sample2_dat','value'),
            ])
def update_correl(sample1_dat,sample2_dat):
     #sample 1 input
    #if comma separated values
    if "," in sample1_dat:
        p1 = np.asarray(sample1_dat.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample1_dat:
        # pe_prices = pe_prices.replace('\n', ',')
        p1 = np.asarray(sample1_dat.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample1_dat = sample1_dat.replace('\n', ',')
        p1 = np.asarray(sample1_dat.split(','),dtype=np.float32)
    
    #sample 2 input
    #if comma separated values
    if "," in sample2_dat:
        p2 = np.asarray(sample2_dat.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample2_dat:
        # pe_quantities = pe_quantities.replace('\n', ',')
        p2 = np.asarray(sample2_dat.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample2_dat = sample2_dat.replace('\n', ',')
        p2 = np.asarray(sample2_dat.split(','),dtype=np.float32)

    pd.DataFrame()
    df = pd.DataFrame(
    {'sample1': p2,
     'sample2': p1})

    corr = df.corr(method="pearson").iloc[0,1]

    return 'The correlation coefficient is: "{}"'.format(corr)

#LINE GRAPH - PRICE ELASTICITY
@app.callback(dash.dependencies.Output('pearson_correl-graph','figure'),
            [dash.dependencies.Input('sample1_dat','value'),
            dash.dependencies.Input('sample2_dat','value'),
            ])
def update_correlgraph(sample1_dat,sample2_dat):
    #sample 1 input
    #if comma separated values
    if "," in sample1_dat:
        p1 = np.asarray(sample1_dat.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample1_dat:
        # pe_prices = pe_prices.replace('\n', ',')
        p1 = np.asarray(sample1_dat.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample1_dat = sample1_dat.replace('\n', ',')
        p1 = np.asarray(sample1_dat.split(','),dtype=np.float32)
    
    #sample 2 input
    #if comma separated values
    if "," in sample2_dat:
        p2 = np.asarray(sample2_dat.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in sample2_dat:
        # pe_quantities = pe_quantities.replace('\n', ',')
        p2 = np.asarray(sample2_dat.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        sample2_dat = sample2_dat.replace('\n', ',')
        p2 = np.asarray(sample2_dat.split(','),dtype=np.float32)

    df = pd.DataFrame(
    {'sample1': p2,
     'sample2': p1})

    trace1 = go.Scatter(
                x=df['sample1'],
                y=df['sample2'],
                mode = 'markers',
                name = 'Scatter Plot'
            )

    return {
       
        'data':[trace1],
        'layout': {
            'title': 'Scatter Plot',
            'xaxis' : {'title': 'Sample 2'},
            'yaxis' : {'title': 'Sample 1'},
        }
    }

# INDEX
#############################################################################
#############################################################################

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/sample_size_calculator':
        return page_1_layout
    elif pathname == '/chi_squared_test':
        return page_2_layout
    elif pathname == '/2_sample_ttest':
        return page_3_layout
    elif pathname == '/2_sample_survival':
        return page_4_layout
    elif pathname == '/count_data':
        return page_5_layout
    elif pathname == '/cvar':
        return page_6_layout
    elif pathname == '/pe':
        return page_7_layout
    elif pathname == '/pe_raw':
        return page_8_layout
    elif pathname == '/pearson_correl':
        return page_9_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)