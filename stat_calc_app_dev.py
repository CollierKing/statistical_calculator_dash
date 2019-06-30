import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
# from scipy.stats import norm
from sklearn.neighbors import KernelDensity
#additional:
import numpy as np,pandas as pd
import numpy.random as npr
## sample size
from scipy.stats import zscore
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
# optimization
from scipy.optimize import curve_fit
import scipy.optimize as optimize
# survival
from lifelines.statistics import logrank_test
from scipy.stats import gamma
# price elasticity
import statsmodels.api as sm
# r-squared
# from sklearn.metrics import pearson_correl_output
from sklearn.metrics import r2_score

print(dcc.__version__) # 0.6.0 or above is required

app = dash.Dash(__name__)


# TO-DO LIST
#####################################
# Solving for Optimal Price from Demand Function (raw values)


# Since we're adding callbacks to elements that don't exist in the app.layout,
# Dash will raise an exception to warn us that we might be
# doing something wrong.
# In this case, we're adding the elements through a callback, so we can ignore
# the exception.
app.config.supress_callback_exceptions = True

# UP-FRONT FUNCTION DEFINITIONS
## Text Parser
def text_parse(blob1,blob2):
    '''
    Takes a blob of values input into 'TextArea' box widget and returns a numpy array
    Text blobs can be comma, space or line delimited
    '''
    #blob 1 input
    ##if comma separated values
    if "," in blob1:
            p1 = np.asarray(blob1.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in blob1:
        p1 = np.asarray(blob1.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        blob1 = blob1.replace('\n', ',')
        p1 = np.asarray(blob1.split(','),dtype=np.float32)
    #blob 2 input
    #if comma separated values
    if "," in blob2:
        p2 = np.asarray(blob2.split(','),dtype=np.float32)
    # #if space separated values
    elif " " in blob2:
        p2 = np.asarray(blob2.split(' '),dtype=np.float32)
    # #if line delimited values
    else:
        blob2 = blob2.replace('\n', ',')
        p2 = np.asarray(blob2.split(','),dtype=np.float32)
    return p1,p2

# HOME
##############################################################################
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Br(),
])

markdown_intro = '''
    This is a collection of calculators for solving certain statistical, financial and economic questions.  
    The site is built on Ploty's Dash framework for Python.
    '''


index_page = html.Div([
    html.H1('Research Calculators'),

    html.P(markdown_intro, className='blue-text'),
    # html.Br(),
    html.A("Github Repository", href='https://github.com/CollierKing/statistical_calculator_dash', target="_blank"),
    # dcc.Link('Github Repository',href='https://dash.plot.ly/?_ga=2.58527913.24526322.1561223044-278868159.1561223044'),
    html.Br(),
    # dcc.Link('Dash Documentation',href='https://dash.plot.ly/?_ga=2.58527913.24526322.1561223044-278868159.1561223044'),
    html.A("Dash Documentation", href='https://dash.plot.ly/?_ga=2.58527913.24526322.1561223044-278868159.1561223044', target="_blank"),
    html.Br(),

    html.H2('Statistical:'),
    html.P(
    'This section has tools related to A/B testing and data comparison across groups.', className='blue-text'),
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

    html.H2('Economic:'),
    html.P(
    'This section provides calculations related to economics such as pricing.', className='blue-text'),
    # dcc.Link('Price Elasticity of Demand - Values',href='/pe'),
    # html.Br(),
    dcc.Link('Price Elasticity of Demand',href='/pe_raw'),
    html.Br(),
    # dcc.Link('Price Optimization',href='/price_optimization'),
    # html.Br(),

    html.H2('Financial:'),
    html.P(
    'This section provides calculations related to finance.', className='blue-text'),
    dcc.Link('Credit Value at Risk',href='/cvar'),
    html.Br(),
    dcc.Link('Miscellaneous Finance',href='/financial_misc'),
    html.Br(),


])

# PAGE 1 - SAMPLE SIZE CALCULATOR
##############################################################################
page_1_layout = html.Div([
    html.Div([
        html.H2('Sample Size Calculator'),
        dcc.Link('Go back to home', href='/'),
        html.Div([
        html.H4('Description:'),
        html.P('When planning an A/B test, it is important \
                to calculate how many subjects are needed \
                to determine a given change \
                in the conversion or baseline rate... \
                This computation requires a pre-defined \
                significance level and statistical power.', className='blue-text'),
        dcc.Link('Reading Link: Statistical Power',href='https://en.wikipedia.org/wiki/Statistical_power'),
        ],style={'width': "80%"}),
        html.H4('Parameters:'),
        html.Div([
            html.Strong('Baseline Rate'),
            html.Br(),
            html.P("The underlying success rate you are trying to measure against."),
            html.Strong('Effect Size'),
            html.Br(),
            html.P("The new rate which includes the change to measure."),
            # html.Br(),
            html.Strong('Power (1-FNR)'),
            html.P("The likelihood that a detectable effect will actually be detected by an analysis."),
            html.Strong('Significance Level'),
            html.Br(),
            html.P("The probability of rejecting the null hypothesis, given that the null hypothesis is true."),
        ],style={'columnCount': 1,'width': "80%"}),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        html.H4('Inputs:'),
        html.Div([
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
            html.Label('Power'),
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
            ],style={'columnCount': 2,'width': "80%"}),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        html.H4('Result:'),
        html.Div(id='sample_size_content',style={'color':'blue'}),
        html.Div([
            dcc.Graph(id='example-graph'),
            html.P(
                'Conversion rates within the error bar range \
                will be indistinguishable from the baseline rate.', className='blue-text')
            ],style={'columnCount': 1,'width': "100%",'height': "50%"})
    ]),
],style={'columnCount': 2,'width': "100%"})

#BAR GRAPH - SAMPLE SIZE
@app.callback(dash.dependencies.Output('example-graph','figure'),
            [dash.dependencies.Input('baseline_input','value'),
            dash.dependencies.Input('effect_size_input','value')
            ])
def update_graph(baseline_input,effect_size_input):
    err_bar = float(effect_size_input)-float(baseline_input)
    return {
            'data': [
                {'x': [baseline_input],
                'y': ['Conversion Rate'],
                'error_x':{
                    "array":[err_bar]},
                'type': 'bar', 'name': 'SF',
                'orientation':'h'},
                ],
                'layout': {
                    'title': 'Baseline with Effect',
                    'sub title': 'Plot Subtitle'
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

    return 'Required Sample Size: "{}"'.format(n)


#PAGE 2 - CHI-SQUARED TESTS
##############################################################################
##############################################################################
#Text

page_2_layout = html.Div([
    html.H2('Chi Squared Test'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.H4('Description:'),
    html.P(
        'In an A/B test, a Chi Squared Test can be used to compare success rates \
        across different samples.'
    ),
    #PARAMETERS
    html.H4("Parameters:"),
    html.Div(
    [
        html.Strong("Sample 1 Successes"),
        html.P("The number of trials that succeeded in sample 1."),
        # html.Br(),
        html.Strong("Sample 1 Trials"),
        html.P("The number of trials that where conducted in sample 1."),
        # html.Br(),
        html.Strong("Sample 2 Successes"),
        html.P("The number of trials that succeeded in sample 2."),
        # html.Br(),
        html.Strong("Sample 2 Trials"),
        html.P("The number of trials that where conducted in sample 2."),
        # html.Br(),
    ],style={'columnCount': 2,'width': "70%"}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H3("Inputs:"),
    #INPUTS
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
        html.Br(),
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
        ],style={'columnCount': 2,'width': "70%"}),
        # html.Div([
        html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        # html.Br(),
        html.H3('Result:'),
        html.Div(id='chisq_content',style={'color':'blue'}),
        dcc.Graph(id='chisq-graph')

        # ],style={'columnCount': 1,'width': "100%",'height': "100%"}),
    ],style={'columnCount': 2,'width':'100%'})

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
                {'y': ['sample1','sample2'],
                'x': [sample1_prop,sample2_prop],
                'error_x':{
                    "array":[err1,err2]},
                'type': 'bar',
                'name': 'SF2',
                'orientation':'h'},
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
    return 'p-value: "{}"'.format(p)


#PAGE 3 - 2 SAMPLE T-TESTS
##############################################################################
##############################################################################
page_3_layout = html.Div([

    html.H2('Two Sample T-Test'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.H4('Description:'),
    html.P(
        'A T-Test can be used to determine if the means of two groups are unequal with statistical significance.'
    ),
    #PARAMETERS
    html.H4("Inputs:"),
    html.P("Copy and paste the values for each sample below...\
    Values can be space, comma, or line delimited."),
    html.Div([
        html.Label('Sample 1 Values:'),
        dcc.Textarea(
            id='sample1_dat_ttest',
            placeholder="Paste Samples Here",
                value="64.2,28.4,85.3,83.1,13.4,56.8,44.2,90",
            style={'width': '100%',
            'height':'500%'},
        ),
        html.Br(),
        html.Label('Sample 2 Values:'),
        dcc.Textarea(
            id='sample2_dat_ttest',
            placeholder='Paste Samples Here',
            value="45,29.5,32.3,49.3,18.3,34.2,43.9,13.8,27.4,43.4",
            style={'width': '100%',
            'height':'500%'}),

    # ],style={'columnCount': 2}),
    ],style={'columnCount': 2,'width': "80%"}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4('Result:'),
    html.Div(id='sample1_dat_summ'),
    html.Div(id='ttest_result',style={'color':'blue'}),
    html.Div([
            dcc.Graph(id='ttest-graph')
        ]),
    html.Br(),
    # dcc.Link('Go to Page 1', href='/sample_size_calculator'),
    html.Br(),

    ],style={'columnCount': 2,'width': "60%"})

#TTEST - RESULT
@app.callback(dash.dependencies.Output('ttest_result', 'children'),
              [dash.dependencies.Input('sample1_dat_ttest', 'value'),
              dash.dependencies.Input('sample2_dat_ttest', 'value')
              ])
def ttest_result(sample1_dat_ttest,sample2_dat_ttest):
    p1, p2 = text_parse(sample1_dat_ttest,sample2_dat_ttest)
    t, p = ttest_ind(p1, p2, equal_var=False)

    return 'p-value: "{}"'.format(p)


@app.callback(dash.dependencies.Output('ttest-graph', 'figure'),
              [dash.dependencies.Input('sample1_dat_ttest', 'value'),
              dash.dependencies.Input('sample2_dat_ttest', 'value')
              ])
def ttest_plot(sample1_dat_ttest,sample2_dat_ttest):
    p1, p2 = text_parse(sample1_dat_ttest,sample2_dat_ttest)

    p1_mean = np.mean(p1)
    p1_std = np.std(p1)

    p2_mean = np.mean(p2)
    p2_std = np.std(p2)

    s1 = np.random.normal(p1_mean, p1_std, 1000)
    s2 = np.random.normal(p2_mean, p2_std, 1000)

    import matplotlib.pyplot as plt
    count1, bins1, ignored1 = plt.hist(s1, 30, normed=True)
    count2, bins2, ignored2 = plt.hist(s2, 30, normed=True)

    lines1 = 1/(p1_std * np.sqrt(2 * np.pi)) * np.exp( - (bins1 - p1_mean)**2 / (2 * p1_std**2))
    lines2 = 1/(p2_std * np.sqrt(2 * np.pi)) * np.exp( - (bins2 - p2_mean)**2 / (2 * p2_std**2))

    trace1 = go.Scatter(
            x=bins1,
            y=lines1,
            mode = 'lines',
            name = 'Sample 1'
        )

    trace2 = go.Scatter(
            x=bins2,
            y=lines2,
            mode = 'lines',
            name = 'Sample 2'
        )

    return {
        'data':[trace1,trace2],
        'layout': {
                'title': 'Density Plots',
                'xaxis' : {'title': ''},
                'yaxis' : {'title': ''},
            }

    }

#PAGE 4 - 2 SAMPLE SURVIVAL TIMES
##############################################################################
##############################################################################
page_4_layout = html.Div([

    html.H2('Survival Times - Curves'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    dcc.Link('Compare means', href='/2_sample_survival_means'),
    html.Br(),
    html.H4('Description:'),
    html.P(
        'A Survival Analysis lets us analyze the expected duration of time until one or more events happen.'
    ),
    #PARAMETERS
    html.H4("Parameters:"),
    html.Div([
        # INPUT BOX 1 - Sample 1
        html.Label('Sample 1:'),
        dcc.Textarea(
            id='sample1_dat_survival',
            placeholder="Paste Samples Here",
                value="1,1,2,3,4,4,5,5,8,8,8,8,11,11,12,12,15,17,22,23",
            style={'width': '100%',
            'height':'50%'},
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        # INPUT BOX 2 - Sample 2
        html.Label('Sample 2:'),
        dcc.Textarea(
            id='sample2_dat_survival',
            placeholder='Paste Samples Here',
            value='6,6,7,9,10,11,13,15,16,19,20,22,23,32,6,10,17,19,24,25,25,28,28,32,33,34,35,39',
            style={'width': '100%'}),
        html.Br(),
        html.Br(),
    # INPUT BOX 3 - Confidence Level
    html.Label('Confidence Level:'),
    dcc.Input(
        id='survival_confidence_level',
        placeholder='0.95',
        type='text',
        value='0.95'
    ),
    ],style={'columnCount': 2,'width': "55%"}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H3('Result:'),
    html.Div(id='survival_output',style={'color':'blue'}),
    html.Div([
        dcc.Graph(id='survival-graph')
    ]),
    ],style={'columnCount': 2,'width': "80%"})

#RESULT- SURVIVAL
@app.callback(
    dash.dependencies.Output('survival_output','children'),
            [dash.dependencies.Input('sample1_dat_survival','value'),
            dash.dependencies.Input('sample2_dat_survival','value'),
            dash.dependencies.Input('survival_confidence_level','value')
            ])
def update_survival(sample1_dat_survival,sample2_dat_survival,
                    survival_confidence_level):

    p1, p2 = text_parse(sample1_dat_survival,sample2_dat_survival)

    x = logrank_test(p1, p2, alpha=float(survival_confidence_level))

    p_result = float(x.p_value)

    return 'p-value: "{}"'.format(p_result)


#LINE GRAPH - SURVIVAL
@app.callback(dash.dependencies.Output('survival-graph','figure'),
            [dash.dependencies.Input('sample1_dat_survival','value'),
            dash.dependencies.Input('sample2_dat_survival','value'),
            dash.dependencies.Input('survival_confidence_level','value')
            ])
def update_survivalgraph(sample1_dat_survival,sample2_dat_survival,
                        survival_confidence_level):

    p1, p2 = text_parse(sample1_dat_survival,sample2_dat_survival)

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
        'data':[trace1,trace2],
        'layout': {
                'title': 'Survival Function Plot',
                'xaxis' : {'title': 'Time'},
                'yaxis' : {'title': 'Survival Rate'},
            }

    }

#PAGE 10 - 2 SAMPLE SURVIVAL TIME MEANS
##############################################################################
##############################################################################
page_11_layout = html.Div([

    html.H2('Survival Times'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.H4('Description:'),
    html.P(
        'A Survival Analysis lets us analyze the expected duration of time until one or more events happen.'
    ),
    #PARAMETERS
    html.H4("Parameters:"),
    html.Div([
        # INPUT BOX 1 - Sample 1
        html.Label('Sample 1 Size:'),
        dcc.Input(
            id='sample1_dat_survival_size',
            placeholder='100',
            type='text',
            value='100',
            style={'width': '70%'}
        ),
        html.Br(),
        # INPUT BOX 2 - Sample 2
        html.Label('Sample 1 Mean:'),
        dcc.Input(
            id='sample1_dat_survival_mean',
            placeholder='55',
            type='text',
            value='55',
            style={'width': '70%'}
        ),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        # INPUT BOX 1 - Sample 1
        html.Label('Sample 2 Size:'),
        dcc.Input(
            id='sample2_dat_survival_size',
            placeholder='120',
            type='text',
            value='120',
            style={'width': '70%'}
        ),
        html.Br(),
        # INPUT BOX 2 - Sample 2
        html.Label('Sample 2 Mean:'),
        dcc.Input(
            id='sample2_dat_survival_mean',
            placeholder='45',
            type='text',
            value='45',
            style={'width': '70%'}
        ),
    # ],style={'columnCount': 2}),
    html.Br(),
    # INPUT BOX 3 - Confidence Level
    html.Label('Confidence Level:'),
    dcc.Input(
        id='survival_confidence_level_means',
        placeholder='0.95',
        type='text',
        value='0.95',
        style={'width': '70%'}
    ),
    ],style={'columnCount': 2,'width': "55%"}),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4("Results:"),
    html.Label('Log Rank P-Value:'),
    html.Div(id='survival_output_means'),
        html.Div([
            dcc.Graph(id='survival-graph_means')
        ]),
    html.Br(),
# ])
],style={'columnCount': 2,'width': "80%"})

#RESULT- SURVIVAL
@app.callback(
    dash.dependencies.Output('survival_output_means','children'),
            [dash.dependencies.Input('sample1_dat_survival_mean','value'),
            dash.dependencies.Input('sample1_dat_survival_size','value'),
            dash.dependencies.Input('sample2_dat_survival_mean','value'),
            dash.dependencies.Input('sample2_dat_survival_size','value'),
            dash.dependencies.Input('survival_confidence_level_means','value')
            ])
def update_survival(sample1_dat_survival_mean,sample1_dat_survival_size,
                    sample2_dat_survival_mean,sample2_dat_survival_size,
                    survival_confidence_level_means):

    if float(sample1_dat_survival_mean) > float(sample2_dat_survival_mean):
        f1 = float(sample1_dat_survival_mean) / float(sample2_dat_survival_mean)
        df1 = 2 * float(sample1_dat_survival_size)
        df2 = 2 * float(sample2_dat_survival_size)
    else:
        f1 = float(sample2_dat_survival_mean) / float(sample1_dat_survival_mean)
        df1 = 2 * float(sample2_dat_survival_size)
        df2 = 2 * float(sample1_dat_survival_size)

    p_value = 2 * (1.0 - beta.cdf((df1 * f1) / (df1 * f1 + df2), df1 / 2, df2 / 2))

    return p_value

    p_result = float(x.p_value)

    return 'The p-value is: "{}"'.format(p_result)


#BAR GRAPH - SURVIVAL MEANS
@app.callback(dash.dependencies.Output('survival-graph_means','figure'),
            [dash.dependencies.Input('sample1_dat_survival_mean','value'),
            dash.dependencies.Input('sample1_dat_survival_size','value'),
            dash.dependencies.Input('sample2_dat_survival_mean','value'),
            dash.dependencies.Input('sample2_dat_survival_size','value'),
            dash.dependencies.Input('survival_confidence_level_means','value')
            ])
def update_survivalgraph(sample1_dat_survival_mean,sample1_dat_survival_size,
                        sample2_dat_survival_mean,sample2_dat_survival_size,
                        survival_confidence_level_means):

    alpha = float(1 - float(survival_confidence_level_means))

    n1 = float(sample1_dat_survival_size)
    t1 = float(sample1_dat_survival_mean)

    n2 = float(sample2_dat_survival_size)
    t2 = float(sample2_dat_survival_mean)

    result1 = ( n1 * t1 / gamma.ppf(1-alpha/2, n1), n1 * t1 / gamma.ppf(alpha/2, n1))

    result2 = ( n2 * t2 / gamma.ppf(1-alpha/2, n2), n2 * t2 / gamma.ppf(alpha/2, n2))

    s_err1 = (result1[1] - result1[0])
    s_err2 = (result2[1] - result2[0])

    return {
        'data': [
            {'y': ['Sample1','Sample2'],
            'x': [sample1_dat_survival_mean,sample2_dat_survival_mean],
            'error_x':{
                "array":[s_err1,s_err2]},
            'type': 'bar',
            'name': 'SF3',
            'orientation':'h'},
            ],
            'layout': {
                'title': 'Survival Times',
            }
        }


#LINE GRAPH - SURVIVAL
# @app.callback(dash.dependencies.Output('survival-graph_means','figure'),
#             [dash.dependencies.Input('sample1_dat_survival_means','value'),
#             dash.dependencies.Input('sample2_dat_survival_means','value'),
#             dash.dependencies.Input('survival_confidence_level_means','value')
#             ])
# def update_survivalgraph(sample1_dat_survival_means,sample2_dat_survival_means,
#                         survival_confidence_level_means):

#     p1, p2 = text_parse(sample1_dat_survival_means,sample2_dat_survival_means)

#     y1 = []
#     for i in np.sort(p1):
#         x = 1 - np.compress((i > p1), p1).size/len(p1)
#         y1.append(x)

#     x1 = np.arange(0,len(p1),1).tolist()

#     y2 = []
#     for i in np.sort(p2):
#         x = 1 - np.compress((i > p2), p2).size/len(p2)
#         y2.append(x)

#     x2 = np.arange(0,len(p2),1).tolist()

#     trace1 = go.Scatter(
#                 x=x1,
#                 y=y1,
#                 mode = 'lines',
#                 name = 'Sample 1'
#             )
#     trace2 = go.Scatter(
#                 x=x2,
#                 y=y2,
#                 mode = 'lines',
#                 name = 'Sample 2'
#             )

#     return {
#         'data':[trace1,trace2],
#         'layout': {
#                 'title': 'Survival Function Plot',
#                 'xaxis' : {'title': 'Time'},
#                 'yaxis' : {'title': 'Survival Rate'},
#             }

#     }

#PAGE 5 - Count Data
##############################################################################
##############################################################################

page_5_layout = html.Div([
    html.H2('Count Data'),
    # dcc.Markdown(children=markdown_text_poisson),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.H4('Description:'),
    html.P(
        'A Poisson test is a way to determine if there is a statistically significant difference between the rates of two samples of count data.'
    ), 
    #PARAMETERS
    html.H4("Parameters:"),

        html.Div(
        [
            # INPUT BOX 1 - Sample 1 # of Events
            html.Label('Sample 1: # of Events:'),
            dcc.Input(
                id='sample1_events',
                placeholder='20',
                type='text',
                value='20',
                style={'width': '100%',
            'height':'50%'}
            ),
            # html.Div(id='sample1_successes_out'), #WIP

            # INPUT BOX 2 - Sample 1 # of Days
            html.Label('Sample 1: # of Days:'),
            dcc.Input(
                id='sample1_days',
                placeholder='1',
                type='text',
                value='1',
                style={'width': '100%',
            'height':'50%'}
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            # INPUT BOX 3 - Sample 2 # of Events
            html.Label('Sample 2: # of Events:'),
            dcc.Input(
                id='sample2_events',
                placeholder='25',
                type='text',
                value='25',
                style={'width': '100%',
            'height':'50%'}
            ),

            # INPUT BOX 4 - Sample 2 # of Days
            html.Label('Sample 2: # of Days:'),
            dcc.Input(
                id='sample2_days',
                placeholder='1',
                type='text',
                value='1',
                style={'width': '100%',
            'height':'50%'}
            ),
            # ],style={'columnCount': 2}),
            # html.Div([
            # INPUT BOX 4 - Confidence Level
            html.Label('Confidence Level:'),
            dcc.Input(
                id='poisson_confidence_level',
                placeholder='0.95',
                type='text',
                value='0.95',
                style={'width': '100%',
            'height':'50%'}
            ),
            ],style={'columnCount': 2,'width': "55%"}),
            html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H4('Result:'),
    html.Div(id='poisson_output'),
        # ]),
    html.Div(
        [
            dcc.Graph(id='poisson-graph')

        ]),

    html.Br(),
    html.Br(),
    html.Br(),
],style={'columnCount': 2,'width': "80%"})

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
                {'y': [1,2],
                'x': [sample1_events,sample2_events],
                'error_x':{
                    "array":[err1,err2]},
                'type': 'bar', 'name': 'SF2',
                'orientation':'h'},
                ],
                'layout': {
                    'title': 'Poisson Confidence Interval'
                }
            }

#PAGE 6 - Value at Risk
##############################################################################
##############################################################################
#Text
# markdown_text_chi_squared = '''
#     Value at Risk is...
# ''' 

page_6_layout = html.Div([
    html.H2('Credit Value at Risk'),
    dcc.Link('Go back to home', href='/'),
    html.H4('Description:'),
    html.P(
        'Credit Value at Risk is the largest likely loss expected due to a counterparty default over a given period of time with a given confidence interval.'
    ),
    # html.Br(),
    html.P('For example, a company might estimate its CVAR over 10 days to be $100 million with a confidence interval of 95%. This means there is a one-in-20 (5%) chance of a loss larger than $100 million in the next 10 days.'),
    #PARAMETERS
    html.H4("Parameters:"),
    # html.Br(),
    html.Div(
        [
        html.Div(
        [
            # INPUT BOX 1 - Asset Value at Time 0
            html.Label('Asset Value at Time 0:'),
            dcc.Input(
                id='s0',
                placeholder='1000',
                type='text',
                value='1000',
                style={'width': '80%',
                'height':'50%'}
            ),

            # INPUT BOX 2 - Constant riskless short rate
            html.Label('Const. riskless short rate:'),
            dcc.Input(
                id='riskless_short_rate',
                placeholder='0.05',
                type='text',
                value='0.05',
                style={'width': '80%',
                'height':'50%'}
            ),

            # INPUT BOX 3 - Constant volatility
            html.Label('Constant volatility'),
            dcc.Input(
                id='constant_volatility',
                placeholder='0.25',
                type='text',
                value='0.25',
                style={'width': '80%',
                'height':'50%'}
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            # INPUT BOX 4 - Random Draw Size
            html.Label('# Simulations:'),
            dcc.Input(
                id='random_iter',
                placeholder='10000',
                type='text',
                value='10000',
                style={'width': '80%',
                'height':'50%'}
            ),

            # INPUT BOX 5 - Number of Years
            html.Label('Time Horizon:'),
            dcc.Input(
                id='time_horizon',
                placeholder='1',
                type='text',
                value='1',
                style={'width': '80%',
                'height':'50%'}
            ),

            # INPUT BOX 6 - Loss %
            html.Label('Loss Probability:'),
            dcc.Input(
                id='loss_pct',
                placeholder='0.05',
                type='text',
                value='0.05',
                style={'width': '80%',
                'height':'50%'}
            ),

            # INPUT BOX 7 - Default Probability
            html.Label('Default Probability:'),
            dcc.Input(
                id='default_prob',
                placeholder='0.05',
                type='text',
                value='0.05',
                style={'width': '80%',
                'height':'50%'}
            ),
            # ],style={'columnCount': 2}),
            ],style={'columnCount': 2,'width': "60%"}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
            html.H4("Results:"),
            html.Div(id='cvar_output'),
        ]),
        html.Div(
        [
            dcc.Graph(id='cvar-graph')
        ]),
    html.Br(),

# ])
],style={'columnCount': 2,'width': "70%"}),
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

# #PAGE 7 - PRICE ELASTICITY - VALUES
# ##############################################################################
# ##############################################################################

# page_7_layout = html.Div([
#     html.H1('Price Elasticity of Demand - Values'),
#     # dcc.Markdown(children=markdown_text_chi_squared),
#     dcc.Link('Go back to home', href='/'),
#     # html.Br(),
#     html.H3('Description:'),
#     html.P(
#         'Price Elasticity of Demand is defined as the responsiveness of the quanitity demanded to a change in its price, all else held constant.'
#     ), #todo: finish
#     #PARAMETERS
#     html.H4("Parameters:"),
#         html.Div(
#         [
#             # INPUT BOX 1 - Original Quantity
#             html.Label('Original Quantity:'),
#             dcc.Input(
#                 id='original_quantity',
#                 placeholder='2000',
#                 type='text',
#                 value='2000',
#                 style={'width': '100%',
#             'height':'50%'}
#             ),

#             # INPUT BOX 2 - New Quantity
#             html.Label('New Quantity:'),
#             dcc.Input(
#                 id='new_quantity',
#                 placeholder='4000',
#                 type='text',
#                 value='4000',
#                 style={'width': '100%',
#             'height':'50%'}
#             ),

#             # INPUT BOX 3 - Original Price
#             html.Label('Original Price:'),
#             dcc.Input(
#                 id='original_price',
#                 placeholder='1.50',
#                 type='text',
#                 value='1.50',
#                 style={'width': '100%',
#             'height':'50%'}
#             ),

#             # INPUT BOX 4 - New Price
#             html.Label('New Price:'),
#             dcc.Input(
#                 id='new_price',
#                 placeholder='1.00',
#                 type='text',
#                 value='1.00',
#                 style={'width': '100%',
#             'height':'50%'}
#             ),
#         # ]),
#         ],style={'columnCount': 2,'width': "55%"}),
#             html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.Br(),
#         html.H3('Result:'),
#         html.Label('Result PE:'),
#         html.Div(id='pe_output'),
#     html.Div(
#         [
#             dcc.Graph(id='pe-graph')

#         ]),

#     html.Br(),
#     html.Br(),
# ],style={'columnCount': 2,'width': "80%"})
# # ])

# #BAR GRAPH - PE
# @app.callback(
#     dash.dependencies.Output('pe_output','children'),
#             [dash.dependencies.Input('original_quantity','value'),
#             dash.dependencies.Input('new_quantity','value'),
#             dash.dependencies.Input('original_price','value'),
#             dash.dependencies.Input('new_price','value'),
#             ])
# def update_pe(original_quantity,new_quantity,
#                         original_price,new_price):

#     pe = ( (float(new_quantity)-float(original_quantity) ) / (float(new_quantity)+float(original_quantity) ) ) /\
#      ( (float(new_price) - float(original_price ) ) / (float(new_price) + float(original_price)))

#     # return 'The PE: "{}"'.format(pe)
#     return 'An increase in price by  1% changes the quantity demanded by: "{}"%'.format(pe)


# #BAR GRAPH - PE (CHART)
# @app.callback(
#     dash.dependencies.Output('pe-graph','figure'),
#             [dash.dependencies.Input('original_quantity','value'),
#             dash.dependencies.Input('new_quantity','value'),
#             dash.dependencies.Input('original_price','value'),
#             dash.dependencies.Input('new_price','value'),
#             ])
# def update_pe_chart(original_quantity,new_quantity,
#                         original_price,new_price):

#     # pe = ( (float(new_quantity)-float(original_quantity) ) / (float(new_quantity)+float(original_quantity) ) ) /\
#     #  ( (float(new_price) - float(original_price ) ) / (float(new_price) + float(original_price)))

#     trace1 = go.Scatter(
#             x=[original_quantity,new_quantity],
#             y=[original_price,new_price],
#             mode = 'lines',
#             name = 'Quantity Demanded'
#         )

#     return {

#         'data':[trace1],
#         'layout': {
#             'title': 'Demand as a Function of Price',
#             'xaxis' : {'title': 'Quantity Demand'},
#             'yaxis' : {'title': 'Price'},
#         }

#     }


#PAGE 8 - PRICE ELASTICITY (RAW DATA)
##############################################################################
##############################################################################

page_8_layout = html.Div([
    html.H2('Price Elasticity of Demand'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.H4('Description:'),
    html.P(
        'Price Elasticity of Demand is defined as the responsiveness of the quanitity demanded to a change in its price, all else held constant.'
    ), #todo: finish
    html.H4("Parameters:"),
        html.Div(
        [
            # INPUT BOX 1 - Sample 1
            html.Label('Quantities:'),
            dcc.Textarea(
                id='pe_quantities',
                placeholder="Paste Samples Here",
                    value="3, 8, 1, 5, 4, 7, 10, 9, 6, 2",
                style={'width': '100%',
                'height':'50%'},
            ),
            html.Br(),
            # INPUT BOX 2 - Sample 2
            html.Label('Prices:'),
            dcc.Textarea(
                id='pe_prices',
                placeholder='Paste Samples Here',
                value="445, 187, 746, 328, 432, 212, 63, 98, 241, 601",
                style={'width': '100%',
                'height':'50%'},
            ),
            ],style={'columnCount': 2,'width': "55%"}),
            
            # html.Label('Result Price Elasticity:'),
            html.H4("Result Price Elasticity:"),
            html.Div(id='pe_data_output'),
            # html.Label('Optimal Price:'),
            html.H4("Optimal Price:"),
            html.Div(id='optimal_price_output'),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            # html.Br(),
        # ]),
    html.Div(
    [
        dcc.Graph(id='pe_raw-graph')

    ]),
    html.Div(
    [
        dcc.Graph(id='optimal_price_chart')

    ]),
    html.Br(),
    html.Br(),
# ])
],style={'columnCount': 2,'width': "80%"})

#BAR GRAPH - PE
@app.callback(
        dash.dependencies.Output('pe_data_output','children'),
            [dash.dependencies.Input('pe_prices','value'),
            dash.dependencies.Input('pe_quantities','value'),
            ])
def update_pe(pe_prices,pe_quantities):
    p1, p2 = text_parse(pe_prices,pe_quantities)

    est = sm.OLS(np.log(p2), sm.add_constant(np.log(p1))).fit()
    pe = est.params[1]

    return 'An increase in price by  1% changes the quantity demanded by: "{}"%'.format(pe)

#OPTIMAL PRICE VAL
@app.callback(
        dash.dependencies.Output('optimal_price_output','children'),
            [dash.dependencies.Input('pe_prices','value'),
            dash.dependencies.Input('pe_quantities','value'),
            ])
def optimal_price(pe_prices,pe_quantities):
    p1, p2 = text_parse(pe_prices,pe_quantities)

    df = pd.DataFrame({
    'demand': p2,
    'price': p1
    })

    #find demands at price levels
    # df['demand'] = ""
    # for idx,row in df.iterrows():
    #     x = np.sum(df[df['price']>=row['price']]['quantity'])
    #     df.set_value(idx,'demand',x)

    def f(x, A, B):
        return A*x + B

    #curve fit for slope & intercept
    A,B = curve_fit(f, np.array(df['demand'].astype(float)), 
                    np.array(df['price'].astype(float)))[0]

    #price function
    def price(x,a=B):
        return (a-x)

    #optimization function
    def objective(x_t,a=B):
        return -1.0 * np.sum(x_t * price(x_t, a=a))


    #get profits (revenues)
    profits = [-1*objective(i) for i in range(1000)]
    profits = pd.DataFrame(profits).reset_index()
    profits.columns = ["price","revenue"]
    profits = profits[profits['revenue']>0]
    
    #optimize
    x_start = B/2
    result2 = optimize.minimize(objective, x0=x_start, args=(B),
                            method='SLSQP', bounds="",  constraints="")
    result2 = np.sum(result2['x'])
    result2 = float(result2)

    return 'The optimal price to maximize revenue is: $"{}"'.format(result2)


#OPTIMAL PRICE CHART
@app.callback(
        dash.dependencies.Output('optimal_price_chart','figure'),
            [dash.dependencies.Input('pe_prices','value'),
            dash.dependencies.Input('pe_quantities','value'),
            ])
def optimal_price_chart(pe_prices,pe_quantities):
    p1, p2 = text_parse(pe_prices,pe_quantities)

    df = pd.DataFrame({
        'demand': p2,
        'price': p1
        })

    # survey = pd.DataFrame({
    #     "price":[445, 187, 746, 328, 432, 212, 63, 98, 241, 601],
    #     "demand":[3, 8, 1, 5, 4, 7, 10, 9, 6, 2],
    # })

    #find demands at price levels
    # df['demand'] = ""
    # for idx,row in df.iterrows():
    #     x = np.sum(df[df['price']>=row['price']]['quantity'])
    #     df.set_value(idx,'demand',x)

    from scipy.optimize import curve_fit

    def f(x, A, B):
        return A*x + B

    A,B = curve_fit(f, np.array(df['demand'].astype(float)), 
                    np.array(df['price'].astype(float)))[0]
    print("p =",B,A,"* q")

    #price function
    def price(x,a=B):
        return (a-x)

    #optimization function
    def objective(x_t,a=B):
        return -1.0 * np.sum(x_t * price(x_t, a=a))

    price_search = np.linspace(min(df['price']),max(df['price']),num=1000)

    #get profits (revenues)
    profits = [-1*objective(i) for i in price_search]

    profits = pd.DataFrame({"price":price_search,"revenue":profits})

    profits = profits[profits['revenue']>0]

    REVENUE = profits['revenue']
    PRICE = profits['price']

    # REVENUE = [0,10,20,30,40,50,40,30,20,10,0]
    # PRICE = [0,1,2,3,4,5,6,7,8,9,10]
    
    trace1 = go.Scatter(
            x=PRICE,
            y=REVENUE,
            mode = 'lines',
            name = 'Revenue by Price'
        )

    return {

        'data':[trace1],
        'layout': {
            'title': 'Revenue as a Function of Price',
            'xaxis' : {'title': 'Price'},
            'yaxis' : {'title': 'Revenue'},
        }

    }



#LINE GRAPH - PRICE ELASTICITY
@app.callback(dash.dependencies.Output('pe_raw-graph','figure'),
            [dash.dependencies.Input('pe_prices','value'),
            dash.dependencies.Input('pe_quantities','value'),
            ])
def update_pegraph(pe_prices,pe_quantities):

    p1, p2 = text_parse(pe_prices,pe_quantities)

    # p1 = [445, 187, 746, 328, 432, 212, 63, 98, 241, 601]
    # p2 = [3, 8, 1, 5, 4, 7, 10, 9, 6, 2]

    df = pd.DataFrame(
    {'quantity': p2,
        'price': p1})

    df['demand'] = ""
    for idx,row in df.iterrows():
        x = np.sum(df[df['price']>=row['price']]['quantity'])
        df.set_value(idx,'demand',x)

    df.sort_values(by="price",inplace=True)
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

# #LINE GRAPH - PRICE ELASTICITY (LOG SCALE)
# @app.callback(dash.dependencies.Output('pe_raw2-graph','figure'),
#             [dash.dependencies.Input('pe_prices','value'),
#             dash.dependencies.Input('pe_quantities','value'),
#             ])
# def update_pegraph2(pe_prices,pe_quantities):

#     p1, p2 = text_parse(pe_prices,pe_quantities)

#     df = pd.DataFrame(
#     {'quantity': p2,
#      'price': p1})

#     df['demand'] = ""
#     for idx,row in df.iterrows():
#         x = np.sum(df[df['price']>=row['price']]['quantity'])
#         df.set_value(idx,'demand',x)

#     df['demand'] = df['demand'].astype(float)
#     PRICE_LOG = np.log(np.log(df['price']))
#     DEMAND_LOG = np.log(df['demand'])

#     trace2 = go.Scatter(
#             x=DEMAND_LOG,
#             y=PRICE_LOG,
#             mode = 'lines',
#             name = 'Quantity Demanded'
#         )

#     return {

#         'data':[trace2],
#         'layout': {
#             'title': 'Demand as a Function of Price',
#             'xaxis' : {'title': 'Quantity Demand'},
#             'yaxis' : {'title': 'Price'},
#         }

#     }

#PAGE 9 - PEARSON CORRELATION
##############################################################################
##############################################################################
#Text

page_9_layout = html.Div([
    html.H2('Pearson Correlation'),
    dcc.Link('Go back to home', href='/'),
    html.Br(),
    html.Div([
        html.H4('Description:'),
        html.P(
        'A Pearson Correlation Coefficient measures the linear correlation between two variables \
        and ranges between -1 and +1 where +1 is total positive linear correlation, 0 is no linear \
        correlation and -1 is total negative linear correlation.'
    ),
    ],style={'width': '80%'}),
    #PARAMETERS
    html.H4("Input:"),
        html.Div([
            # INPUT BOX 1 - Sample 1
            html.Label('Variable 1 Values:'),
            dcc.Textarea(
                id='sample1_dat',
                placeholder="Paste Samples Here",
                    value="14.300000190734901,15.1000003814697,15.300000190734901,15.199999809265099,15.800000190734901,14.699999809265099,16.799999237060501,17.600000381469702,19.299999237060501,19.799999237060501,19.200000762939499,20.600000381469702,20.600000381469702,21.100000381469702,21.299999237060501,22.899999618530302,24.5,25.100000381469698,25.200000762939499,26.299999237060501,27.399999618530302,27.399999618530302,28.299999237060501,27.100000381469698,27.0,26.399999618530302,28.5,29.0,30.399999618530302,32.799999237060497,32.700000762939503,33.700000762939503,33.900001525878899,34.0,35.299999237060497,36.400001525878899,37.200000762939503,39.400001525878899,39.599998474121101,40.900001525878899,42.400001525878899,44.099998474121101,46.5,48.200000762939503,48.799999237060497,48.200000762939503,48.799999237060497,49.5,49.799999237060497,52.900001525878899,53.200000762939503,53.900001525878899",
                style={'width': '60%',
                'height':'50%'},
            ),
            html.Br(),
            # INPUT BOX 2 - Sample 2
            html.Label('Variable 2 Values:'),
            dcc.Textarea(
                id='sample2_dat',
                placeholder='Paste Samples Here',
                value="2.8838200569152797,2.8038499355316202,2.7584900856018102,2.67040991783142,2.3940498828887899,2.5,2.1617600917816202,2.0391499996185298,1.95848000049591,1.7732000350952102,1.77026998996735,1.5852799415588399,1.6556299924850499,1.6111099720001201,1.5548399686813401,1.58095002174377,1.6265399456024199,1.4610799551010099,1.4712599515914899,1.47411000728607,1.35052001476288,1.3061699867248502,1.27751004695892,1.73649001121521,1.4665299654007,1.5130100250244098,1.3514900207519498,1.2755800485611,1.3128800392150901,1.2010999917983998,1.14563000202179,1.0616099834442101,0.98237997293472301,0.96687000989913907,1.0490900278091402,0.97118997573852506,1.0529199838638299,0.99735999107360795,1.0574799776077299,1.1056499481201201,1.0321300029754601,0.96696001291274991,0.940129995346069,0.95502001047134411,0.94533997774124101,0.93307000398635898,0.97259002923965499,0.98754000663757291,0.97913998365402188,0.97118997573852506,0.94599002599716198,0.94862002134323087",
                style={'width': '60%',
                'height':'50%'},
            ),
            ],style={'columnCount': 2,'width': "100%"}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H3('Result:'),
            html.Div(id='pearson_correl_output',style={'color':'blue'}),
        html.Br(),
        html.Div(
        [
            dcc.Graph(id='pearson_correl-graph')

        ]),
    html.Br(),
    html.Br(),
],style={'columnCount': 2,'width': "60%"})
# ])

#BAR GRAPH - PE
@app.callback(
        dash.dependencies.Output('pearson_correl_output','children'),
            [dash.dependencies.Input('sample1_dat','value'),
            dash.dependencies.Input('sample2_dat','value'),
            ])
def update_correl(sample1_dat,sample2_dat):
    p1, p2 = text_parse(sample1_dat,sample2_dat)

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
    p1, p2 = text_parse(sample1_dat,sample2_dat)

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
            'xaxis' : {'title': 'Variable 2'},
            'yaxis' : {'title': 'Variable 1'},
        }
    }

#PAGE 10 - FINANCIAL MISC
##############################################################################
##############################################################################
#Text

page_10_layout = html.Div([
    html.H2('Financial Calculator Misc'),
    # dcc.Markdown(children=markdown_text_finance_calc),
    dcc.Link('Go back to home', href='/'),
html.H4('Description:'),
html.P(
    '''
    This section has some tools for common financial calculations,
    including the valuation of annuities and perpetuities.
    ''' 
),
# html.Br(),
html.P('An Annuity is a fixed sum of money paid each year or interval.'),
html.P('A Perpetuity is a fixed sum of money paid each year or interval, forever.'),


    # html.Br(),

        html.Div(
        [
            # INPUT BOX 1 - Discount Rate
            html.H4('Calculation Type:'),
            # dcc.Input(
            #     id='calc_type',
            #     placeholder='Present Value',
            #     type='text',
            #     value='Present Value'
            # ),
            dcc.Dropdown(
                id='calc_type',
                options=[
                    {'label': 'Present Value of an Annuity', 'value': 'PV_A'},
                    {'label': 'Present Value of a Growing Annuity', 'value': 'PV_GA'},
                    {'label': 'Present Value of a Perpetuity', 'value': 'PV_P'},
                    {'label': 'Present Value of a Growing Perpetuity', 'value': 'PV_GP'},
                    {'label': 'Future Value of an Annuity', 'value': 'FV_A'},
                ],
            value='PV_A'
            ),

            # INPUT BOX 1 - Discount Rate
            html.Label('Discount Rate:'),
            dcc.Input(
                id='discount_rate',
                placeholder='0.05',
                type='text',
                value='0.05'
            ),

            # INPUT BOX 2 - Growth Rate
            html.Label('Growth Rate:'),
            dcc.Input(
                id='growth_rate',
                placeholder='0.03',
                type='text',
                value='0.03'
            ),

            # INPUT BOX 2 - Payment Amount
            html.Label('Payment Amt:'),
            dcc.Input(
                id='pmt_amt',
                placeholder='50',
                type='text',
                value='50'
            ),

            # INPUT BOX 3 - Number of Periods
            html.Label('Number of Periods:'),
            dcc.Input(
                id='num_periods',
                placeholder='12',
                type='text',
                value='12'
            ),

            html.Label('Result Amount:'),
            html.Div(id='result'),
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

@app.callback(
    dash.dependencies.Output('result','children'),
            [dash.dependencies.Input('discount_rate','value'),
            dash.dependencies.Input('growth_rate','value'),
            dash.dependencies.Input('pmt_amt','value'),
            dash.dependencies.Input('num_periods','value'),
            dash.dependencies.Input('calc_type','value'),
            ])
def update_fincalc(discount_rate,growth_rate,
                        pmt_amt,num_periods,calc_type):

    result = ""
    pmt_amt = float(pmt_amt)
    discount_rate = float(discount_rate)
    num_periods = float(num_periods)
    growth_rate = float(growth_rate)

    if calc_type == "PV_A":
        result = pmt_amt * (1/discount_rate) * (1 - (1/(1+discount_rate)**num_periods))

    elif calc_type == "PV_GA":
        result = pmt_amt * (1/(discount_rate-growth_rate) * (1 - ((1+growth_rate)/(1+discount_rate))**num_periods))

    elif calc_type == "PV_P":
        result = pmt_amt/discount_rate

    elif calc_type == "PV_GP":
        result = pmt_amt/(discount_rate - growth_rate)

    elif calc_type == "FV_A":
        result = pmt_amt * (1/discount_rate) * ((1+discount_rate)**num_periods - 1)


    return 'The result is: "${}"'.format(result)


# # Page 11 - Price Optimization
# #############################################################################
# #############################################################################

# page_11_layout = html.Div([
#     html.H1('Price Optimization'),
#     # dcc.Markdown(children=markdown_text_finance_calc),
#     dcc.Link('Go back to home', href='/'),
#     html.H3('Description:'),
#     html.P(
#         '''
#         This section finds the optimal price to reach maxiumum revenue, 
#         given values for price and demand.
#         ''' 
#     ),
#    #PARAMETERS
#     html.H4("Input:"),
#         html.Div([
#             # INPUT BOX 1 - Sample 1
#             html.Label('Price:'),
#             dcc.Textarea(
#                 id='sample1_dat_price',
#                 placeholder="Paste Samples Here",
#                     value="10,20,15,25,20,30",
#                 style={'width': '60%',
#                 'height':'50%'},
#             ),
#             html.Br(),
#             # INPUT BOX 2 - Sample 2
#             html.Label('Demand:'),
#             dcc.Textarea(
#                 id='sample2_dat_demand',
#                 placeholder='Paste Samples Here',
#                 value="200, 100, 350, 400, 500, 230",
#                 style={'width': '60%',
#                 'height':'50%'},
#             ),
#             ],style={'columnCount': 2,'width': "60%"}),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             # html.Br(),
#             html.Br(),
#             html.Label('Result:'),
#             html.Div(id='result2'),
#         ]),

# @app.callback(
#     dash.dependencies.Output('result2', 'children'),
#               [dash.dependencies.Input('sample1_dat_price', 'value'),
#               dash.dependencies.Input('sample1_dat_demand', 'value')
#               ])
# def optimal_price(sample1_dat_price,sample1_dat_demand):
#     # result2 = ""
#     # p,d = text_parse(sample1_dat_price,sample1_dat_demand)
#     # survey = pd.DataFrame({
#     #     'price':p,
#     #     'demand':d
#     #     })
#     # def f(x, A, B):
#     #     return A*x + B
#     # A,B = curve_fit(f, np.array(survey['demand'].astype(float)), 
#     #                 np.array(survey['price'].astype(float)))[0]
#     # def price(x,a=B):
#     #     return (a-x)
#     # def objective(x_t,a=B):
#     #     return -1.0 * np.sum(x_t * price(x_t, a=a))
#     # x_start = B/2
#     # result2 = optimize.minimize(objective, x0=x_start, args=(B),
#     #                         method='SLSQP', bounds="",  constraints="")
#     # result2 = np.sum(result2['x'])
#     # result2 = float(result2)
#     result2 = "lol"
#     return 'Optimal Price: "{}"'.format(result2)



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
    elif pathname == '/2_sample_survival_means':
        return page_11_layout
    elif pathname == '/count_data':
        return page_5_layout
    elif pathname == '/cvar':
        return page_6_layout
    # elif pathname == '/pe':
        # return page_7_layout
    elif pathname == '/pe_raw':
        return page_8_layout
    elif pathname == '/pearson_correl':
        return page_9_layout
    elif pathname == '/financial_misc':
        return page_10_layout
    # elif pathname == '/price_optimization':
    #     return page_11_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)