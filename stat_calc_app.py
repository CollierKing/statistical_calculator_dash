import dash
import dash_core_components as dcc
import dash_html_components as html

#additional:
import numpy as np
## sample size
from scipy.stats import norm, zscore
## chisquared
from statsmodels.stats.proportion import proportions_chisquare

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
##############################################################################

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Br(),
    # html.Label('Power (1-False Negative Rate)'),

    # dcc.Slider(
    #     min=0,
    #     max=10,
    #     marks={i: 'Label {}'.format(i) if i == 0.1 else str(i) for i in range(0, 1)},
    #     value=8,
    # )
])


index_page = html.Div([
    html.H1('Statistical Tools:'),
    html.Br(),
    dcc.Link('Sample Size Calculator', href='/sample_size_calculator'),
    html.Br(),
    dcc.Link('Chi Squared Test', href='/chi_squared_test'),
])

# PAGE 1 - SAMPLE SIZE CALCULATOR
##############################################################################
##############################################################################

page_1_layout = html.Div([
    html.H1('Sample Size Calculator'),
    html.Br(),

    # INPUT BOX 1 - CONV RATE
    html.Label('Baseline Rate'),
    dcc.Input(
        id='baseline_input',
        placeholder='0.3',
        type='text',
        value='0.3'
    ),
    html.Div(id='baseline_content'), #WIP
    html.Br(),

    # INPUT BOX 2 - EFFECT SIZE
    html.Label('Effect Size'),
    dcc.Input(
        id='effect_size_input',
        placeholder='0.05',
        type='text',
        value='0.35'
    ),
    html.Div(id='effect_size_content'), #WIP
    html.Br(),

    # INPUT BOX 3 - POWER LEVEL
    html.Label('Power (1-False Negative Rate)'),
    dcc.Input(
        id='statistical_power_input',
        placeholder='0.8',
        type='text',
        value='0.8'
    ),
    html.Div(id='statistical_power_content'),
    html.Br(),

    # INPUT BOX 4 - SIGNIF LEVEL
    html.Label('Significance Level'),
    dcc.Input(
        id='significance_level_input',
        placeholder='0.05',
        type='text',
        value='0.05'
    ),
    html.Div(id='significance_level_content'),
    html.Br(),

    html.Label('Sample Size:'),
    html.Div(id='sample_size_content'),
    html.Br(),
    html.Br(),
    dcc.Link('Go to Page 2', href='/chi_squared_test'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])

#BASELINE
@app.callback(dash.dependencies.Output('baseline_content', 'children'),
              [dash.dependencies.Input('baseline_input', 'value')])
def baseline_content(value):
    return 'You have selected "{}"'.format(value)

#EFFECT
@app.callback(dash.dependencies.Output('effect_size_content', 'children'),
              [dash.dependencies.Input('effect_size_input', 'value')])
def effect_size_content(value):
    return 'You have selected "{}"'.format(value)

#POWER
@app.callback(dash.dependencies.Output('statistical_power_content', 'children'),
              [dash.dependencies.Input('statistical_power_input', 'value')])
def statistical_power_content(value):
    return 'You have selected "{}"'.format(value)

#SIGNIF
@app.callback(dash.dependencies.Output('significance_level_content', 'children'),
              [dash.dependencies.Input('significance_level_input', 'value')])
def significance_level_content(value):
    return 'You have selected "{}"'.format(value)

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
    html.Br(),

    # INPUT BOX 1 - Sample1 Successes
    html.Label('Sample 1: # successes'),
    dcc.Input(
        id='sample1_successes',
        placeholder='150',
        type='text',
        value='150'
    ),
    html.Div(id='sample1_successes_out'), #WIP
    html.Br(),

    # INPUT BOX 2 - Sample1 Trials
    html.Label('Sample 1: # trials'),
    dcc.Input(
        id='sample1_trials',
        placeholder='1000',
        type='text',
        value='1000'
    ),
    html.Div(id='sample1_trials_out'), #WIP
    html.Br(),

    # INPUT BOX 3 - Sample2 Successes
    html.Label('Sample 2: # successes'),
    dcc.Input(
        id='sample2_successes',
        placeholder='180',
        type='text',
        value='180'
    ),
    html.Div(id='sample2_successes_out'),
    html.Br(),

    # INPUT BOX 4 - Sample2 Trials
    html.Label('Sample 2: # trials'),
    dcc.Input(
        id='sample2_trials',
        placeholder='1000',
        type='text',
        value='1000'
    ),
    html.Div(id='sample2_trials_out'),
    html.Br(),

    html.Label('Result p-value:'),
    html.Div(id='chisq_content'),
    html.Br(),
    html.Br(),
    dcc.Link('Go to Page 1', href='/sample_size_calculator'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),

])

# Sample1 Successes
@app.callback(dash.dependencies.Output('sample1_successes_out', 'children'),
              [dash.dependencies.Input('sample1_successes', 'value')])
def sample1_successes(value):
    return 'You have selected "{}"'.format(value)

# Sample1 Trials
@app.callback(dash.dependencies.Output('sample1_trials_out', 'children'),
              [dash.dependencies.Input('sample1_trials', 'value')])
def sample1_trials(value):
    return 'You have selected "{}"'.format(value)

# Sample2 Successes
@app.callback(dash.dependencies.Output('sample2_successes_out', 'children'),
              [dash.dependencies.Input('sample2_successes', 'value')])
def sample2_successes(value):
    return 'You have selected "{}"'.format(value)

# Samples2 Trials
@app.callback(dash.dependencies.Output('sample2_trials_out', 'children'),
              [dash.dependencies.Input('sample2_trials', 'value')])
def sample2_trials(value):
    return 'You have selected "{}"'.format(value)

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














# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/sample_size_calculator':
        return page_1_layout
    elif pathname == '/chi_squared_test':
        return page_2_layout
    else:
        return index_page
    # You could also return a 404 "URL not found" page here

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


if __name__ == '__main__':
    app.run_server(debug=True)