import dash
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
from matplotlib.pyplot import title
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
import ast
import configparser
import load_data

from requests import options
import utility as ut
from dbc_cards import *


# Build your components
app = dash.Dash(external_stylesheets=[dbc.themes.CERULEAN])

global files
files=[]

config = configparser.ConfigParser()
config.read('Parameters.ini')
params_ini = config._sections


def rangeslider_fft():
    return html.Div([dcc.RangeSlider(float(params_ini['time-params-background']['tmin']),
                    float(
        params_ini['time-params-background']['tmax']),
        (float(params_ini['time-params-background']['tmax'])
         - float(params_ini['time-params-background']['tmin']))/int(params_ini['fft-rangeslider']['nslider']),
        marks=None, tooltip={"placement": "bottom", "always_visible": True},
        value=[float(params_ini['fft-rangeslider']['tmin']),
               float(params_ini['fft-rangeslider']['tmax'])], id='fft-rangeslider')], id='fft-rangeslider_children')


def rangeslider_graph1_xlim(xmin=-1, xmax=3, value=[-0.5, 2], N=100):
    xmin = np.round(xmin, 1)
    xmax = np.round(xmax, 1)
    return html.Div(
        [dcc.RangeSlider(xmin, xmax, (xmax-xmin)/N, value=value,
                         marks=None, tooltip={"placement": "bottom", "always_visible": True},
                         id='graph1_xlim-rangeslider')],
        id='graph1_xlim-rangeslider_children')


def write_params_file():
    config = configparser.ConfigParser()
    for elements in list(params_ini):
        config[elements] = params_ini[elements]
    with open('Parameters.ini', 'w') as configfile:
        config.write(configfile)

app.layout = html.Div([
    html.H1("Data analysis tool box"),
    # html.Hr(),
    dcc.Tabs(vertical=False,children=[
        dcc.Tab(label='Single data',children=[
            dbc.Container(
                [
                    # html.H1("Data analysis tool box"),
                    # html.Hr(),
                    dbc.Row(
                        [
                            dbc.Col([file_card, html.Br(), Params_card_1], md=3),
                            dbc.Col(
                                html.Div(
                                    [
                                        dcc.Graph(id='raw-signal', figure={}),
                                        html.Div(
                                            [
                                                html.H6('xlim'),
                                                rangeslider_graph1_xlim(),
                                                html.H6('fft time range'),
                                                rangeslider_fft()

                                            ], style={'width': '85%', 'marginLeft': '50px'}
                                        )

                                    ]),

                                md=9),
                        ],
                        align="center",
                    ),
                    html.Br(),
                    dbc.Row(
                        [
                            dbc.Col([html.Br(), html.Br(), html.Br(),
                                    html.Br(), fft_card], md=3),
                            dbc.Col(
                                html.Div(
                                    [
                                        dcc.Graph(id='fft-graphs', figure={})

                                    ]
                                )

                            )
                        ]

                    ),


                    html.Br(), html.Br(), 

                    dbc.Row(
                        [
                            dbc.Col(time_freq_card,md=3),
                            
                            dbc.Col(
                                html.Div(
                                    [
                                        dcc.Graph(id='time_frequency_Graph', figure={}),

                                    ]
                                ),
                                md=9
                            )

                        ]
                    ),
                    
                    html.Br(), html.Br(),
                ],
                fluid=True,
            )
        ]
        ),

# Tab 2 data comparison



        dcc.Tab(
            label='Compare data', children=[
                dcc.Dropdown(files,
                     [],
                     multi=True, id='file_list_compare'),

            ])


    ])
])

# select file call back


@app.callback(
    Output('file_list', 'options'),
    Output('file_list_compare', 'options'),
    Input('filepath', 'value'),)
def update_output(filepath):
    files = []
    try:
        allfiles = os.listdir(filepath)
        files_1 = [fname for fname in allfiles if (fname.endswith('.txt'))]
        files = [fname for fname in files_1 if not(fname.endswith('lab.txt'))]
    except:
        pass

    return files,files


@app.callback(
    Output('raw-signal', 'figure'),
    Output('graph1_xlim-rangeslider_children', 'children'),
    Output('fft-rangeslider_children', 'children'),
    Output('file_info', 'value'),
    Input('file_format', 'value'),
    Input('file_list', 'value'),
    State('filepath', 'value'),
    State('background_select', 'value'),
    State('background_params', 'children'),
    State('time-params-background', 'children'),
    Input('graph1_xlim-rangeslider', 'value'),
    Input('fft-rangeslider', 'value'),
    Input('submit-val_1', 'n_clicks'),

)
def update_graph(filetype, file_name, filepath, background_fn, background_params, t_params_background, x_range, fft_rangeslider, n_click1):
    
    global data
    global data_filt
    global data_back

    if file_name is None:
        return dash.no_update
    else:
        # data = np.loadtxt(os.path.join(filepath, file_name))
        if filetype =="LAUM":
            temp= load_data.Load(file_name, filepath, NETA_BIS=True )
        else:
            temp= load_data.Load(file_name, filepath, False )
        
        
        data = temp.data
        data[:, 0] = data[:, 0] - float(params_ini['time-params-background']['t0'])
        data[:, 1] = data[:, 1] - np.average(data[:, 1][data[:, 0] < 0])
        fig = go.Figure(make_subplots(rows=2, cols=1))
        fig.add_trace(go.Scatter(name='Raw signal',
                      x=data[:, 0], y=data[:, 1]), row=1, col=1)

        dt = data[:, 0][1]-data[:, 0][0]

    if len(background_params) > 0:

        # write background params
        if isinstance(params_ini['background_functions']['select'], str):
            params_ini['background_functions']['select'] = eval(params_ini['background_functions']['select'])
        
        params_ini['background_functions']['select']['val'] =background_fn

        for ii, params in enumerate(params_ini[background_fn]):
            params_ini[background_fn][params] = background_params[0][
                'props']['children'][ii]['props']['children'][2]['props']['value']
            print(params_ini[background_fn][params])
        write_params_file()

        # write time-params-background
        for ii, params in enumerate(params_ini['time-params-background']):
            params_ini['time-params-background'][params] = t_params_background[0]['props']['children'][ii][
                'props']['children'][2]['props']['value']
        write_params_file()

        # write fft time range
        params_ini['fft-rangeslider']['tmin'] = fft_rangeslider[0]
        params_ini['fft-rangeslider']['tmax'] = fft_rangeslider[1]
        write_params_file()

        #Back ground subtraction
        data_back, data_filt = ut.background_sub(data,
                                                 background_fn, params_ini['time-params-background'], params_ini[background_fn])
        
        #add acoustic signal and background traces
        fig.add_trace(go.Scatter(
            name=background_fn, x=data_back[:, 0], y=data_back[:, 1]), row=1, col=1)
        fig.add_trace(go.Scatter(
            name='Acoustic signal', x=data_filt[:, 0], y=data_filt[:, 1]), row=2, col=1)

        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1, title_text ='time (ns)')

        return fig, rangeslider_graph1_xlim(xmin=data[:, 0][0], xmax=data[:, 0][-1], value=x_range), rangeslider_fft(), temp.pretty_print()
    else:
        return dash.no_update


@app.callback(
    Output('background_params', 'children'),
    Input('background_select', 'value')

)
def update_background_fn(background_fn):

    background_fit = params_ini[background_fn]
    
    child1 = dbc.Row(
        [
            dbc.Col([
                dbc.Label("{}".format(_)),
                html.Br(),
                dcc.Input(id='background_params{}'.format(_),
                          type="number", value=background_fit[_], style={'width': 80})

            ])for _ in background_fit

        ],

    ),
    return child1


@app.callback(
    Output('fft-graphs', 'figure'),
    State('fft-params', 'children'),
    Input('fft-rangeslider', 'value'),
    Input('raw-signal', 'figure'),
    Input('background_select', 'value'),
    Input('submit-val_2','n_clicks')

)
def update_fft_graphs(fftparams, ffttimerange, fig1, background_fn, nclicks2):

    if 'data_filt' in globals():
        
        for ii, params in enumerate(params_ini['fft-params']):
            params_ini['fft-params'][params] = fftparams[0]['props'][
                        'children'][ii]['props']['children'][2]['props']['value']

            write_params_file()

        if 'bandpass' in background_fn:
            ncols = 4
            subplot_titles=('Raw signal fft', 'Background fit fft','Acousic signal fft','Filter design')
        else:
            ncols = 3
            subplot_titles=('Raw signal fft', 'Background fit fft','Acousic signal fft')

        fig2 = go.Figure(make_subplots(rows=1, cols=ncols,subplot_titles=subplot_titles,horizontal_spacing = 0.075))

        freq_raw, resufft_raw = ut.fft_fn(
            data, params_ini['fft-rangeslider'], params_ini['fft-params'])
        fig2.add_trace(go.Scatter(x=freq_raw, y=resufft_raw), row=1, col=1)

        freq_back, resufft_back = ut.fft_fn(
            data_back, params_ini['fft-rangeslider'], params_ini['fft-params'])
        fig2.add_trace(go.Scatter(x=freq_back, y=resufft_back), row=1, col=2)

        freq_filt, resufft_filt = ut.fft_fn(
            data_filt, params_ini['fft-rangeslider'], params_ini['fft-params'])
        fig2.add_trace(go.Scatter(x=freq_filt, y=resufft_filt), row=1, col=3)
        if 'bandpass' in background_fn:
            w, h =ut.bandfilt_design(data, params_ini[background_fn])

            fig2.add_trace(go.Scatter(x=w, y=20 *np.log10(abs(h))), row=1, col=4)

        for coln in range(1,ncols+1):
            fig2.update_xaxes(title_text ='Frequency (GHz)', row=1, col=coln)
            
            if coln<4:
                fig2.update_yaxes(title_text ='|FFT|', row=1, col=coln)
            else:
                fig2.update_yaxes(title_text ='Amplitude (dB)', row=1, col=coln)
            
    
        fig2.update_layout(showlegend=False)
        return fig2
    else:
        return dash.no_update


@app.callback(
    Output('TF-parameters','children'),
    Input('TF-options', 'value')
)

def update_TF_Params_children(method):
 
    if isinstance(params_ini['TF-options']['select'], str) :
         params_ini['TF-options']['select']= eval(params_ini['TF-options']['select'])
    params_ini['TF-options']['select']['val'] = method
    write_params_file()

    if method =='stft':
        inputs=[]
        for key in params_ini[method]:
            inputs.append(dbc.Col([dbc.Label(key),html.Br(),
            dcc.Input(value= float(params_ini[method][key]),style={'width': 70}, type='number')
            ]))
    else:
        inputs=[]
        for key in params_ini[method]:
            if key == 'wavelets':
                if isinstance(params_ini[method][key],str):
                    params_ini[method][key] =eval(params_ini[method][key])

                inputs.append(dbc.Col([dbc.Label(key),html.Br(),
                dcc.Dropdown(options=params_ini[method][key]['opt'], 
                value= params_ini[method][key]['val'],style={'width': 90})
                ]))

            else:
                inputs.append(dbc.Col([dbc.Label(key),html.Br(),
                dcc.Input(value= float(params_ini[method][key]),style={'width': 70}, type='number')
                ]))
                

    return dbc.Row(inputs)


@app.callback(
    Output('time_frequency_Graph', 'figure'),
    Output('TF_card_wlet_resolution','children'),
    Input('TF-parameters','children'),
    Input('TF-options', 'value'),
    Input("select_colorscales", "value"),
    State('fft-graphs', 'figure'),
    Input('submit-val_3', 'n_clicks'),
    Input('cmap-rangeslider','value')
)
def update_time_frequency_graph(tf_child, method, cmap,FIG3,clicks,scale):

    
    

    for ii, key in enumerate(params_ini[method]):
        temp = tf_child[
                    'props']['children'][ii]['props']['children'][2]['props']['value']
        if isinstance(temp, str):
            if isinstance(params_ini[method][key] ,str):
                params_ini[method][key] =eval(params_ini[method][key]) 
            params_ini[method][key]['val'] =temp

        else:
            params_ini[method][key] =temp
    write_params_file()

    wlet_resolution =html.P(" ")
    if 'data_filt' in globals():
        subplot_titles =('Acoustic signal', method)
        fig3 =go.FigureWidget(make_subplots(rows=2, cols=1, subplot_titles=subplot_titles))
        fig3.add_trace(go.Scatter(x=data_filt[:,0], y= data_filt[:,1]),row=1, col=1)
        if method == 'stft':
            t,f, Zstft = ut.stft_fn(data_filt, params_ini[method])
            zmin=np.max(Zstft)*scale[0]/100; zmax =np.max(Zstft)*scale[1]/100
            fig3.add_trace(go.Heatmap(x=t, y= f, z=Zstft, colorscale=cmap, zmin=zmin, zmax=zmax),row=2, col=1)
        else:
            t,f, Zstft, Ncycle = ut.wavelet_fn(data_filt, params_ini)
            zmin=np.max(Zstft)*scale[0]/100; zmax =np.max(Zstft)*scale[1]/100
            fig3.add_trace(go.Heatmap(x=t, y= f, z=Zstft, colorscale=cmap,zmin=zmin, zmax=zmax),row=2, col=1)
            wlet_resolution =html.P(f" wavelet resolution: {np.round(Ncycle,1)} Cycles")

                
        fig3.update_xaxes(title_text ='time (ns)', row=2, col=1)
        fig3.update_yaxes(title_text ='Frequency (GHz)', row=2, col=1)
        fig3.update_yaxes(title_text ='Amplitude (V)', row=1, col=1)
        fig3.update_layout(showlegend=False)


        return fig3, wlet_resolution#,cmpa_scale
    else:
        return dash.no_update

    




# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8054)
