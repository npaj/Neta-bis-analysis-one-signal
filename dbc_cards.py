import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import configparser

wavelet_details = ["trial 'mu' values for wavelets",html.Br(),
"gmw   : {'beta' -> 'mu' =60, 'gamma':3}",html.Br(),
"morlet:{'mu' = 13.4} ",html.Br(),
"bump  : {'mu'  =5 , 's':1.0} ",html.Br(),
"cmhat : {'mu' =1 , 's':1.0} ",html.Br(),
"hhhat :{'mu'  5}",html.Br(),
#html.Link()
html.A("Reference", href="https://github.com/OverLordGoldDragon/ssqueezepy/", target="_blank")
]

color_card ="dark"
outline =True

config = configparser.ConfigParser()
config.read('Parameters.ini')

params_ini = config._sections

def background_values():
    if isinstance(params_ini['background_functions']['select'], str):
        params_ini['background_functions']['select'] = eval(params_ini['background_functions']['select'])

    return [params_ini['background_functions']['select']['val'], 
            params_ini['background_functions']['select']['opt']]



file_card = dbc.Card(
    [
        dbc.CardHeader("Select file"),
        dbc.CardBody(
            [
                html.Div(
                    [
                        # dbc.Label("Enter file path", style={'font-weight': 'bold'}),
                        # html.Br(),
                        dcc.Input(id='filepath', type="text",
                                  placeholder="paste file path...", name="file", size='37'),
                        html.Br(),html.Br(),
                        # dbc.Label("Choose file"),
                        dcc.Dropdown(id='file_list', options=[])

                    ]
                ),

            ]
        )

    ], color=color_card, outline=outline
)


# Parameter card

Params_card_1 = dbc.Card(
    [dbc.CardHeader("Background parameters"),
        dbc.CardBody(
            [
                # select time window for processing
                html.Div([
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(f'{_}'),
                                    html.Br(),
                                    dcc.Input(id=f'time-params-background_{_}', type="number",
                                              value=params_ini['time-params-background'][_], style={'width': 80}),
                                ]

                            )for _ in params_ini['time-params-background']
                        ],
                    ),

                ], id='time-params-background'

                ),

                html.Br(),
                # background subtraction method
                dbc.Row(
                    [
                        dbc.Col([
                            dcc.Dropdown(background_values()[1], background_values()[0],
                                           id='background_select') #inline=True, 
                        ],
                        ),
                    ]
                ),
                html.Div([

                ], id='background_params'),
                # submit button
                html.Br(),
                dbc.Row(
                    [
                        dbc.Button('Submit', id='submit-val_1', n_clicks=0,
                                   style={'width': 100}, color="dark", outline=True)
                    ]
                )

            ]
    )

    ], color=color_card, outline=outline

)

fft_card = dbc.Card(
    [
        dbc.CardHeader("FFT parameters"),
        dbc.CardBody([
            html.Div(
                [
                    dbc.Row(
                     [
                         dbc.Col(
                             [
                                 dbc.Label(f'{_}'),
                                 html.Br(),
                                 dcc.Input(id=f'fft-params_{_}', type="number",
                                           value=params_ini['fft-params'][_], style={'width': 80}),

                             ]

                         )for _ in params_ini['fft-params']

                     ]
                     ),

                ], id='fft-params'

            ),
            html.Br(),
            html.P("(Tukey window :: alpha = 0 ;rectangular window, alpha = 1 ;Hann window)",
                   ),
            html.Br(),
            dbc.Row(
                [
                    dbc.Button('Submit', id='submit-val_2', n_clicks=0,
                               style={'width': 100}, color="dark", outline=True,)
                ]
            ),
        ],
        )

    ], color=color_card, outline=outline
)



time_freq_card = dbc.Card(
    [
        dbc.CardHeader("time-frequency analysis"),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dcc.Dropdown(id='TF-options',
                             options= eval(params_ini['TF-options']['select'])['opt'], value=eval(params_ini['TF-options']['select'])['val'])
                        
                    ]
                ),
                html.Br(),
                html.Div(
                    [
                    ],id='TF-parameters'
                ),
                html.Br(),
                dbc.Row(
                [   
                    dbc.Button('Submit', id='submit-val_3', n_clicks=0,
                               style={'width': 100}, color="dark", outline=True,),

                ]
                ),
                
                html.Div(children=[],id='TF_card_wlet_resolution'),

                html.Div(
                                        html.Details(id='wavelet_details', open=False, children=
                    [
                       html.Summary( 'wavelet details'),
                       html.P(wavelet_details)

                        ]

                    )
                ),


            ]

        )
    ], color=color_card, outline=outline
)