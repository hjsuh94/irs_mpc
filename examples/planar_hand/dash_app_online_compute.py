import os
import json
import pickle

import meshcat
import numpy as np
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from qsim.system import cpp_params_from_py_params
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)
from planar_hand_setup import (model_directive_path, h,
                               robot_stiffness_dict, object_sdf_dict,
                               robot_l_name, robot_r_name, object_name)

from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from rrt.utils import set_orthographic_camera_yz
from rrt.planner import ConfigurationSpace

from dash_app_common import (add_goal, X_WG0, hover_template_1step,
                             hover_template_trj)

# %% quasistatic dynamics
sim_params = QuasistaticSimParameters()
q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params,
    internal_vis=True)

# construct C++ backend.
sim_params_cpp = cpp_params_from_py_params(sim_params)
q_sim_cpp = QuasistaticSimulatorCpp(
    model_directive_path=model_directive_path,
    robot_stiffness_str=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=sim_params_cpp)

q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)

model_a_l = q_sim_py.plant.GetModelInstanceByName(robot_l_name)
model_a_r = q_sim_py.plant.GetModelInstanceByName(robot_r_name)
model_u = q_sim_py.plant.GetModelInstanceByName(object_name)

cspace = ConfigurationSpace(
    model_u=model_u, model_a_l=model_a_l, model_a_r=model_a_r, q_sim=q_sim_py)

#%% meshcat
vis = q_sim_py.viz.vis
set_orthographic_camera_yz(vis)
add_goal(vis)

# %%
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dcc.Graph(
                id='reachable-sets',
                figure={}),
            width={'size': 6, 'offset': 0, 'order': 0},
        ),
        dbc.Col(
            html.Iframe(src='http://127.0.0.1:7000/static/',
                        height=800, width=1000),
            width={'size': 6, 'offset': 0, 'order': 0},
        )
    ]),
    dbc.Row([
        dbc.Col([
            html.H3('1. object pose: y, z, theta'),
            dcc.Slider(id='y-slider', min=-0.5, max=0.5, value=0, step=0.01,
                       marks={-0.5: {'label': '-0.5'},
                              0: {'label': '0'},
                              0.5: {'label': '0.5'}},
                       tooltip={"placement": "bottom", "always_visible": False}
                       ),
            dcc.Slider(id='z-slider', min=-0.2, max=0.8, value=0.3, step=0.01,
                       marks={-0.2: {'label': '-0.2'},
                              0.8: {'label': '0.8'},
                              0.3: {'label': '0.3'}},
                       tooltip={"placement": "bottom", "always_visible": False}
                       ),
            dcc.Slider(id='theta-slider', min=-np.pi / 2, max=np.pi / 2,
                       value=0, step=0.01,
                       marks={-np.pi / 2: {'label': '-pi/2'},
                              0: {'label': '0'},
                              np.pi / 2: {'label': 'pi/2'}},
                       tooltip={"placement": "bottom", "always_visible": False}
                       ),
            html.Div(id='q_u0_display',
                     children='Current q_u0:'),
            html.H3('2. sample grasp'),
            html.Button('Sample Grasp', id='sample_grasp', n_clicks=0),
            html.Div(id='q_a0_display', children='Current q_a0'),
            dcc.Store(id='q_u0'),
            dcc.Store(id='q_a0')
        ],
            width={'size': 3, 'offset': 0, 'order': 0},
        )
    ]),
], fluid=True)


@app.callback(
    [Output('q_u0', 'data'),  Output('q_u0_display', 'children')],
    [Input('y-slider', 'value'), Input('z-slider', 'value'),
     Input('theta-slider', 'value')])
def update_qu0(y, z, theta):
    p_WB = np.array([0, y, z])
    vis['drake/plant/sphere/sphere'].set_transform(
        meshcat.transformations.translation_matrix(p_WB) @
        meshcat.transformations.rotation_matrix(theta, [1, 0, 0]))
    q_u0_json = json.dumps({'y': y, 'z': z, 'theta': theta})

    return q_u0_json, html.Div(q_u0_json)


@app.callback(
    [Output('q_a0', 'data'), Output('q_a0_display', 'children')],
    Input('sample_grasp', 'n_clicks'),
    State('q_u0', 'data'))
def update_qa0(n_clicks, qu0_data):
    if qu0_data is None:
        return json.dumps({'qa_l': [0., 0.], 'qa_r': [0., 0.]})
    q_u0_dict = json.loads(qu0_data)
    q_u0 = np.array([q_u0_dict['y'], q_u0_dict['z'], q_u0_dict['theta']])
    q_dict = cspace.sample_contact(q_u=q_u0)
    q_sim_py.update_mbp_positions(q_dict)
    q_sim_py.draw_current_configuration()

    q_a0_dict = {'qa_l': q_dict[model_a_l].tolist(),
                 'qa_r': q_dict[model_a_r].tolist()}
    q_a0_json = json.dumps(q_a0_dict)

    return q_a0_json, html.Div(q_a0_json)


if __name__ == '__main__':
    app.run_server(debug=True)
