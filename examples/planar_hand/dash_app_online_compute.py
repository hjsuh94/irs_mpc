import json

import dash
import dash_bootstrap_components as dbc
import meshcat
import numpy as np
import plotly.graph_objects as go
import tqdm
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash_app_common import (add_goal, hover_template_reachability,
                             layout,
                             calc_principal_points,
                             create_pca_plots, calc_X_WG)
from irs_lqr.irs_lqr_quasistatic import (IrsLqrQuasistaticParameters,
                                         IrsLqrQuasistatic)
from irs_lqr.quasistatic_dynamics import QuasistaticDynamics
from planar_hand_setup import (gravity, contact_detection_tolerance,
                               decouple_AB, use_workers, gradient_mode,
                               task_stride, num_samples)
from planar_hand_setup import (model_directive_path, h,
                               robot_stiffness_dict, object_sdf_dict,
                               robot_l_name, robot_r_name, object_name)
from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from qsim.system import cpp_params_from_py_params
from quasistatic_simulator_py import (QuasistaticSimulatorCpp)
from rrt.planner import ConfigurationSpace
from rrt.utils import set_orthographic_camera_yz, sample_on_sphere

# %% quasistatic dynamics
sim_params = QuasistaticSimParameters(
    gravity=gravity,
    nd_per_contact=2,
    contact_detection_tolerance=contact_detection_tolerance,
    is_quasi_dynamic=True)
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
dim_x = q_dynamics.dim_x
dim_u = q_dynamics.dim_u
model_a_l = q_sim_py.plant.GetModelInstanceByName(robot_l_name)
model_a_r = q_sim_py.plant.GetModelInstanceByName(robot_r_name)
model_u = q_sim_py.plant.GetModelInstanceByName(object_name)

cspace = ConfigurationSpace(
    model_u=model_u, model_a_l=model_a_l, model_a_r=model_a_r, q_sim=q_sim_py)

# %% irs-lqr
params = IrsLqrQuasistaticParameters()
params.Q_dict = {
    model_u: np.array([10, 10, 10]),
    model_a_l: np.array([1e-3, 1e-3]),
    model_a_r: np.array([1e-3, 1e-3])}
params.Qd_dict = {model: Q_i * 100 for model, Q_i in params.Q_dict.items()}
params.R_dict = {
    model_a_l: 5 * np.array([1, 1]),
    model_a_r: 5 * np.array([1, 1])}

params.sampling = lambda u_initial, i: u_initial / (i ** 0.8)
params.std_u_initial = np.ones(dim_u) * 0.3

params.decouple_AB = decouple_AB
params.use_workers = use_workers
params.gradient_mode = gradient_mode
params.task_stride = task_stride
params.num_samples = num_samples
params.u_bounds_abs = np.array(
    [-np.ones(dim_u) * 2 * h, np.ones(dim_u) * 2 * h])
params.publish_every_iteration = False

T = int(round(2 / h))  # num of time steps to simulate forward.
params.T = T
duration = T * h

irs_lqr_q = IrsLqrQuasistatic(q_dynamics=q_dynamics, params=params)

# %% meshcat
vis = q_sim_py.viz.vis
q_sim_py.viz.reset_recording()
set_orthographic_camera_yz(vis)
add_goal(vis)

# %% dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

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
            html.Div(id='q_a0_display', children=''),
            dcc.Store(id='q_u0'),
            dcc.Store(id='q_a0'),
            dcc.Store(id='q_a_samples')
        ],
            width={'size': 3, 'offset': 0, 'order': 0},
        ),
        dbc.Col([
            html.H3('3. 1-step Reachable set'),
            html.Button('Compute and Draw', id='btn-1-step-reachability',
                        n_clicks=0),
            html.H3('4. Calc Trajectory'),
            html.Button('Calc', id='btn-calc-trj', n_clicks=0),
            html.Pre(id='calc-trj-update', style=styles['pre']),
            ],
            width={'size': 3, 'offset': 0, 'order': 0}
        ),
        dbc.Col([
            dcc.Markdown(
                """
                **Hover Data**
            
                Mouse over values in the graph.
                """),
            html.Pre(id='hover-data', style=styles['pre'])],
            width={'size': 3, 'offset': 0, 'order': 0}),
        dbc.Col([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
                """),
            html.Pre(id='click-data', style=styles['pre']),
            dcc.Store(id='selected-goal')],
            width={'size': 3, 'offset': 0, 'order': 0}),
    ])
], fluid=True)


@app.callback(
    [Output('q_u0', 'data'), Output('q_u0_display', 'children')],
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
def update_qa0(n_clicks, q_u0_json):
    if q_u0_json is None:
        return json.dumps({'qa_l': [0., 0.], 'qa_r': [0., 0.]}), html.Div('')
    q_u0_dict = json.loads(q_u0_json)
    q_u0 = np.array([q_u0_dict['y'], q_u0_dict['z'], q_u0_dict['theta']])
    q_dict = cspace.sample_contact(q_u=q_u0)
    q_sim_py.update_mbp_positions(q_dict)
    q_sim_py.draw_current_configuration()

    q_a0_dict = {'qa_l': q_dict[model_a_l].tolist(),
                 'qa_r': q_dict[model_a_r].tolist()}
    q_a0_json = json.dumps(q_a0_dict)

    return q_a0_json, html.Div(q_a0_json)


def get_x0_and_u0_from_json(q_u0_json, q_a0_json):
    q_u0_dict = json.loads(q_u0_json)
    q_a0_dict = json.loads(q_a0_json)
    q_u0 = np.array([q_u0_dict['y'], q_u0_dict['z'], q_u0_dict['theta']])

    q0_dict = {model_u: q_u0,
               model_a_l: q_a0_dict['qa_l'],
               model_a_r: q_a0_dict['qa_r']}

    x0 = q_dynamics.get_x_from_q_dict(q0_dict)
    u0 = q_dynamics.get_u_from_q_cmd_dict(q0_dict)

    return x0, u0, q_u0


@app.callback(
    [Output('reachable-sets', 'figure'), Output('q_a_samples', 'data')],
    Input('btn-1-step-reachability', 'n_clicks'),
    [State('q_u0', 'data'), State('q_a0', 'data')])
def update_reachability(n_clicks, q_u0_json, q_a0_json):
    if q_u0_json is None or q_a0_json is None:
        return {}, json.dumps(None)
    # initial system config
    x0, u0, q_u0 = get_x0_and_u0_from_json(q_u0_json, q_a0_json)

    # sampling 1-step reachable set
    n_samples = 2000
    radius = 0.2
    du = np.random.rand(n_samples, 4) * radius * 2 - radius
    qu_samples = np.zeros((n_samples, 3))
    qa_l_samples = np.zeros((n_samples, 2))
    qa_r_samples = np.zeros((n_samples, 2))

    def save_x(x: np.ndarray):
        q_dict = q_dynamics.get_q_dict_from_x(x)
        qu_samples[i] = q_dict[model_u]
        qa_l_samples[i] = q_dict[model_a_l]
        qa_r_samples[i] = q_dict[model_a_r]

    for i in tqdm.tqdm(range(n_samples)):
        u = u0 + du[i]
        x_1 = q_dynamics.dynamics(x0, u, requires_grad=False)
        save_x(x_1)

    # PCA of 1-step reachable set.
    principal_points = calc_principal_points(qu_samples, r=0.5)
    principal_axes_plots = create_pca_plots(principal_points)

    # goal poses
    q_u_goal_samples = q_u0 + sample_on_sphere(radius=0.5, n_samples=1000)

    plot_qu0 = go.Scatter3d(x=[q_u0[0]],
                            y=[q_u0[1]],
                            z=[q_u0[2]],
                            name='q_u0',
                            mode='markers',
                            hovertemplate=hover_template_reachability,
                            marker=dict(size=12, symbol='cross', opacity=0.8))

    plot_1_step = go.Scatter3d(x=qu_samples[:, 0],
                               y=qu_samples[:, 1],
                               z=qu_samples[:, 2],
                               name='1_step',
                               mode='markers',
                               hovertemplate=hover_template_reachability,
                               marker=dict(size=2))

    plot_goals = go.Scatter3d(
        x=q_u_goal_samples[:, 0],
        y=q_u_goal_samples[:, 1],
        z=q_u_goal_samples[:, 2],
        name="goals",
        mode="markers",
        hovertemplate=hover_template_reachability,
        marker=dict(size=6, opacity=0.8, color='gray'))

    fig = go.Figure(
        data=[plot_1_step, plot_qu0, plot_goals] + principal_axes_plots,
        layout=layout)
    return fig, json.dumps({'qa_l': qa_l_samples.tolist(),
                            'qa_r': qa_r_samples.tolist()})


@app.callback(
    Output('hover-data', 'children'),
    Input('reachable-sets', 'hoverData'),
    [State('reachable-sets', 'figure'), State('q_a_samples', 'data')])
def hover_callback(hover_data, figure, q_a_samples_json):
    hover_data_json = json.dumps(hover_data, indent=2)
    if hover_data is None:
        return hover_data_json
    q_a_samples = json.loads(q_a_samples_json)

    point = hover_data['points'][0]
    idx_fig = point['curveNumber']
    name = figure['data'][idx_fig]['name']
    idx = point["pointNumber"]

    if name == '1_step':
        q_dict = {
            model_u: np.array([point['x'], point['y'], point['z']]),
            model_a_l: q_a_samples['qa_l'][idx],
            model_a_r: q_a_samples['qa_r'][idx]}
        q_sim_py.update_mbp_positions(q_dict)
        q_sim_py.draw_current_configuration()

    return hover_data_json


@app.callback(
    [Output('click-data', 'children'), Output('selected-goal', 'data')],
    Input('reachable-sets', 'clickData'),
    State('reachable-sets', 'figure'))
def click_callback(click_data, figure):
    if click_data is None:
        return json.dumps(click_data, indent=2), json.dumps(None)

    point = click_data['points'][0]
    idx_fig = point['curveNumber']
    name = figure['data'][idx_fig]['name']
    idx = point["pointNumber"]

    msg = ''
    if name == 'goals':
        msg = "selected goal\n"
        msg += 'y: {}\n'.format(point['x'])
        msg += 'z: {}\n'.format(point['y'])
        msg += 'theta: {}\n'.format(point['z'])

        X_WG = calc_X_WG(y=point['x'], z=point['y'], theta=point['z'])
        vis['goal'].set_transform(X_WG)

    return msg, json.dumps({'y': point['x'],
                            'z': point['y'],
                            'theta': point['z']})


@app.callback(
    Output('calc-trj-update', 'children'),
    Input('btn-calc-trj', 'n_clicks'),
    [State('selected-goal', 'data'), State('q_u0', 'data'),
     State('q_a0', 'data')])
def calc_trajectory(n_clicks, q_u_goal_json, q_u0_json, q_a0_json):
    if q_u_goal_json is None:
        return ''
    q_u_goal_dict = json.loads(q_u_goal_json)
    if q_u_goal_dict is None:
        return ''

    # Traj Opt------------------------------------------------------------
    q_u_goal = np.array([q_u_goal_dict['y'], q_u_goal_dict['z'],
                         q_u_goal_dict['theta']])
    x0, u0, q_u0 = get_x0_and_u0_from_json(q_u0_json, q_a0_json)
    x_goal = np.array(x0)
    x_goal[q_sim_py.velocity_indices[model_u]] = q_u_goal
    irs_lqr_q.initialize_problem(
        x0=x0,
        x_trj_d=np.tile(x_goal, (T + 1, 1)),
        u_trj_0=np.tile(u0, (T, 1)))

    irs_lqr_q.iterate(max_iterations=10)
    result = irs_lqr_q.package_solution()
    q_dynamics.publish_trajectory(result['x_trj'])
    # --------------------------------------------------------------------

    msg = 'Trying to reach goal\n'
    msg += 'y: {}\n'.format(q_u_goal[0])
    msg += 'z: {}\n'.format(q_u_goal[1])
    msg += 'theta: {}\n'.format(q_u_goal[2])
    msg += "cost: {}".format(result['cost']['Qu_f'])
    return msg


if __name__ == '__main__':
    app.run_server(debug=False)
