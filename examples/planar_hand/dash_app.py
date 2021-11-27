import json
import pickle
import dash
from dash import dcc, html

from dash.dependencies import Input, Output
import plotly.graph_objects as go

from qsim.simulator import (QuasistaticSimulator, QuasistaticSimParameters)
from planar_hand_setup import (model_directive_path,
                               robot_stiffness_dict, object_sdf_dict,
                               robot_l_name, robot_r_name, object_name)

from rrt.utils import set_orthographic_camera_yz

# %% meshcat
q_sim_py = QuasistaticSimulator(
    model_directive_path=model_directive_path,
    robot_stiffness_dict=robot_stiffness_dict,
    object_sdf_paths=object_sdf_dict,
    sim_params=QuasistaticSimParameters(),
    internal_vis=True)

set_orthographic_camera_yz(q_sim_py.viz.vis)

model_a_l = q_sim_py.plant.GetModelInstanceByName(robot_l_name)
model_a_r = q_sim_py.plant.GetModelInstanceByName(robot_r_name)
model_u = q_sim_py.plant.GetModelInstanceByName(object_name)

# %% load data from disk.
data_file_suffix = '_r0.2'
with open(f"du_{data_file_suffix}.pkl", 'rb') as f:
    du = pickle.load(f)

with open(f"qa_l_{data_file_suffix}.pkl", 'rb') as f:
    qa_l = pickle.load(f)

with open(f"qa_r_{data_file_suffix}.pkl", 'rb') as f:
    qa_r = pickle.load(f)

with open(f"qu_{data_file_suffix}.pkl", 'rb') as f:
    qu = pickle.load(f)

# %%
layout = go.Layout(scene=dict(aspectmode='data'), height=1000)
data_1_step = go.Scatter3d(x=qu['1_step'][:, 0],
                           y=qu['1_step'][:, 1],
                           z=qu['1_step'][:, 2],
                           mode='markers',
                           marker=dict(color=0x00ff00,
                                       size=1.5,
                                       sizemode='diameter'))
data_multi = go.Scatter3d(x=qu['multi_step'][:, 0],
                          y=qu['multi_step'][:, 1],
                          z=qu['multi_step'][:, 2],
                          mode='markers',
                          marker=dict(color=0x00ff00,
                                      size=1.5,
                                      sizemode='diameter'))

fig = go.Figure(data=[data_1_step, data_multi],
                layout=layout)
fig.update_scenes(camera_projection_type='orthographic',
                  xaxis_title_text='y',
                  yaxis_title_text='z',
                  zaxis_title_text='theta')

# %%
app = dash.Dash(__name__)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Hover Data**

                Mouse over values in the graph.
            """),
            html.Pre(id='hover-data', style=styles['pre'])
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Selection Data**

                Choose the lasso or rectangle tool in the graph's menu
                bar and then select points in the graph.

                Note that if `layout.clickmode = 'event+select'`, selection data also
                accumulates (or un-accumulates) selected data if you hold down the shift
                button while clicking.
            """),
            html.Pre(id='selected-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Zoom and Relayout Data**

                Click and drag on the graph to zoom or click on the zoom
                buttons in the graph's menu bar.
                Clicking on legend items will also fire
                this event.
            """),
            html.Pre(id='relayout-data', style=styles['pre']),
        ], className='three columns')
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    Input('basic-interactions', 'hoverData'))
def display_hover_data(hoverData):
    if hoverData is None:
        return json.dumps(hoverData, indent=2)
    point = hoverData['points'][0]
    if point['curveNumber'] == 1:
        name = "multi_step"
    elif point['curveNumber'] == 0:
        name = "1_step"

    idx = point["pointNumber"]
    q_dict = {
        model_u: qu[name][idx],
        model_a_l: qa_l[name][idx],
        model_a_r: qa_r[name][idx]}

    q_sim_py.update_mbp_positions(q_dict)
    q_sim_py.draw_current_configuration()

    return json.dumps(hoverData, indent=2)


@app.callback(
    Output('click-data', 'children'),
    Input('basic-interactions', 'clickData'))
def display_click_data(clickData):
    if clickData is None:
        return json.dumps(clickData, indent=2)
    point = clickData['points'][0]
    if point['curveNumber'] == 1:
        name = "1_step"
    elif point['curveNumber'] == 0:
        name = "multi_step"

    idx = point["pointNumber"]
    q_dict = {
        model_u: qu[name][idx],
        model_a_l: qa_l[name][idx],
        model_a_r: qa_r[name][idx]}

    q_sim_py.update_mbp_positions(q_dict)
    q_sim_py.draw_current_configuration()

    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    Input('basic-interactions', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)


@app.callback(
    Output('relayout-data', 'children'),
    Input('basic-interactions', 'relayoutData'))
def display_relayout_data(relayoutData):
    return json.dumps(relayoutData, indent=2)


if __name__ == '__main__':
    app.run_server(debug=True)
