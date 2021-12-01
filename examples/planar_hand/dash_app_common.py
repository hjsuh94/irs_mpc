import numpy as np
import meshcat


#%% meshcat
X_WG0 = meshcat.transformations.rotation_matrix(np.pi/2, [0, 0, 1])


def add_goal(vis: meshcat.Visualizer):
    # goal
    vis["goal/cylinder"].set_object(
        meshcat.geometry.Cylinder(height=0.001, radius=0.25),
        meshcat.geometry.MeshLambertMaterial(color=0xdeb948, reflectivity=0.8))
    vis['goal/box'].set_object(
        meshcat.geometry.Box([0.02, 0.005, 0.25]),
        meshcat.geometry.MeshLambertMaterial(color=0x00ff00, reflectivity=0.8))
    vis['goal/box'].set_transform(
        meshcat.transformations.translation_matrix([0, 0, 0.125]))

    # rotate cylinder so that it faces the x-axis.
    vis['goal'].set_transform(X_WG0)


#%% hover template
hover_template_1step = (
        '<i>y</i>: %{x:.4f}<br>' +
        '<i>z</i>: %{y:.4f}<br>' +
        '<i>theta</i>: %{z:.4f}')

hover_template_trj = hover_template_1step + '<br><i>cost</i>: %{marker.color:.4f}'

