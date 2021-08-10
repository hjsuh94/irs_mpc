import threading
import os

import zmq

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimulator, QuasistaticSimParameters)
from quasistatic_simulator_py import (QuasistaticSimulatorCpp,
    QuasistaticSimParametersCpp)
from quasistatic_simulator.core.quasistatic_system import (
    cpp_params_from_py_params)
from quasistatic_dynamics import *

from planar_hand_setup import *
from zmq_parallel_cmp.array_io import *


def f_worker():
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:5557")

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5558")

    thread_id = threading.current_thread().ident
    print("worker", thread_id, "ready.")

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
        internal_vis=False)

    sim_params_cpp = cpp_params_from_py_params(sim_params)
    sim_params_cpp.gradient_lstsq_tolerance = gradient_lstsq_tolerance

    q_sim_cpp = QuasistaticSimulatorCpp(
        model_directive_path=model_directive_path,
        robot_stiffness_str=robot_stiffness_dict,
        object_sdf_paths=object_sdf_dict,
        sim_params=sim_params_cpp)

    q_dynamics = QuasistaticDynamics(h=h, q_sim_py=q_sim_py, q_sim=q_sim_cpp)

    # Process tasks forever
    i_tasks = 0
    while True:
        x_u_nominal, t, n_samples, std = recv_array(receiver)
        x_nominal = x_u_nominal[:q_dynamics.dim_x]
        u_nominal = x_u_nominal[q_dynamics.dim_x:]

        Ahat, Bhat = q_dynamics.calc_AB_first_order(
            x_nominal=x_nominal,
            u_nominal=u_nominal,
            n_samples=n_samples,
            std=std)

        # Send results to sink
        send_array(sender, A=np.hstack([Ahat, Bhat]),
                   t=t, n_samples=n_samples, std=std)

        i_tasks += 1
        if i_tasks % 10 == 0:
            print(thread_id, "has processed", i_tasks, "tasks.")


if __name__ == "__main__":
    f_worker()
    # p_list = []
    #
    # for _ in range(5):
    #     p = Process(target=f_worker)
    #     p_list.append(p)
    #     p.start()
    #
    # for p in p_list:
    #     p.join()
