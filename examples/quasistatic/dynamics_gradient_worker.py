import multiprocessing
import time

from quasistatic_simulator.core.quasistatic_simulator import (
    QuasistaticSimParameters)
from quasistatic_simulator.core.quasistatic_system import (
    cpp_params_from_py_params)

try:
    from irs_lqr.quasistatic_dynamics import *
    from zmq_parallel_cmp.array_io import *
except ImportError:
    from irs_lqr.irs_lqr.quasistatic_dynamics import *
    from irs_lqr.zmq_parallel_cmp.array_io import *

# TODO: make workers system-agnostic. Maybe passsing in a yml file describing
#  the sytem from the command line?
from planar_hand_setup import *


def f_worker(lock: multiprocessing.Lock):
    context = zmq.Context()

    # Socket to receive messages on
    receiver = context.socket(zmq.PULL)
    receiver.connect("tcp://localhost:5557")

    # Socket to send messages to
    sender = context.socket(zmq.PUSH)
    sender.connect("tcp://localhost:5558")

    pid = multiprocessing.current_process().pid
    print("worker", pid, "ready.")

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
        x_u_nominal, t_list, n_samples, std = recv_array(receiver)
        assert len(x_u_nominal.shape) == 2
        x_nominals = x_u_nominal[:, :q_dynamics.dim_x]
        u_nominals = x_u_nominal[:, q_dynamics.dim_x:]

        ABhat = q_dynamics.calc_AB_first_order_batch(
            x_nominals=x_nominals,
            u_nominals=u_nominals,
            n_samples=n_samples,
            std_u=std)

        # Send results to sink
        send_array(sender, A=ABhat, t=t_list, n_samples=-1, std=[-1])

        i_tasks += 1
        if i_tasks % 10 == 0:
            lock.acquire()
            print(pid, "has processed", i_tasks, "tasks.")
            lock.release()


if __name__ == "__main__":
    p_list = []
    try:
        lock = multiprocessing.Lock()
        for _ in range(30):
            p = multiprocessing.Process(target=f_worker, args=(lock,))
            p_list.append(p)
            p.start()
        time.sleep(100000)
    except KeyboardInterrupt:
        for p in p_list:
            p.terminate()
            p.join()

