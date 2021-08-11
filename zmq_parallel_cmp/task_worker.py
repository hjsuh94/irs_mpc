# Task worker
# Connects PULL socket to tcp://localhost:5557
# Collects workloads from ventilator via that socket
# Connects PUSH socket to tcp://localhost:5558
# Sends results to sink via that socket
#
# Author: Lev Givon <lev(at)columbia(dot)edu>

import threading
from multiprocessing import Process
import sys
import time

from array_io import *


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

    # Process tasks forever
    i_tasks = 0
    while True:
        A, t, n_samples, std = recv_array(receiver)
        time.sleep(n_samples / 1000)
        # Send results to sink
        send_array(sender, A * 2, t=t, n_samples=n_samples, std=std)

        i_tasks += 1
        if i_tasks % 10 == 0:
            print(thread_id, "has processed", i_tasks, "tasks.")


if __name__ == "__main__":
    p_list = []

    for _ in range(5):
        p = Process(target=f_worker)
        p_list.append(p)
        p.start()

    for p in p_list:
        p.join()
