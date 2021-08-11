# Planar Hand Ball Manipulation Example

1. Launch gradient works (should work from any directory):
```bash
python3 [path-to-irs_lqr]/examples/quasistatic/dynamics_gradient_worker.py
```
The number of workers is hard coded in the `__main__` part of the script.


2. Run `run_planar_hand.py` with `meshcat-server` open in the background.

THe script publishes the trajectory to `meshcat` and plots different components of the cost as a function of the number of iterations.