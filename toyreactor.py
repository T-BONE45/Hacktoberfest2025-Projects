"""
Fictional "Arc Reactor" Simulator (SAFE, educational, and purely fictional)

- Models a toy reactor core with:
    core_temp: rises with generated power, reduced by cooling
    power_output: function of core_temp up to a fictional max
    pid controller: adjusts "cooling_power" to keep core_temp near setpoint

This is a simulation for learning control loops and plotting dynamics.
NOTHING in this code corresponds to real reactor physics or control systems.
Do NOT use for real hardware.

Dependencies: numpy, matplotlib
Run: python arc_reactor_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Simulation parameters (entirely abstract units) ---
dt = 0.1                 # time step (s)
t_final = 200.0          # total simulation time (s)
times = np.arange(0, t_final, dt)

# Reactor parameters (fictional)
ambient_temp = 20.0           # ambient temperature (units)
thermal_capacity = 100.0      # how much energy raises core temp (units)
passive_cooling_coeff = 0.02  # base heat loss rate (1/s)
power_gain = 0.8              # how core_temp increases per unit internal power
max_fictional_power = 100.0   # max simulated power output (arbitrary units)

# PID controller gains (tune for behavior)
Kp = 5.0
Ki = 0.2
Kd = 1.0

# Setpoint schedule (desired core temperature)
def desired_temperature(t):
    # step up at t=30s, step down at t=120s
    if t < 30:
        return 30.0
    elif t < 120:
        return 60.0
    else:
        return 45.0

# Initialize state variables
core_temp = 30.0
integral = 0.0
prev_error = 0.0

# Data logging
log_temp = []
log_power = []
log_cooling = []
log_setpoint = []

for t in times:
    # desired temperature
    setpoint = desired_temperature(t)
    error = setpoint - core_temp
    integral += error * dt
    derivative = (error - prev_error) / dt if dt > 0 else 0.0
    prev_error = error

    # PID controller outputs a cooling action (fictional)
    cooling_power = Kp * error + Ki * integral + Kd * derivative
    # Bound cooling_power to [0, 200] (fictional)
    cooling_power = max(0.0, min(200.0, cooling_power))

    # Internal generated power is some function of how far core_temp is above ambient
    # (purely fictional mapping)
    internal_power = max_fictional_power * (1 - np.exp(-0.05 * max(core_temp - ambient_temp, 0)))

    # Temperature dynamics (very simplified and fictional)
    # heat_increase from internal_power, heat_loss from cooling and passive cooling
    heat_in = power_gain * internal_power * dt / thermal_capacity
    heat_loss = (passive_cooling_coeff * (core_temp - ambient_temp) + 0.01 * cooling_power) * dt
    core_temp += heat_in - heat_loss

    # log
    log_temp.append(core_temp)
    log_power.append(internal_power)
    log_cooling.append(cooling_power)
    log_setpoint.append(setpoint)

# Plot results
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(times, log_temp, label="Core Temp (fictional)")
plt.plot(times, log_setpoint, '--', label="Setpoint")
plt.ylabel("Temperature (arb. units)")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(times, log_power, label="Simulated Power Output")
plt.plot(times, log_cooling, label="Cooling (PID output)")
plt.ylabel("Power / Cooling (arb. units)")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)

plt.suptitle("Fictional Arc Reactor Simulation (SAFE & Educational)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
