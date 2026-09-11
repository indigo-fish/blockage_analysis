#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd


# In[2]:


def find_nearest_height(Z, target_height):
    # Calculate the absolute difference between each element in Z and the target height
    abs_diff = np.abs(np.array(Z) - target_height)

    # Find the index of the minimum absolute difference
    nearest_index = np.unravel_index(np.argmin(abs_diff), Z.shape)

    return nearest_index


# In[3]:


neutral_turbine_10 = xr.open_dataset(Path("processed") / "neutral_10ms" / "turbine_time_average.nc")
neutral_nwf_10 = xr.open_dataset(Path("processed") / "neutral_10ms" / "noturbine_time_average.nc")
stable_turbine_10 = xr.open_dataset(Path("processed") / "stable_10ms" / "turbine_time_average.nc")
stable_nwf_10 = xr.open_dataset(Path("processed") / "stable_10ms" / "noturbine_time_average.nc")
unstable_turbine_10 = xr.open_dataset(Path("processed") / "unstable_10ms" / "turbine_time_average.nc")
unstable_nwf_10 = xr.open_dataset(Path("processed") / "unstable_10ms" / "noturbine_time_average.nc")

neutral_turbine_8 = xr.open_dataset(Path("processed") / "neutral_8ms" / "turbine_time_average.nc")
neutral_nwf_8 = xr.open_dataset(Path("processed") / "neutral_8ms" / "noturbine_time_average.nc")
stable_turbine_8 = xr.open_dataset(Path("processed") / "stable_8ms" / "turbine_time_average.nc")
stable_nwf_8 = xr.open_dataset(Path("processed") / "stable_8ms" / "noturbine_time_average.nc")

stable_turbine_6 = xr.open_dataset(Path("processed") / "stable_6ms" / "turbine_time_average.nc")
stable_nwf_6 = xr.open_dataset(Path("processed") / "stable_6ms" / "noturbine_time_average.nc")

stable_turbine_12 = xr.open_dataset(Path("processed") / "stable_12ms" / "turbine_time_average.nc")
stable_nwf_12 = xr.open_dataset(Path("processed") / "stable_12ms" / "noturbine_time_average.nc")

stable_turbine_16 = xr.open_dataset(Path("processed") / "stable_16ms" / "turbine_time_average.nc")
stable_nwf_16 = xr.open_dataset(Path("processed") / "stable_16ms" / "noturbine_time_average.nc")

dtu_neutral_turbine_10 = xr.open_dataset(Path("processed") / "neutral_dtu10" / "turbine_time_average.nc")
dtu_stable_turbine_10 = xr.open_dataset(Path("processed") / "stable_dtu10" / "turbine_time_average.nc")

full_dict = {"stable_10": {"turbine": stable_turbine_10, "nwt": stable_nwf_10},
             "neutral_10": {"turbine": neutral_turbine_10, "nwt": neutral_nwf_10},
            "stable_8": {"turbine": stable_turbine_8, "nwt": stable_nwf_8},
            "neutral_8": {"turbine": neutral_turbine_8, "nwt": neutral_nwf_8},
            "stable_6": {"turbine": stable_turbine_6, "nwt": stable_nwf_6},
            "stable_12": {"turbine": stable_turbine_12, "nwt": stable_nwf_12},
            "stable_16": {"turbine": stable_turbine_16, "nwt": stable_nwf_16},
            "unstable_10": {"turbine": unstable_turbine_10, "nwt": unstable_nwf_10},
            "neutral_DTU_10": {"turbine": dtu_neutral_turbine_10, "nwt": neutral_nwf_10},
            "stable_DTU_10": {"turbine": dtu_stable_turbine_10, "nwt": stable_nwf_10}}


# In[4]:


dx = 10 # [m]
hub_height = 90 # [m]
dtu_hub_height = 119 # [m]
rotor_diameter = 127 # [m]

hub_index = find_nearest_height(neutral_nwf_10["Z"], 90)[0]
dtu_hub_index = find_nearest_height(neutral_nwf_10["Z"], 119)[0]
print(hub_index)
print(dtu_hub_index)
lower_index = find_nearest_height(neutral_nwf_10["Z"], 90 - rotor_diameter / 2)[0]
upper_index = find_nearest_height(neutral_nwf_10["Z"], 90 + rotor_diameter / 2)[0]
print(lower_index, upper_index)


# In[5]:


stable_turbine_10


# In[6]:


plt.plot(neutral_nwf_10["V2"][17, 251, :], label='2.8 MW hub height')
plt.plot(neutral_nwf_10["V2"][23, 251, :], label='10 MW hub height')
plt.legend()


# In[7]:


x = np.arange(-250, 1) * 10
# --- Secondary axis ---
def meters_to_D(x):
    return x / rotor_diameter


def D_to_meters(x):
    return x * rotor_diameter

fig, ax = plt.subplots()
labels = {"stable_10": "stable 10 m/s",
          "unstable_10": "unstable 10 m/s",
          "neutral_10": "neutral 10 m/s",
          "stable_16": "stable 16 m/s",
          "stable_6": "stable 6 m/s",
          "stable_DTU_10": "stable 10 m/s DTU 10MW"}
# for i, key in enumerate(["stable_10", "unstable_10", "neutral_10", "stable_16", "stable_6", "stable_DTU_10"]):
for i, key in enumerate(["stable_10"]):
    turb_data = full_dict[key]["turbine"]
    nwf_data = full_dict[key]["nwt"]
    if key != "stable_DTU_10":
        diff = turb_data["V2"][17, 251, :251] - nwf_data["V2"][17, 251, :251]
        std = (turb_data['V2_std'][17, 251, :251]**2 + nwf_data['V2_std'][17, 251, :251]**2)**.5
    else:
        diff = turb_data["V2"][23, 251, :251] - nwf_data["V2"][23, 251, :251]
        std = (turb_data['V2_std'][23, 251, :251]**2 + nwf_data['V2_std'][23, 251, :251]**2)**.5
    std_err = std / np.sqrt(360)
    ax.plot(x, diff, label=labels[key])
    ax.fill_between(
        x,
        diff - std_err,
        diff + std_err,
        alpha=0.2,
        label=f'±1 Std Err'
    )

ax.legend(bbox_to_anchor=[0.02, 0.5], loc='upper left')
ax.set_xticks(np.linspace(-2000, 0, 5))
ax.set_xlabel('X (m)')
ax.set_ylabel('Wind speed deficit (m/s)')

secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
secax.set_xlabel(r'X (Rotor Diameters, $D$)')
secax.set_xticks(np.linspace(-16, 0, 5))

ax.set_xlim(-2000, 0)
ax.set_ylim(-2.5, 0.5)
plt.savefig('Figures/axial_induction/stable_10ms.png', dpi=200)
fig.show()


# In[8]:


x = np.arange(-250, 1) * 10
# --- Secondary axis ---
def meters_to_D(x):
    return x / rotor_diameter


def D_to_meters(x):
    return x * rotor_diameter

fig, ax = plt.subplots()
labels = {"stable_10": "stable 10 m/s",
          "unstable_10": "unstable 10 m/s",
          "neutral_10": "neutral 10 m/s",
          "stable_16": "stable 16 m/s",
          "stable_6": "stable 6 m/s",
          "stable_DTU_10": "stable 10 m/s DTU 10MW"}
# for i, key in enumerate(["stable_10", "unstable_10", "neutral_10", "stable_16", "stable_6", "stable_DTU_10"]):
for i, key in enumerate(["stable_10", "unstable_10", "neutral_10"]):
    turb_data = full_dict[key]["turbine"]
    nwf_data = full_dict[key]["nwt"]
    if key != "stable_DTU_10":
        diff = turb_data["V2"][17, 251, :251] - nwf_data["V2"][17, 251, :251]
        std = (turb_data['V2_std'][17, 251, :251]**2 + nwf_data['V2_std'][17, 251, :251]**2)**.5
    else:
        diff = turb_data["V2"][23, 251, :251] - nwf_data["V2"][23, 251, :251]
        std = (turb_data['V2_std'][23, 251, :251]**2 + nwf_data['V2_std'][23, 251, :251]**2)**.5
    std_err = std / np.sqrt(360)
    ax.plot(x, diff, label=labels[key])
    ax.fill_between(
        x,
        diff - std_err,
        diff + std_err,
        alpha=0.2,
        label=f'± Std Err'
    )

ax.legend(bbox_to_anchor=[0.02, 0.5], loc='upper left')
ax.set_xticks(np.linspace(-2000, 0, 5))
ax.set_xlabel('X (m)')
ax.set_ylabel('Wind speed deficit (m/s)')

secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
secax.set_xlabel(r'X (Rotor Diameters, $D$)')
secax.set_xticks(np.linspace(-16, 0, 5))

ax.set_xlim(-2000, 0)
ax.set_ylim(-2.5, 0.5)
plt.savefig('Figures/axial_induction/varied_stabilities_10ms.png', dpi=200)
fig.show()


# In[9]:


x = np.arange(-250, 1) * 10
# --- Secondary axis ---
def meters_to_D(x):
    return x / rotor_diameter


def D_to_meters(x):
    return x * rotor_diameter

fig, ax = plt.subplots()
labels = {"stable_10": "stable 10 m/s",
          "unstable_10": "unstable 10 m/s",
          "neutral_10": "neutral 10 m/s",
          "stable_16": "stable 16 m/s",
          "stable_6": "stable 6 m/s",
          "stable_DTU_10": "stable 10 m/s DTU 10MW"}
# for i, key in enumerate(["stable_10", "unstable_10", "neutral_10", "stable_16", "stable_6", "stable_DTU_10"]):
for i, key in enumerate(["stable_10", "unstable_10", "neutral_10"]):
    turb_data = full_dict[key]["turbine"]
    nwf_data = full_dict[key]["nwt"]
    if key != "stable_DTU_10":
        diff = turb_data["V2"][17, 251, :251] - nwf_data["V2"][17, 251, :251]
        std = (turb_data['V2_std'][17, 251, :251]**2 + nwf_data['V2_std'][17, 251, :251]**2)**.5
    else:
        diff = turb_data["V2"][23, 251, :251] - nwf_data["V2"][23, 251, :251]
        std = (turb_data['V2_std'][23, 251, :251]**2 + nwf_data['V2_std'][23, 251, :251]**2)**.5
    std_err = std / np.sqrt(360)
    ax.plot(x, diff / nwf_data["V2"][23, 251, :251], label=labels[key])
    ax.fill_between(
        x,
        (diff - std_err) / nwf_data["V2"][23, 251, :251],
        (diff + std_err) / nwf_data["V2"][23, 251, :251],
        alpha=0.2,
        label=f'± Std Err'
    )

ax.legend(bbox_to_anchor=[0.02, 0.5], loc='upper left')
ax.set_xticks(np.linspace(-2000, 0, 5))
ax.set_xlabel('X (m)')
ax.set_ylabel('Normalized wind speed deficit')

secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
secax.set_xlabel(r'X (Rotor Diameters, $D$)')
secax.set_xticks(np.linspace(-16, 0, 5))

ax.set_xlim(-2000, 0)
ax.set_ylim(-0.3, 0.1)
plt.savefig('Figures/axial_induction/varied_stabilities_normalized_10ms.png', dpi=200)
fig.show()


# In[10]:


x = np.arange(-250, 1) * 10
# --- Secondary axis ---
def meters_to_D(x):
    return x / rotor_diameter


def D_to_meters(x):
    return x * rotor_diameter

fig, ax = plt.subplots()
labels = {"stable_10": "stable 10 m/s",
          "unstable_10": "unstable 10 m/s",
          "neutral_10": "neutral 10 m/s",
          "stable_16": "stable 16 m/s",
          "stable_6": "stable 6 m/s",
          "stable_DTU_10": "stable 10 m/s DTU 10MW"}
# for i, key in enumerate(["stable_10", "unstable_10", "neutral_10", "stable_16", "stable_6", "stable_DTU_10"]):
for i, key in enumerate(["stable_10", "stable_6", "stable_16"]):
    turb_data = full_dict[key]["turbine"]
    nwf_data = full_dict[key]["nwt"]
    if key != "stable_DTU_10":
        diff = turb_data["V2"][17, 251, :251] - nwf_data["V2"][17, 251, :251]
        std = (turb_data['V2_std'][17, 251, :251]**2 + nwf_data['V2_std'][17, 251, :251]**2)**.5
    else:
        diff = turb_data["V2"][23, 251, :251] - nwf_data["V2"][23, 251, :251]
        std = (turb_data['V2_std'][23, 251, :251]**2 + nwf_data['V2_std'][23, 251, :251]**2)**.5
    std_err = std / np.sqrt(360)
    ax.plot(x, diff / nwf_data["V2"][23, 251, :251], label=labels[key])
    ax.fill_between(
        x,
        (diff - std_err) / nwf_data["V2"][23, 251, :251],
        (diff + std_err) / nwf_data["V2"][23, 251, :251],
        alpha=0.2,
        label=f'± Std Err'
    )

ax.legend(bbox_to_anchor=[0.02, 0.5], loc='upper left')
ax.set_xticks(np.linspace(-2000, 0, 5))
ax.set_xlabel('X (m)')
ax.set_ylabel('Normalized wind speed deficit')

secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
secax.set_xlabel(r'X (Rotor Diameters, $D$)')
secax.set_xticks(np.linspace(-16, 0, 5))

ax.set_xlim(-2000, 0)
ax.set_ylim(-0.3, 0.1)
plt.savefig('Figures/axial_induction/varied_windspeeds_normalized.png', dpi=200)
fig.show()


# In[11]:


x = np.arange(-250, 1) * 10
# --- Secondary axis ---
def meters_to_D(x):
    return x / rotor_diameter


def D_to_meters(x):
    return x * rotor_diameter

fig, ax = plt.subplots()
labels = {"stable_10": "stable 10 m/s NREL 2.8MW",
          "unstable_10": "unstable 10 m/s",
          "neutral_10": "neutral 10 m/s",
          "stable_16": "stable 16 m/s",
          "stable_6": "stable 6 m/s",
          "stable_DTU_10": "stable 10 m/s DTU 10MW"}
# for i, key in enumerate(["stable_10", "unstable_10", "neutral_10", "stable_16", "stable_6", "stable_DTU_10"]):
for i, key in enumerate(["stable_10", "stable_DTU_10"]):
    turb_data = full_dict[key]["turbine"]
    nwf_data = full_dict[key]["nwt"]
    if key != "stable_DTU_10":
        diff = turb_data["V2"][17, 251, :251] - nwf_data["V2"][17, 251, :251]
        std = (turb_data['V2_std'][17, 251, :251]**2 + nwf_data['V2_std'][17, 251, :251]**2)**.5
    else:
        diff = turb_data["V2"][23, 251, :251] - nwf_data["V2"][23, 251, :251]
        std = (turb_data['V2_std'][23, 251, :251]**2 + nwf_data['V2_std'][23, 251, :251]**2)**.5
    std_err = std / np.sqrt(360)
    ax.plot(x, diff / nwf_data["V2"][23, 251, :251], label=labels[key])
    ax.fill_between(
        x,
        (diff - std_err) / nwf_data["V2"][23, 251, :251],
        (diff + std_err) / nwf_data["V2"][23, 251, :251],
        alpha=0.2,
        label=f'± Std Err'
    )

ax.legend(bbox_to_anchor=[0.02, 0.5], loc='upper left')
ax.set_xticks(np.linspace(-2000, 0, 5))
ax.set_xlabel('X (m)')
ax.set_ylabel('Normalized wind speed deficit')

secax = ax.secondary_xaxis('top', functions=(meters_to_D, D_to_meters))
secax.set_xlabel(r'X (Rotor Diameters, $D$)')
secax.set_xticks(np.linspace(-16, 0, 5))

ax.set_xlim(-2000, 0)
ax.set_ylim(-0.3, 0.1)
plt.savefig('Figures/axial_induction/varied_turbines_normalized.png', dpi=200)
fig.show()

