# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 19:34:33 2026

@author: aleja
"""

"Code for fitting the NFW profile from the radial density profile by guessing" 
"first the center of mass of the cluster, and obtaining the radial profile"

# Importing packages
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

# Closing plots
plt.close("all") 

# Cosmological Constants and conversion factors and other constants
h = 0.7
msun = 1.989e30 # kg
mass_conv_factor = 1e10 # Msun/h
G_gadget = 43007.1 # kpc (km/s)^2 (1e10 Msun)^-1
G_physical = G_gadget/1e10 # kpc (km/s)^2 (Msun)^-1
Hubble = 70/1000 # km s^-1 kpc^-1

# We import the snapshot files from the simulation
h5py_file = "C:\\Users\\aleja\\astrofisica_teorica\\sim_num\\output_ex5\\snapshot_005.hdf5"

with h5py.File(h5py_file, "r") as f:
    pos1 = np.asarray(f["/PartType1/Coordinates"]) # kpc/h
    print("---File Information---")
    print(list(f.keys()))
    print(list(f["PartType1"].keys()))
    header = f["Header"].attrs
    mass = header["MassTable"] # 1e10 Msun/h
    n_particles = header["NumPart_Total"][1]
    dm_mass = mass[1] # 1e10 Msun/h
    print("")
    print("---Attributes in Header---")
    for key in header.keys():
        value = header[key]
        print(f"{key}: {value}")
       
# Using PartType1 particles projections to guess the center of mass of the 
# cluster to retrieve the radial profile

# Conversion of the quantities to physical units
pos1 = pos1/h # kpc
mass_particle = dm_mass*mass_conv_factor*(1/h) # Msun
#mass_particle = mass_particle*msun # kg
particles_array = np.zeros(n_particles)
mass_particle_array = np.full_like(particles_array, mass_particle)

# Storing position values for plotting the XZ and YZ projections
xpos = np.zeros(len(pos1))
ypos = np.zeros(len(pos1))
zpos = np.zeros(len(pos1))

for ii in range(len(pos1)):
    xpos[ii] = pos1[ii][0]
    ypos[ii] = pos1[ii][1]
    zpos[ii] = pos1[ii][2]
    
# Plotting the XZ and YZ projections
plt.figure(1)
plt.scatter(xpos, zpos, color="lightgray", s=20, label="XZ projection", alpha=0.6, 
            edgecolors="black", linewidth=0.5)
#plt.scatter(ypos, zpos, color="red", s=2, label="YZ projection")
plt.tick_params(direction="in", which="major", length=4, labelsize=28, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=2, labelsize=28, top=True, right=True)
plt.xlabel('X [kpc]', fontsize=22)
plt.ylabel('Z [kpc]', fontsize=22)
plt.legend(prop={'size': 20})
plt.show()

plt.figure(2)
plt.scatter(ypos, zpos, color="lightgray", s=20, label="YZ projection", alpha=0.6,
            edgecolors="black", linewidth=0.5)
plt.tick_params(direction="in", which="major", length=4, labelsize=28, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=2, labelsize=28, top=True, right=True)
plt.xlabel('Y [kpc]', fontsize=22)
plt.ylabel('Z [kpc]', fontsize=22)
plt.legend(prop={'size': 20})
plt.show()

# Guessing the center from thr projections, we obtain an initial value of
# (-1600, -1600, -6000) to make a guess of the CoM
center = np.array([-1600, -1600, -6000]) # kpc
radius = 15000 # kpc

# Implement algorithm to calculate the CoM of the cluster using a sphere 
# centered on that guessed center to retrieve the radial profile
def com_algorithm(particle_mass, particle_mass_array, pos, center_guess, radius):
    d = pos - center_guess # array of position relative to sphere center
    radial = np.linalg.norm(d, axis=1) # radial distance for each particle
    mask = radial <= radius # keep particles inside the sphere
    total_mass = particle_mass_array[mask].sum() # total mass inside the sphere
    
    # Defining the CoM and calculating it
    numerator = particle_mass*pos[mask] # position vector times mass of particle
    num = numerator.sum(axis=0) # summing by column (per x, y or z position)
    com = num/total_mass
    return com

# Center of mass guess
com_guess = com_algorithm(mass_particle, mass_particle_array, pos1, center, radius)
print(f"\nCenter guess value: {com_guess} [kpc]")

# Center convergence algorithm to obtain afterwards the radial profile
def center_convergence(particle_mass, particle_mass_array, pos, center_guess, 
                   radius, shrink=0.95, tol=1e-3, niter=100):
    center = center_guess
    for ii in range(niter):
        new_center = com_algorithm(particle_mass, particle_mass_array, pos, center, radius)
        diff_center = np.linalg.norm(new_center - center)
        center = new_center
        radius *= shrink
    
        if diff_center < tol:
            break
    return center

# Return final convergence of the center
center_result = center_convergence(mass_particle, mass_particle_array, pos1, 
                                    center, radius)
print(f"\nCenter of cluster value: {center_result} [kpc]")

"Obtaining the radial density profile for the cluster"

# Generating firstly the density profile by using np.histogram
nbins = 30 
radial_output = np.linalg.norm(pos1 - center_result, axis=1)
weight_array = np.full_like(radial_output, mass_particle)
r_min, r_max = radial_output.min(), radial_output.max()
bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), nbins + 1)
hist, edges = np.histogram(radial_output, bins=bin_edges, range=(r_min, r_max), weights=weight_array)

# Calculating the volume density (remember bins are spherical shells)
# Lets calculate the volume 
radius_outer = edges[1:]
radius_inner = edges[:-1]
shell_volume = (4*np.pi/3)*(radius_outer**3 - radius_inner**3)
vol_density = hist/shell_volume
# Geometrical mean for the radial value because of logarithmic binning
bin_centers = np.sqrt(radius_inner*radius_outer)

# Masking the values for the fitting procedure
mask = vol_density > 0
bin_centers = bin_centers[mask]
vol_density = vol_density[mask]

# Fit the curve obtained to see the log-log slope 
log_r = np.log10(bin_centers)
log_rho = np.log10(vol_density)

# Printing the value of the slope with polyfit
#slope, intercept = np.polyfit(np.log10(bin_centers), np.log10(vol_density), 1)
#print(f"Power law index slope: {-slope:.2f}")

# Defining the critical density value
rho_crit = (3*(Hubble)**2)/(8*np.pi*G_physical) # Msun kpc^-3
print(f"\nCritical density value: {rho_crit:.3f} [Msun kpc^-3]")

# Fitting to the log NFW profile for a better convergence of the parameters
def log_NFW_profile(radius, log_delta_c, log_scale_radius):
    delta_c = 10**(log_delta_c)
    scale_radius = 10**(log_scale_radius)
    denominator = (radius/scale_radius)*(1 + radius/scale_radius)**2
    return np.log10((rho_crit*delta_c)/(denominator))

# Obtaining the parameter values of delta_c and the scale radius
p0 = [3, 3]
popt, pcov = curve_fit(log_NFW_profile, bin_centers, log_rho, p0=p0, maxfev=10000)
delta_c = 10**(popt[0])
scale_radius = 10**(popt[1]) # kpc
print(f"\nValues of the parameters:\ndelta_c = {delta_c:.3f}, r_s = {scale_radius:.3f} [kpc]")

# Visualizing the result
plt.figure(3)
plt.loglog(bin_centers, vol_density, marker="o", linestyle="None")
plt.tick_params(direction="in", which="major", length=4, labelsize=28, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=2, labelsize=28, top=True, right=True)
plt.xlabel(r"$\log_{10}(r)$ [kpc]", fontsize=26)
plt.ylabel(r"$\log_{10}(\rho$) $[\rm{M_{\odot}} \, kpc^{-3}]$", fontsize=26)
plt.legend(prop={'size': 20})
plt.text(x=3, y=1e1, s=rf"$\delta_{{\rm{{c}}}}$ = {delta_c:.2f}", fontsize=24,
         bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", 
                   alpha=0.8))
plt.text(x=3, y=1e0, s=rf"$r_{{\mathrm{{s}}}}$ = {scale_radius:.2f} [$\mathrm{{kpc}}$]", fontsize=24,
         bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor="black", 
                   alpha=0.8))
plt.title("Radial Density Profile of Cluster", fontsize=26)
plt.show()