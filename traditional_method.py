"""
Name: traditional_method.py
Author: Nathan Erickson
EMail: nathan.erickson@student.nmt.edu
Description: Solves the 1D wave equation for a 1st harmonic standing wave 
			 numerically using the leapfrog method

"""






#Imports
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import ticker
import time




### PARAMETERS ###

#Parameters
max_x = 1
max_t = 2
inv_dx = 100
inv_dt = 100
c = 1

# Space Constraints
nx = int(max_x * inv_dx)
x = np.linspace(0, max_x, nx)
dx = 1/inv_dx

#Time constraints
nt = int(max_t * inv_dt)
t = np.linspace(0, max_t, nt)
dt = 1/inv_dt


#Initialize displacement array
u = np.zeros((nt, nx))
u[0] = np.sin(pi * x)

#Begin timing
start = time.time()


### NUMERICAL SOLUTION ###

#Before we run the loop the first time, we need to get 
#Velocity and acceleration

# These will be useful
inv_dx2 = inv_dx**2
c2 = c**2


#Get the initial velocity
vel = np.zeros(nx)

#First, now, find the accelration
accel = c2 * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]) * inv_dx2

#Find the i + dt/2 velocity (makes it leapfrog)
vel[1:-1] += accel * (dt/2)



#Iterate over the time range
for i in range(nt - 1):

	#Update u
	u[i+1] = u[i] + vel * dt

	#Now, find the accelration
	accel = c2 * (u[i+1, 2:] - 2*u[i+1, 1:-1] + u[i+1, :-2]) * inv_dx2

	#Find the next velocity
	vel[1:-1] += accel * dt





### VERIFICATION ### 

#Use u(x,t) = sin(pi*x)*cos(c*pi*t) for analytical solution
u_ana = np.zeros((nt, nx))
for i in range(nt):
	u_ana[i] = np.sin(pi*x) * np.cos(c*pi*dt*i)

#Compare Results
error = np.abs(u_ana - u)
mse = np.mean(error**2, axis=1)


### TIME ###
end = time.time()
runtime = end - start
print(f"\nThe simulation took {runtime:.3e} seconds to run.\n")





### PLOT RESULTS ###
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,6))
fig.suptitle("Numerical Solution")

#Numerical solution
im1 = ax1.imshow(u.T, extent=[0, max_t, max_x, 0], aspect='auto', cmap='RdBu')
ax1.set_title("Standing Wave Over time (Numerical Solution)")
ax1.set_xlabel("Time")
ax1.set_ylabel("Space")
fig.colorbar(im1, ax=ax1, label="Amplitude")

#Analytical Solution
im2 = ax2.imshow(u_ana.T, extent=[0, max_t, max_x, 0], aspect='auto', cmap='RdBu')
ax2.set_title("Standing Wave Over time (Analytical Solution)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Space")
fig.colorbar(im2, ax=ax2, label="Amplitude")

#Error
im3 = ax3.imshow(error.T, extent=[0, max_t, max_x, 0], aspect='auto', 
                 cmap='Reds')
ax3.set_title("Error")
ax3.set_xlabel("Time")
ax3.set_ylabel("Space")
# Add a colorbar with scientific notation for the tiny error values
cbar3 = fig.colorbar(im3, ax=ax3, label="Error Magnitude")
cbar3.ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
cbar3.ax.ticklabel_format(style='sci', scilimits=(0,0))

#Mean Squared Error
ax4.plot(t, mse)
ax4.set_title("Mean Squared Error")
ax4.set_ylabel("Error")
ax4.set_xlabel("Time")
ax4.set_yscale('log')

plt.tight_layout()
plt.show()

