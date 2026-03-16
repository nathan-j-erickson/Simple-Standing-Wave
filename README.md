# PHYS 2425 Project 1

Author: Nathan Erickson

This project models a 1-dimensional standing wave using both a traditional numerical solution and a physics-informed neural network.



## Requirements

- Python 3.13.5 (Any version of Python 3 should work)
- NumPy
- MatPlotLib
- PyTorch
You should be able to install all dependincies using `pip install -r requirements.txt`


## Files

- `traditional_method.py`: models a 1-dimensional 1st harmonic standing wave using the leapfrog method.
- `PINN_method.py`: models a 1-dimensional 1st harmonic standing wave using a physics-informed neural network.
- `traditional_method_2nd_harmonic.py`: models a 1-dimensional 2nd harmonic standing wave using the leapfrog method.


## Instructions

I ran these using IPython from a terminal, but any other method to run python files should work just fine. The scripts do not require any manual input. 

### Output
Each will output a graph that shows a plot of the spacial displacement ( $u(x,t)$ ) of the wave at various time and space coordiantes, with the analytical solution for comparison. In addition, a plot of the error at each time and space point is shown as well as the mean squared error over time. The `PINN_method.py` additionally features plots of it's error during training which will show before the solution plot.


## Theory

This is the governing equation for the wave:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

I used the following for the analytical solution:

$$u(x,t) = \sin(\pi x)\cos(c\pi t)$$

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
