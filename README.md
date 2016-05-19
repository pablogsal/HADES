# HADES
![alt text](/results/Polarization_map.png?raw=true "Polarization map")
# What is HADES?

HADES is a massively-parallel synchrotron emission raytracer for relativistic jets.

# What is HADES ( using non-technical jargon )

HADES is a software to simulate the synchrotron emission that emmanates from relativistic jets that work really fast. This last claim is possible thanks to the use of NVIDIA's GPU technology: CUDA. This LLVM-based framework allows the use of GPUs to make the hard calculations that are needed for the synchrotron simulations. GPUs are **really** fast when performing parallel taks (10-100 times more faster faster than CPUs) which makes this approach performs specially good when
constructing raytracers as this one.

# How HADES works?

HADES uses RMHD data to construct thermal synchrotron emission models that are used to simulate the jet emission properties. When HADES knows the properties of the jet emission, then starts to lauch thousands of light rays going to the jet until it reaches a simulated receptor. This receptor can be thought as a CCD receptor (as in your digital camera), which is a matrix of pixels that will form the image. When the ray reach the simulated receptor, certain properties of the emission are recorded
in the "pixel" in which the ray ends. When every ray has end its path, the polarization and flux maps are constructed using the Stoke's parameters information of the emission.

# Why is this so great?

When we know the inner structure and properties of a relativistic jet (such as its temperature, magnetic field and pressure distribution) it's very usefull to know how this jet can be seen from the earth with our detectors. But we need much more than a simple "photo", we need to know the polarization of the synchrotron radiation because this allow us to construct reverse maps that can be used to deduce the magnetic properties of the jets that we observe using real detectors.

# How can HADES be used

( In preparation )
