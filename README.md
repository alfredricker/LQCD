# LQCD
This repository includes my Python code for the heat bath algorithm in the theory of Lattice Quantum Chromodynamics. 
For a theoretical summary of the contents within these files, you may view the pdf included on the LQCD research section of my website, **alfredricker.com**.

I will keep my summary of each file brief because the code is commented and you may always reach out to me with any questions at alfred.ricker7@gmail.com.
**su3.py** contains all relevant functions to run the heat bath or Metropolis algorithm for an SU(3) lattice (likewise for su2.py). Note that several of these functions are not used in my code, such as reunitarize, calcUi, and calcUt. 
**su3maps.py** contains functions corresponding to the SU(3) parameterizations derived in the MacFarlane paper (referenced in the pdf on my website). The "S" vector is the 9-dimensional vector of the scalar component and the coefficients of the Gell-Mann matrices, $S = (b_0,\vec{b})$. The Cayley map works just fine, however, I was unable to get the exponential map to consistently produce SU(3) vectors. This file is not likely to be useful because the multiplication law of SU(3) vectors is done in terms of $d$ and $f$ tensors, which takes (a lot) more computation time than ordinary matrix multiplication. See the section in my paper on SU(3) representation theory for a more in-depth explanation of these concerns.

**plaqvupdate.py** contains the code to execute the SU(3) Metropolis algorithm and produces a graph that illustrates the convergence of the system for a given $\beta$. I will not spend any time reviewing this algorithm here, but it may be read about in Creutz et. al.
**heatbathsu2.py** is the original heat bath code that I had written, which calculates average plaquettes for the simple case of an SU(2) lattice. This code is the foundation of the SU(3) code, as the SU(3) heat bath is a generalization of the SU(2) algorithm.
**heatbath.py** is the code that executes the heat bath algorithm for an SU(3) lattice with $m$ iterations. As of now the code outputs pdf files with static information (you have to change the code every time you want to change the file name, title, etc). You may use a command line argument to input beta or input when prompted.
