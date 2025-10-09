# SpinSpec
A toolkit for researchers to simulate, analyze, and explore EPR systems efficiently.

🚀 Overview
This project provides a complete toolkit for:
Quantum Spin System Modeling: Multi-particle spin systems with electrons and nuclei
EPR Spectroscopy Simulation: Continuous-wave and time-resolved EPR spectra
Resonance Field Calculations: Advanced algorithms for finding resonance conditions
Machine Learning Integration: Neural network-based spectra generation and analysis
Optimization Framework: Parameter fitting using multiple optimization algorithms

import spin_system
import particles

# Create a spin system with electron and nucleus
electron = particles.Electron(g_tensor=torch.tensor([[2.002, 0, 0], 
                                                     [0, 2.002, 0], 
                                                     [0, 0, 2.002]]))
nucleus = particles.Nucleus(spin=1))
system = spin_system.SpinSystem([electron, nucleus])


