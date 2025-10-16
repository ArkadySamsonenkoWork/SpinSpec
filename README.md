MarS

A toolkit for researchers to simulate, analyze, and explore EPR systems efficiently.

🚀 Overview

MarS is a comprehensive toolkit designed for modeling, simulation, and analysis of Electron Paramagnetic Resonance (EPR) systems.
It enables researchers to efficiently study quantum spin dynamics, generate spectra, and integrate machine learning methods into EPR analysis.

Core Capabilities

-Quantum Spin System Modeling
Simulate multi-particle spin systems involving both electrons and nuclei.

-EPR Spectroscopy Simulation
Generate continuous-wave (CW) and time-resolved EPR spectra.

-Resonance Field Calculations
Compute resonance conditions using advanced numerical algorithms.

-Machine Learning Integration
Employ neural networks for spectra generation, prediction, and feature extraction.

-Optimization Framework
Fit simulation parameters using state-of-the-art optimization libraries like Optuna and Nevergrad.

⚙️ Features

🧠 General Capabilities

-Construction of Hamiltonians for spin systems, including:
Zeeman, Exchange, Hyperfine, Dipole–Dipole, and Zero-Field Splitting interactions

-Support for common strain distributions

-Simulation of disordered samples (powder, glassy states)

-Simulation of oriented samples

-Powerful spectra simulation and parameter search using Optuna / Nevergrad

-Full support for:
float32 / complex64 and float64 / complex128 data types
CPU and GPU computation backends

⏱️ Time-Resolved Capabilities

-Define arbitrary relaxation processes based on eigenvectors and eigenvalues

-General framework for basis transformations

-Support for temperature-dependent effects

-Partial support for density matrix formalism

🤖 Machine Learning Features

-Extensive tools for spectra and dataset generation

-Encoding methods for spin systems and spectra

-Integrated training workflows for ML-based analysis and prediction

📂 Examples

The examples/ folder contains a collection of practical tutorials demonstrating:
-Spin system construction
-EPR spectra simulation (CW and time-resolved)
-Parameter fitting and optimization
-Machine learning workflows for spectra analysis


