# VIX_LHeston
Implementation of pricing VIX term structure under SV models (INCOMPLETE version)

## Description of each file

  - utils: functions to compute model-based VIX term structure under LHeston;

  - approx for rHeston: illustrations to show the convergence of VIX under LHeston to that under rHeston. For implementations of rHetson, I refer to [Prof. Jacquier's Github](https://github.com/JackJacquier). Comparisons are shown in rHestonMGF_compare.ipynb.
  
  - estimation_LHeston-2011: details of parameter estimation for LHeston model using VIX term structure data for the whole period;

## What we do
  - Derived an analytical formula of VIX under LHeston model, which can be seen as an analytical approximation for pricing VIX under rHeston model (the formula isAn verified through degenerated cases and Monte Carlo simulation);
  - Conducted empirical analysis by comparing the pricing performance of the VIX term structure for several commonly used continuous-time stochastic volatility models, including Heston, Bates, SVCJ, and LHeston.
