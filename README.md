# Decoding the Curve: A Quantitative Research Journey into Interest Rate Models and Dynamics

This repository contains the code and report for my quantitative research project focused on understanding and modelling U.S. interest rate term structures. I wanted to get a practical feel for the kinds of analyses a quantitative researcher might perform on a rates desk.

## What I Walk Through

This was a deep dive for me, covering several key areas:

* **Yield Curve Fitting:** Started by fitting observed U.S. Treasury yields using the **Nelson-Siegel model** to get a handle on the curve's level, slope, and curvature.
* **Dynamic Modelling:** Implemented and calibrated the **one-factor Hull-White model** to explore how the short-term interest rate might evolve and how this model prices bonds. I looked into calibrating it not just to yields but also considering cap volatilities.
* **Understanding Key Drivers:** Used **Principal Component Analysis (PCA)** on historical yield changes to see what really makes the curve move – trying to isolate those main level, slope, and curvature factors.
* **Risk Assessment:** Calculated standard risk metrics like **Duration, Convexity, Value at Risk (VaR), and Expected Shortfall (ES)**. Also put the Hull-White model through its paces with some **stress test scenarios**.
* **Model Reality Check:** Built a **backtesting engine** to see how well the Hull-White model could actually forecast yields one month out, comparing it against a simple "carry" benchmark. (Spoiler: forecasting is hard!)
* **Simulation:** Ran Monte Carlo simulations with the calibrated Hull-White model to see the range of potential future short rates.

## Code & Report

* **Python Code (`.py` files):** The core logic is in Python, structured into classes for different parts of the analysis (e.g., data handling, Nelson-Siegel fitting, Hull-White model, PCA, backtesting). You'll find the main script `interest_rate_hw_model.py` (or similar name you used) orchestrates the analysis.
* **Comprehensive Report (`Report.pdf` or similar):** I've also included a detailed LaTeX report that walks through the theory behind the models, my implementation approach, and a thorough analysis of the results and visualizations.

## Why I Did This

This project was an attempt to apply what I've learned about quantitative finance to a real-world (or as real as I could make it!) problem. It was a great way to get hands-on experience with the entire lifecycle of a quantitative research task, from sourcing data to model validation and interpretation.

## Key Libraries Used

* NumPy
* Pandas
* SciPy (especially for optimization)
* Matplotlib (for all the plots!)
* Scikit-learn (for PCA and scaling)
* Statsmodels (used for some statistical fitting, e.g., AR(1) for PCA factors)
* `pandas_datareader` (for fetching FRED data)
* don't forget to run pip install -r requirements.txt if needed

Feel free to explore the code and the report. Any feedback is welcome!