# CSCI 580 — Assignment 4: TSP with Simulated Annealing and Genetic Algorithms

Solves a 40-city Traveling Salesman Problem with **Simulated Annealing (SA)** (2-opt neighborhood, geometric cooling) and a **Genetic Algorithm (GA)** (tournament selection, Order Crossover, swap mutation, elitism), then compares them.

| Method | Best tour length | Improvement vs nearest neighbor |
|---|---|---|
| Nearest neighbor (baseline) | 6.0214 | — |
| Simulated Annealing | **5.0057** | **16.87%** |
| Genetic Algorithm | 5.3672 | 10.86% |
| GA with inversion mutation (Part 7, 5-seed mean) | 4.926 | 18.2% |

## Files

| File | What it is |
|---|---|
| `TSP_SA_and_GA_Assignment.ipynb` | Completed notebook with all TODOs implemented, Part 6 analysis, Part 7 extra-credit experiments. **All outputs, graphs and tables are saved in the notebook.** |
| `tsp_sa_ga.py` | The same code exported as a plain Python script |
| `CSCI580_Assignment4_Report.pdf` | Technical report (problem, methods with code, results, analysis) |
| `figures/` | Every plot from the notebook as a PNG |
| `requirements.txt` | Python dependencies |

## How to run

### Option 1 — Google Colab (no install)
1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `TSP_SA_and_GA_Assignment.ipynb`.
2. Click **Runtime → Run all**. The full notebook takes about 1–2 minutes.

### Option 2 — Jupyter on your computer
```bash
pip install -r requirements.txt
jupyter notebook TSP_SA_and_GA_Assignment.ipynb
```
Then choose **Kernel → Restart & Run All**.

### Option 3 — Plain Python script
```bash
pip install -r requirements.txt
python tsp_sa_ga.py
```
Plots open one after another in windows (close each to continue).

Every run uses fixed random seeds, so the numbers above are reproduced exactly.

## What's in the notebook

- **Part 5A – SA:** Metropolis acceptance rule (`delta <= 0` → accept, else accept with probability `exp(-delta/T)`) and geometric cooling `T = max(T * alpha, 1e-12)`. An optional `stats` dict records ΔE of accepted/rejected moves, current length and temperature.
- **Part 5B – GA:** `tournament_select`, `order_crossover_ox`, `mutate_swap`. An optional `stats` dict records population average/worst fitness, number of distinct tours and edge-based diversity per generation.
- **Part 6 – Analysis:** ΔE histograms, current-vs-best SA trace, greedy (T=0) control experiment, GA population fitness and diversity, SA vs GA normalized by tour evaluations, 10-seed variability study, written answers to all four questions.
- **Part 7 – Optional:** inversion mutation, PMX crossover, a memetic GA (2-opt local search), runs without the nearest-neighbor seed, and an SA parameter heat-map.

## Parameters used

- **SA:** iters = 20,000 · t0 = 0.2 · alpha = 0.9995 · seed = 123 · start = nearest-neighbor tour
- **GA:** pop_size = 200 · generations = 400 · tournament_k = 5 · crossover_rate = 0.9 · mutation_rate = 0.2 · elite_size = 5 · seed = 999 · initial population = NN tour + 199 random tours
