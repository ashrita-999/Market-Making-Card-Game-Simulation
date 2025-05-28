# Card-Based Market Making Simulation Game

## Project Overview
This project simulates a multi-round market making game where multiple teams compete by estimating the expected value (EV) of a hidden set of cards on the table and placing bid/ask prices accordingly. Each team uses either a Bayesian or Frequentist strategy combined with a risk profile to decide their bids, asks, and trade actions.

The simulation incorporates concepts from probability, Bayesian inference, frequentist statistics, risk management, and decision theory. It aims to explore how different statistical strategies perform under uncertainty and varying risk preferences.

---

## Key Features

- **Custom Card Deck**  
  The deck contains 50 cards: two copies of each integer from 0 to 20, and one copy each of negative values from -10 to -80 in steps of -10, with a total sum of 60.

- **Team Strategies**  
  - **Bayesian Teams:** Estimate expected value and uncertainty by Monte Carlo sampling from the remaining deck conditioned on known private cards.  
  - **Frequentist Teams:** Estimate EV using residual averages based on known totals and cards, with fixed standard deviation for simplicity.

- **Risk Profile Integration**  
  Each team has a risk profile [0, 1] that controls their bid-ask spread width, interpolated between a minimum (2 units) and maximum (6 units) spread.

- **Game Dynamics**  
  - 3 rounds of play; in each round, teams receive private cards from the deck.  
  - One team initiates the bid/ask prices based on their EV estimate.  
  - Teams decide to buy, sell, or hold by comparing their EV against the current market prices.  
  - Decisions occur sequentially, ordered by simulated reaction times.

- **Performance Analysis**  
  After multiple simulated games, the program aggregates results to produce heatmaps showing the number of wins per strategy and risk profile bin, helping identify the most effective strategy-risk combinations.

---

## Code Structure

- **`initialise_deck()`**: Creates the custom deck with specified card counts and values.

- **`Team` class**:  
  - Attributes: strategy type, risk profile, private cards.  
  - Methods:  
    - `estimate_EV()`: Calls Bayesian or Frequentist estimation methods.  
    - `bayesian_EV()`: Uses Monte Carlo sampling for EV and standard deviation.  
    - `frequentist_EV()`: Estimates EV by residual averages with fixed std dev.  
    - `set_bid_ask()`: Computes bid and ask prices using risk profile and EV.  
    - `make_first_bid()`: Generates initial bid/ask prices at game start.  
    - `make_decision()`: Chooses buy, sell, or nothing based on market state.  
    - `compute_reaction_time()`: Simulates decision latency.

- **Gameplay functions**:  
  - `play_game()`: Runs a full game round with print output for tracing decisions.  
  - `play_game_no_print()`: Same as above without console output, for batch simulations.

- **Simulation & Analysis**:  
  - `run_single_game()`: Executes one full game with output.  
  - `run_single_game_no_print()`: Runs a silent game for aggregation.  
  - `run_simulation_heatmap()`: Runs many games and collects win statistics by strategy and risk bins.  
  - Plotting functions (`plot_heatmap_matplotlib`, `plot_best_combo`) visualize results using Matplotlib.

---

## Requirements

- Python 3.x
- Libraries:  
  - `numpy`  
  - `matplotlib`

Install dependencies with:  
```bash
pip install numpy matplotlib
