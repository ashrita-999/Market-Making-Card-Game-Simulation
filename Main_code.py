import time
import random
import numpy as np
import matplotlib.pyplot as plt

# game constants
NUM_TEAMS = 6
NUM_ROUNDS = 3
NUM_CARDS_ON_TABLE = 6
        
    

def initialise_deck():
    deck = []
    # Two copies of each card from 0 to 20
    for card_value in range(21):
        deck.extend([card_value, card_value])
    # One copy of each negative card -10 to -80 stepping by -10
    for neg_value in range(-10, -90, -10):
        deck.append(neg_value)
    return deck

class Team:
    def __init__(self, strategy, risk_profile):
        self.strategy = strategy # Bayesian or Frequentist
        self.risk_profile = risk_profile # float ∈ [0, 1]
        self.private_cards = []
        self.label = f"{strategy} (Risk: {int(risk_profile * 100)}%)"
    
    def estimate_EV(self, known_deck, known_total = 60):
        if self.strategy == 'Bayesian':
            return self.bayesian_EV(known_deck)
        if self.strategy == 'Frequentist':
            return self.frequentist_EV(known_deck)
    
    def bayesian_EV(self, known_deck, num_samples=1000):
        revealed = self.private_cards
        remaining_cards = [c for c in known_deck if c not in revealed]
        
        samples = [sum(random.sample(remaining_cards, 6)) for _ in range(num_samples)]
        mean = sum(samples)/num_samples
        variance = sum((x - mean)**2 for x in samples)/ num_samples
        std = variance** 0.5
        return mean, std
    
    def frequentist_EV(self, known_deck, known_total=60):
        revealed = self.private_cards
        remaining_sum = known_total - sum(revealed)
        remaining_cards = [c for c in known_deck if c not in revealed]
        mean = (remaining_sum/len(remaining_cards))*6
        std = 10 #random value for now

        return mean, std
        
    def set_bid_ask(self, mean, std):
        # Define min and max spreads in absolute value
        max_spread = 6  # For low-risk teams
        min_spread = 2  # For high-risk teams
    
        # Interpolate spread based on risk_profile
        spread = max_spread - (max_spread - min_spread) * self.risk_profile

        bid = mean - spread / 2
        ask = mean + spread / 2
    
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - 0.01
            ask = mid + 0.01
        
        return round(bid, 2), round(ask, 2)
    
    def make_first_bid(self, known_deck):
        mean, std = self.estimate_EV(known_deck)
        return self.set_bid_ask(mean, std)
    
    def make_decision(self, current_bid, current_ask, market_state, known_deck):
        mean, std = self.estimate_EV(known_deck)

        if current_ask is not None and mean > current_ask:
             return 'buy'

        elif current_bid is not None and mean < current_bid:
            return 'sell'
        
        else:
            return 'nothing'
       
        
    def compute_reaction_time(self, known_deck, market_state):
        start_time = time.perf_counter()
        current_bid = market_state.get("current_bid")  
        current_ask = market_state.get("current_ask")
        self.make_decision(current_bid, current_ask, market_state, known_deck)
        end_time = time.perf_counter()
        reaction_time = end_time - start_time
        return reaction_time 

def play_game_2(teams, known_deck, initial_bid, initial_ask):
    current_bid = initial_bid
    current_ask = initial_ask
    market_state = {"current_bid": current_bid, "current_ask": current_ask}

    remaining_teams = teams.copy()
    already_acted = set()

    while remaining_teams:
        # Compute reaction times for remaining teams
        reaction_times = {team: team.compute_reaction_time(known_deck, market_state) for team in remaining_teams}
        
        # Sort teams by fastest reaction time
        ordered_teams = sorted(remaining_teams, key=lambda t: reaction_times[t])
        
        # Process teams in order, updating market state after each action
        for acting_team in ordered_teams:
            # Make decision based on current market state
            current_bid = market_state["current_bid"]
            current_ask = market_state["current_ask"]
            decision = acting_team.make_decision(current_bid, current_ask, market_state, known_deck)
            
            print(f"\nTeam {teams.index(acting_team)} ({acting_team.label}) decides to {decision.upper()} (mean EV: {acting_team.estimate_EV(known_deck)[0]:.2f})")
            
            if decision == 'buy':
                print(f"Team {teams.index(acting_team)} ({acting_team.label}) BUYS at ask: {current_ask:.2f}")
            elif decision == 'sell':
                print(f"Team {teams.index(acting_team)} ({acting_team.label}) SELLS at bid: {current_bid:.2f}")
            else:
                print(f"Team {teams.index(acting_team)} ({acting_team.label}) does NOTHING.")
            
            # After acting, update the bid/ask based on this team's new estimate
            mean, std = acting_team.estimate_EV(known_deck)
            new_bid, new_ask = acting_team.set_bid_ask(mean, std)
            market_state["current_bid"] = new_bid
            market_state["current_ask"] = new_ask
            print(f"Team {teams.index(acting_team)} ({acting_team.label}) BID/ASK: {new_bid:.2f} / {new_ask:.2f}")

            
            
            # Remove this team from remaining teams
            already_acted.add(acting_team)
            remaining_teams = [t for t in teams if t not in already_acted]
            
            # Break if no teams remain
            if not remaining_teams:
                break
                

def play_game_2_no_print(teams, known_deck, initial_bid, initial_ask):
    current_bid = initial_bid
    current_ask = initial_ask
    market_state = {"current_bid": current_bid, "current_ask": current_ask}

    remaining_teams = teams.copy()
    already_acted = set()

    while remaining_teams:
        # Compute reaction times for remaining teams
        reaction_times = {team: team.compute_reaction_time(known_deck, market_state) for team in remaining_teams}
        
        # Sort teams by fastest reaction time
        ordered_teams = sorted(remaining_teams, key=lambda t: reaction_times[t])
        
        # Process teams in order, updating market state after each action
        for acting_team in ordered_teams:
            # Make decision based on current market state
            current_bid = market_state["current_bid"]
            current_ask = market_state["current_ask"]
            decision = acting_team.make_decision(current_bid, current_ask, market_state, known_deck)
            
            
            # After acting, update the bid/ask based on this team's new estimate
            mean, std = acting_team.estimate_EV(known_deck)
            new_bid, new_ask = acting_team.set_bid_ask(mean, std)
            market_state["current_bid"] = new_bid
            market_state["current_ask"] = new_ask
            
            # Remove this team from remaining teams
            already_acted.add(acting_team)
            remaining_teams = [t for t in teams if t not in already_acted]
            
            # Break if no teams remain
            if not remaining_teams:
                break
        

def run_single_game():
    #initialise deck
    deck = initialise_deck()
    random.shuffle(deck)
    teams = []
    market_cards = deck[:6] #first 6 cards of deck
    adj_deck = deck[6:] # adjusted deck 
    
    print('THE CARDS:', market_cards, 'THE SUM:', np.sum(market_cards))
    
    for i in range(NUM_TEAMS):
        #if i % 2 == 0:
            #strategy = 'Bayesian'
        #else:
         #   strategy = 'Frequentist'
        #risk = (i/NUM_TEAMS)
        strategy = random.choice(['Bayesian', 'Frequentist'])
        risk = random.uniform(0,1)
        teams.append(Team(strategy, risk))
    
    for round_num in range (1,4):
        print(f"\n======== ROUND {round_num} ========")
        
        for team in teams:
            team.private_cards.append(adj_deck.pop())
        
        if round_num ==1:
            start_team = random.choice(teams)
            first_bid, first_ask = start_team.make_first_bid(deck)
            print(f"\nStart Team ({teams.index(start_team)}) makes first bid/ask: {first_bid:.2f} / {first_ask:.2f}, mean EV: {team.estimate_EV(deck)[0]:.2f})")
            play_game_2(teams, deck, first_bid, first_ask)
        
        else:
            for team in teams:
                team.private_cards.append(adj_deck.pop())
            start_team = random.choice(teams)
            first_bid, first_ask = start_team.make_first_bid(deck)
            print(f"\nStart Team ({teams.index(start_team)}) makes first bid/ask: {first_bid:.2f} / {first_ask:.2f}, mean EV: {team.estimate_EV(deck)[0]:.2f})")
            play_game_2(teams, deck, first_bid, first_ask)
        
    
    true_table_value = sum(market_cards)
    print(f"\nTrue table value: {true_table_value}")
    
    errors = []
    for i, team in enumerate(teams):
        mean, _ = team.estimate_EV(deck)
        error = abs(mean - true_table_value)
        errors.append((i, error))
        print(f"Team {i}: EV = {mean:.2f}, error = {error:.2f}")
    
    winner = min(errors, key=lambda x: x[1])
    winner_index = winner[0]
    winning_team = teams[winner_index]

    print(f"\nWinner: Team {winner[0]} with lowest error: {winner[1]:.2f}")
    return winning_team.strategy, winning_team.risk_profile
    
def run_single_game_no_print():
        #initialise deck
    deck = initialise_deck()
    random.shuffle(deck)
    teams = []
    market_cards = deck[:6] #first 6 cards of deck
    adj_deck = deck[6:] # adjusted deck 
    
    for i in range(NUM_TEAMS):
        #if i % 2 == 0:
            #strategy = 'Bayesian'
        #else:
         #   strategy = 'Frequentist'
        #risk = (i/NUM_TEAMS)
        strategy = random.choice(['Bayesian', 'Frequentist'])
        risk = random.uniform(0,1)
        teams.append(Team(strategy, risk))
    
    for round_num in range (1,4):
        for team in teams:
            team.private_cards.append(adj_deck.pop())
        
        if round_num ==1:
            start_team = random.choice(teams)
            first_bid, first_ask = start_team.make_first_bid(deck)
            play_game_2_no_print(teams, deck, first_bid, first_ask)
        
        else:
            for team in teams:
                team.private_cards.append(adj_deck.pop())
            start_team = random.choice(teams)
            first_bid, first_ask = start_team.make_first_bid(deck)
            play_game_2_no_print(teams, deck, first_bid, first_ask)
        
    
    true_table_value = sum(market_cards)
    
    errors = []
    for i, team in enumerate(teams):
        mean, _ = team.estimate_EV(deck)
        error = abs(mean - true_table_value)
        errors.append((i, error))
    
    winner = min(errors, key=lambda x: x[1])
    winner_index = winner[0]
    winning_team = teams[winner_index]
    return winning_team.strategy, winning_team.risk_profile


def run_simulation_heatmap(num_games=10, num_risk_bins=10):
    strategies = ['Bayesian', 'Frequentist']
    # Initialize counts: rows=strategies, cols=risk bins
    win_counts = np.zeros((len(strategies), num_risk_bins))

    bin_edges = np.linspace(0, 1, num_risk_bins + 1)

    for _ in range(num_games):
        strategy, risk = run_single_game_no_print()  # from previous example
        strat_idx = strategies.index(strategy)
        # Find risk bin index
        risk_bin = np.digitize(risk, bin_edges) - 1
        if risk_bin == num_risk_bins:  # handle edge case when risk == 1.0
            risk_bin -= 1
        win_counts[strat_idx, risk_bin] += 1

    return win_counts, strategies, bin_edges



def plot_heatmap_matplotlib(win_counts, strategies, bin_edges):
    risk_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

    fig, ax = plt.subplots(figsize=(12, 5))
    cax = ax.imshow(win_counts, cmap='YlGnBu', aspect='auto')

    # Set ticks
    ax.set_xticks(np.arange(len(risk_labels)))
    ax.set_yticks(np.arange(len(strategies)))

    # Label ticks
    ax.set_xticklabels(risk_labels, rotation=45, ha='right')
    ax.set_yticklabels(strategies)

    # Add colorbar
    fig.colorbar(cax, ax=ax, label='Number of Wins')

    # Annotate each cell with win count
    for i in range(win_counts.shape[0]):
        for j in range(win_counts.shape[1]):
            ax.text(j, i, int(win_counts[i, j]), ha='center', va='center', color='black')

    ax.set_xlabel("Risk Profile Bins")
    ax.set_ylabel("Strategy")
    ax.set_title("Number of Wins by Strategy and Risk Profile")

    plt.tight_layout()
    plt.show()

def plot_best_combo(win_counts, strategies, bin_edges):
    num_strats, num_bins = win_counts.shape

    # Flatten the matrix and get the index of the maximum win count
    max_index = np.unravel_index(np.argmax(win_counts), win_counts.shape)
    max_strategy_idx, max_bin_idx = max_index
    max_strategy = strategies[max_strategy_idx]
    max_risk_range = (bin_edges[max_bin_idx], bin_edges[max_bin_idx + 1])
    max_wins = win_counts[max_strategy_idx, max_bin_idx]

    # Prepare data for plotting
    labels = []
    values = []
    for strat_idx in range(num_strats):
        for bin_idx in range(num_bins):
            label = f"{strategies[strat_idx]} | {bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx+1]:.1f}"
            labels.append(label)
            values.append(win_counts[strat_idx, bin_idx])

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(labels, values, color='skyblue')

    # Highlight best combo in red
    best_bar_idx = max_strategy_idx * num_bins + max_bin_idx
    bars[best_bar_idx].set_color('orange')

    ax.set_ylabel("Number of Wins")
    ax.set_xlabel("Strategy | Risk Profile Bin")
    ax.set_title(f"Best Combo: {max_strategy} with Risk {max_risk_range[0]:.1f}-{max_risk_range[1]:.1f} (Wins: {int(max_wins)})")
    ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()



run_single_game()
win_counts, strategies, bin_edges = run_simulation_heatmap(num_games=500)
plot_heatmap_matplotlib(win_counts, strategies, bin_edges)
plot_best_combo(win_counts, strategies, bin_edges)
