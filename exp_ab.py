#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:25:26 2024

@author: zhangyuzhu
"""


import numpy as np
import matplotlib.pyplot as plt


lambda1 = 0.8
lambda2 = 1
q = 0.5
mean_abandon_time = 3
simulation_time = 100000

# Run the simulation and output the matching probability
def run_simulation(lambda_star, lambda2, q, mean_abandon_time):
    # Queue length for class 1 or 2 users
    queue1 = 0
    queue2 = 0
    # Matching times
    matches = 0
    # Number of times class 1 users arrive
    class1_arrival = 0
    # Cumulative queue length of class 1 or 2 users
    AQ1 = 0
    AQ2 = 0
    
    
    events = []
    TNOW = 0
    # Initial arrival time of class 1 users
    arrival_time1 = np.random.exponential(1/max(lambda_star, 0.001)) # Avoid division by zero
    # Initial arrival time of class 2 users
    arrival_time2 = np.random.exponential(1/lambda2) 
    termination_time = simulation_time
    
    # Add the initial event to the event list
    events.append((arrival_time1, 'arrival1'))
    events.append((arrival_time2, 'arrival2'))
    events.append((termination_time, 'termination'))
        
    
    queue1_history = []
    queue2_history = []
    matches_history = []
    time_history = []
    
    
    while TNOW < termination_time:
        
        # Sort the event list in chronological order
        events.sort(key=lambda x: x[0])
        # Get the earliest event from the event calendar
        TNEXT, event_type = events.pop(0)
        
        # Update the cumulative queue length
        AQ1 += (TNEXT - TNOW) * queue1
        AQ2 += (TNEXT - TNOW) * queue2
    
        # Update the current time
        TNOW = TNEXT
 

        if event_type == 'arrival1':
            class1_arrival += 1 # Class 1 user arrival count
            if queue2 > 0 and np.random.rand() > (1 - q)**queue2:
                
                queue2 -= 1 # Class 2 user queue length decreases
                matches += 1 # Match count increases
                
            else:
                queue1 += 1 # Class 1 user enters queue
                
                # Schedule abandonment event
                abandonment_time = TNOW + np.random.exponential(mean_abandon_time)

                events.append((abandonment_time, 'abandon1'))
            # Schedule the next arrival event of class 1 user
            next_arrival_time = TNOW + np.random.exponential(1/max(lambda_star, 0.001))
            events.append((next_arrival_time, 'arrival1'))
        
        elif event_type == 'arrival2':
            if queue1 > 0 and np.random.rand() > (1 - q)**queue1:
                
                queue1 -= 1 # Class 1 user queue length decreases
                matches += 1 # Match count increases
                
            else:
                queue2 += 1 # Class 2 users enter the queue
                # Schedule abandonment events
                abandonment_time = TNOW + np.random.exponential(mean_abandon_time)

                events.append((abandonment_time, 'abandon2'))
            # Schedule the next class 2 user arrival event
            next_arrival_time = TNOW + np.random.exponential(1/lambda2)
            events.append((next_arrival_time, 'arrival2'))
    
        elif event_type == 'abandon1':
            if queue1 > 0:
                queue1 -= 1
    
        elif event_type == 'abandon2':
            if queue2 > 0:
                queue2 -= 1

        queue1_history.append(queue1)
        queue2_history.append(queue2)
        matches_history.append(matches)
        time_history.append(TNOW)
        
        if event_type == 'termination':
            break
    
    pr_match = matches / class1_arrival if class1_arrival > 0 else 0
    

    print(f"Number of matches: {matches}")
    print(f"Queue1 length: {queue1}")
    print(f"Queue2 length: {queue2}")
    
    average_queue1_length = AQ1 / TNOW
    average_queue2_length = AQ2 / TNOW
    average_matches_per_time_unit = matches / simulation_time
    
    print(f"Average queue 1 length: {average_queue1_length}")
    print(f"Average queue 2 length: {average_queue2_length}")
    print(f"Average matches per time unit: {average_matches_per_time_unit}")
    
    return pr_match



# Calculate the equilibrium


def calculate_equilibrium(lambda1, lambda2, q, mean_abandon_time, p, epsilon=0.001):
    lambda_star = lambda1  
    stopped = False
    while not stopped:
        pr_match = run_simulation(lambda_star, lambda2, q, mean_abandon_time)  
        if pr_match == 0:
            lambda_star = 0.1
        else:
            prob_joining = (200 * pr_match - p) / (200 * pr_match)  
            if prob_joining <= 0:
                prob_joining = 0.001
            new_lambda_star = lambda1 * prob_joining  
        if abs(new_lambda_star - lambda_star) < epsilon:  
            stopped = True
        else:
            lambda_star = new_lambda_star  
    return lambda_star, lambda_star * p , pr_match, prob_joining




# calculate_optimal_price


def calculate_optimal_price(lambda1, lambda2, q, mean_abandon_time):
    steps=100
    P = np.zeros(steps)
    R = np.zeros(steps)
    ls = np.zeros(steps)
    for i in range(steps):
        
        p = 200/steps*(i+1)
        lambda_star, revenue = calculate_equilibrium(lambda1, lambda2, q, mean_abandon_time, p)
            
        P[i] = p
        R[i] = revenue
        ls[i] = lambda_star
        
    if len(R) == 0 or np.max(R) == 0:  # Prevent errors in optimal income calculation
        return None, None, None

    optimal_price = P[np.argmax(R)]
    ls_optimal = ls[np.argmax(R)]
    optimal_revenue = np.max(R)
    
    print(f"Optimal price: {optimal_price}")
    print(f"Optimal revenue: {optimal_revenue}")
    print(f"Optimal lambda_star: {ls_optimal}")
    
    return optimal_price, optimal_revenue, ls_optimal

    
def plot_parameter_variation(param_name, param_values, lambda1, lambda2, q, mean_abandon_time):
    optimal_prices = []
    optimal_revenues = []
    lambda_stars = []
    #valid_param_values = []
    
    for value in param_values:
        if param_name == 'q':
            optimal_price, optimal_revenue, lambda_star = calculate_optimal_price(lambda1, lambda2, value, mean_abandon_time)
        elif param_name == 'mean_abandon_time':
            optimal_price, optimal_revenue, lambda_star = calculate_optimal_price(lambda1, lambda2, q, value)
        elif param_name == 'lambda1':
            optimal_price, optimal_revenue, lambda_star = calculate_optimal_price(value, lambda2, q, mean_abandon_time)
        elif param_name == 'lambda2':
            optimal_price, optimal_revenue, lambda_star = calculate_optimal_price(lambda1, value, q, mean_abandon_time)
        
        if optimal_price is None or optimal_revenue is None or lambda_star is None:
            continue  # If None is returned, skip this parameter value
        
        \
        #valid_param_values.append(value)
        optimal_prices.append(optimal_price)
        optimal_revenues.append(optimal_revenue)
        lambda_stars.append(lambda_star)
        
    if not optimal_prices or not optimal_revenues or not lambda_stars:
        print("No valid results were returned for the given parameter range.")
        return

    plt.figure(figsize=(14, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(param_values[:len(optimal_prices)], optimal_prices, label=f'Optimal Price vs {param_name}')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Optimal Price')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(param_values[:len(optimal_revenues)], optimal_revenues, label=f'Optimal Revenue vs {param_name}')
    plt.xlabel(f'{param_name}')
    plt.ylabel('Optimal Revenue')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(param_values[:len(lambda_stars)], lambda_stars, label=f'lambda* vs {param_name}')
    plt.xlabel(f'{param_name}')
    plt.ylabel('lambda*')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()






q_values = np.linspace(0.1, 0.9, 10)  # q parameter variation range
mean_abandon_time_values = np.linspace(1, 8, 10)  # mean_abandon_time parameter variation range
lambda1_values = np.linspace(0.5, 2.0, 10)  # lambda1 parameter variation range
lambda2_values = np.linspace(0.5, 2.0, 10)  # lambda2 parameter variation range


plot_parameter_variation('q', q_values, lambda1, lambda2, q, mean_abandon_time=3)
plot_parameter_variation('mean_abandon_time', mean_abandon_time_values, lambda1, lambda2, q, mean_abandon_time=3)
plot_parameter_variation('lambda1', lambda1_values, lambda1, lambda2, q, mean_abandon_time=3)
plot_parameter_variation('lambda2', lambda2_values, lambda1, lambda2, q, mean_abandon_time=3)


