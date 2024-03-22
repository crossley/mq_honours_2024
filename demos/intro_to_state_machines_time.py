"""
This script demonstrates the basic layout and logic of using
a state-machine to implement a simple game or experiment.
We define only two states -- state 1 and state 2 -- and
transition between them by moving from state to the other
after a certain amount of time has past.
"""

import pygame

pygame.init()

# initial state
state_init = 0

# specify a maximum amount of time for each state
time_max_state_0 = 1000
time_max_state_1 = 2000

# specify a maximum amount o time for the entire experiment
time_max_exp = 9000

# create clocks to keep time
clock_state = pygame.time.Clock()
clock_exp = pygame.time.Clock()

time_state = 0.0
time_exp = 0.0

# set the current state to the initial state
state_current = state_init

# set the experiment to begin running
keep_running = True

# begin iterating through the experiment loop
while keep_running:

    # keep track of total experiment time
    time_exp += clock_exp.tick()

    # implement things you want to happen when
    # `state_current==0`.
    if state_current == 0:

        # keep track of how long we have been in this state
        time_state += clock_state.tick()

        # TODO: do things here that you want to happen in
        # state 0

        # we must always implement code that exits the current
        # state under some specific set of conditions
        if time_state > time_max_state_0:

            print("Exiting state 0 at time: ", time_state)

            # reset `time_state` so that we start
            # counting fresh in the next state
            time_state = 0

            # set `state_current` to the next state you wish
            # to occupy
            state_current = 1

    # implement things you want to happen when
    # `state_current==1`.
    if state_current == 1:

        # keep track of how long we have been in this state
        time_state += clock_state.tick()

        # TODO: do things here that you want to happen in
        # state 1

        # implement state-transition logic
        if time_state > time_max_state_1:

            print("Exiting state 1 at time: ", time_state)

            # reset `time_state` so that we start
            # counting fresh in the next state
            time_state = 0

            # set `state_current` to the next state you wish
            # to occupy
            state_current = 0

    # implement exp-wide stopping rule
    if time_exp > time_max_exp:
        print("Finished Experiment!")
        keep_running = False

pygame.quit()
