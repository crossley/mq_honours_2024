import pygame

pygame.init()

clock = pygame.time.Clock()
time = 0.0
time_max = 5000

# set the experiment to begin running
keep_running = True

# begin iterating through the experiment loop
while keep_running:

    # keep track of time
    time += clock.tick()

    # implement exp-wide stopping rule
    if time > time_max:
        print("Finished Experiment!")
        keep_running = False

pygame.quit()
