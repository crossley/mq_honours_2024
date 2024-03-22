from imports import *
from util_func import *

"""
- This project examines category learning while switching on
  a psuedo-random trial-by-trial basis between between
  sub-tasks that require 100% incongruent / conflicting
  stimulus-response mappings.

- It follows from Crossley et al. which showed that learning
  while switching between such sub-tasks can occur but
  apparently only if the response keys used within each
  sub-task were distinct.

    - sub-task 1 required S-R associations {S-R1, S-R2}
    - sub-task 1 required S-R associations {S-R3, S-R4}

- Learning did not occur / was severely impaired if the same
  response keys were used in both sub-tasks.

    - sub-task 1 required S-R associations {S-R1, S-R2}
    - sub-task 1 required S-R associations {S-R1, S-R2}

- It therefore appears that unique motor plans (i.e., as
  implied by unique response keys) are required to faciliate
  learning while task switching.

- However, it is unclear in this context what exactly
  constitutes a "motor plan". In particular, is it critical
  that unique motor effectors (e.g., fingers) be used,
  unique goal positions (e.g., response key), or both? We
  address that question here.

- There are four conditions: 2F2K, 2F4K, 4F2K, 4F4K. The
  first number indicates the number of fingers used and the
  second number indicates the number of keys used. The 2F2K
  and 4F4K conditions are replications of Crossley et al
  referenced above.

- Subjects are assigned to a condition based on their
  subject number.

- There is not currently a way to pause the experiment and
  there are no blocks or breaks. We may wish to add these
  but we can pilot without them to find out.

- Task instructions must be given verbally in the lab. They
  are not automated in this code.

- Consent must also currently be given and recorded manually
  in the lab, but we may pivot to automation down the road.
"""

# set subject number
subject = 1
dir_data = "../data"
f_name = f"sub_{subject}_data.csv"
full_path = os.path.join(dir_data, f_name)

# Uncomment to check if file already exists
# if os.path.exists(full_path):
#     print(f"File {f_name} already exists. Aborting.")
#     sys.exit()

ds = make_stim_cats()

# # plot the stimuli coloured by label
# fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 6))
# sns.scatterplot(data=ds, x="x", y="y", hue="cat", alpha=0.5, ax=ax[0, 0])
# sns.scatterplot(data=ds, x="xt", y="yt", hue="cat", alpha=0.5, ax=ax[0, 1])
# ax[0, 0].plot([0, 100], [0, 100], 'k--')
# ax[0, 1].plot([0, 5], [0, np.pi / 2], 'k--')
# plt.show()

# plot_stim_space_examples(ds)

# Initialize Pygame
pygame.init()

# useful constants but need to change / verify on each computer
pixels_per_inch = 227 / 2
px_per_cm = pixels_per_inch / 2.54

# grating size
size_cm = 5
size_px = int(size_cm * px_per_cm)

# set small window potentially useful for debugging
# screen_width, screen_height = 800, 600
# center_x = screen_width // 2
# center_y = screen_height // 2
# screen = pygame.display.set_mode((screen_width, screen_height))

# set full screen
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
center_x = screen_width // 2
center_y = screen_height // 2
screen = pygame.display.set_mode((screen_width, screen_height),
                                 pygame.FULLSCREEN)

# Hide the mouse cursor
pygame.mouse.set_visible(False)

# Set up fonts
font = pygame.font.Font(None, 36)

# Define colors
black = (0, 0, 0)
white = (255, 255, 255)
grey = (126, 126, 126)
green = (0, 255, 0)
red = (255, 0, 0)

# create clocks to keep time
clock_state = pygame.time.Clock()
clock_exp = pygame.time.Clock()

time_state = 0.0
time_exp = 0.0

# set the current state to the initial state
state_current = "state_init"

# behavioural measurements
resp = -1
rt = -1

# trial counter
trial = -1
n_trial = ds.shape[0]

# choose 1 or 2 randomly
sub_task = np.random.choice([1, 2])

# record keeping
trial_data = {
    'subject': [],
    'trial': [],
    'sub_task': [],
    'cat': [],
    'x': [],
    'y': [],
    'xt': [],
    'yt': [],
    'resp': [],
    'rt': []
}

running = True
while running:

    time_exp += clock_exp.tick()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                keep_running = False
                pygame.quit()
            else:
                resp = event.key

    if state_current == "state_init":
        time_state += clock_state.tick()
        text = font.render("Please press the space bar to begin", True,
                           (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

        condition_list = ["2F2K", "2F4K", "4F2K", "4F4K"]
        condition = condition_list[(subject - 1) % 4]

        if resp == pygame.K_SPACE:
            time_state = 0
            resp = -1
            state_current = "state_iti"

    if state_current == "state_finished":
        time_state += clock_state.tick()
        text = font.render("You finished! Thank you for being awesome!", True,
                           (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

    if state_current == "state_iti":
        time_state += clock_state.tick()
        screen.fill(black)
        pygame.draw.line(screen, white, (center_x, center_y - 10),
                         (center_x, center_y + 10), 4)
        pygame.draw.line(screen, white, (center_x - 10, center_y),
                         (center_x + 10, center_y), 4)
        if time_state > 1000:
            resp = -1
            rt = -1
            time_state = 0
            sub_task = np.random.choice([1, 2])
            trial += 1
            if trial == n_trial:
                state_current = "state_finished"
            else:
                sf = ds['xt'].iloc[trial] * px_per_cm**-1
                ori = ds['yt'].iloc[trial]
                cat = ds['cat'].iloc[trial]
                state_current = "state_stim"

    if state_current == "state_stim":
        time_state += clock_state.tick()
        screen.fill(black)

        if sub_task == 1:
            pygame.draw.rect(screen, grey,
                             (center_x - 2 * size_px / 2, center_y -
                              2 * size_px / 2, 2 * size_px, 2 * size_px))
        else:
            pygame.draw.polygon(screen, grey,
                                [(center_x, center_y - 1.3 * size_px),
                                 (center_x + 1.3 * size_px, center_y),
                                 (center_x, center_y + 1.3 * size_px),
                                 (center_x - 1.3 * size_px, center_y)])

        grating_patch = create_grating_patch(size_px, sf, ori)
        grating_surface = grating_to_surface(grating_patch)
        screen.blit(grating_surface,
                    (center_x - size_px / 2, center_y - size_px / 2))

        if (resp == pygame.K_d) or (resp == pygame.K_k):
            rt = time_state
            time_state = 0
            state_current = "state_feedback"

    if state_current == "state_feedback":
        time_state += clock_state.tick()

        if condition in ["2F2K", "4F2K"]:
            resp_keys = [pygame.K_d, pygame.K_k, pygame.K_d, pygame.K_k]
        elif condition in ["2F4K", "4F4K"]:
            resp_keys = [pygame.K_d, pygame.K_k, pygame.K_s, pygame.K_l]

        # The purpose of the -1 in the following line of
        # code is to convert cat[trial] from (1, 2) into (0,
        # 1) so that it can be used as an appropriate index
        # into resp_keys.
        if sub_task == 1:
            if resp == resp_keys[cat - 1]:
                fb = 'Correct'
            else:
                fb = 'Incorrect'

        elif sub_task == 2:
            # The +2 in the following line comes from how
            # resp_keys is defined above and is why the
            # order of key listings there matters.
            if resp == resp_keys[cat - 1 + 2]:
                fb = 'Correct'
            else:
                fb = 'Incorrect'

        if fb == "Correct":
            pygame.draw.circle(screen, green, (center_x, center_y),
                               size_px / 2 + 10, 5)

        elif fb == "Incorrect":
            pygame.draw.circle(screen, red, (center_x, center_y),
                               size_px / 2 + 10, 5)

        if time_state > 1000:
            trial_data['subject'].append(subject)
            trial_data['trial'].append(trial)
            trial_data['sub_task'].append(sub_task)
            trial_data['cat'].append(cat)
            trial_data['x'].append(np.round(ds.x[trial], 2))
            trial_data['y'].append(np.round(ds.y[trial], 2))
            trial_data['xt'].append(np.round(sf, 2))
            trial_data['yt'].append(np.round(ori, 2))
            trial_data['resp'].append(resp)
            trial_data['rt'].append(rt)
            pd.DataFrame(trial_data).to_csv(full_path, index=False)
            time_state = 0
            state_current = "state_iti"

    pygame.display.flip()
