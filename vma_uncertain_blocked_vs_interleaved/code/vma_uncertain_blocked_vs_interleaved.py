from imports import *

"""
- This project aims to following up a surprising finding we
  have recently reported.

- In a paradigm where midpoint feedback was provided to
  allow and encourage online within-movement corrections,
  adaptation given high sensory uncertainty was greater than
  adaptation given low sensory uncertainty.

- Something like this has been reported many times before
  but our results are surprising because we compared a
  blocked design to an interleaved design and found that the
  adpatation to low sensory uncertainty was much greater in
  the interleaved condition than in the blocked condition.

- To our knowledge, this has never been reported before. It
  is also unpredicted by any current theory.

- Here, we aim to investigate blocked vs interleaved
  conditions with center-out reaches (no midpoint feedback)
  in order to establish the boundaries of this effect.

- There are three conditions: blocked_low, blocked_high, and
  interleaved. Subjects are assigned to a condition based on
  their subject number.

- There is not currently a way to pause the experiment and
  there are no blocks or breaks. We may wish to add these
  but I'm not sure.

- Task instructions must be given verbally in the lab. They
  are not automated in this code.

- Consent must also currently be given and recorded manually
  in the lab, but we may pivot to automation down the road.

"""

subject = 1
dir_data = "../data"
f_name = f"sub_{subject}_data.csv"
full_path = os.path.join(dir_data, f_name)

# # Uncomment to check if file already exists
# if os.path.exists(full_path):
#     print(f"File {f_name} already exists. Aborting.")
#     sys.exit()

# useful constants but need to change / verify on each computer
pixels_per_inch = 227 / 2
px_per_cm = pixels_per_inch / 2.54

n_trial = 400

condition_list = ["blocked_low", "blocked_high", "interleaved"]
condition = condition_list[(subject - 1) % 3]

if condition == "interleaved":
    su = np.random.choice([1 * px_per_cm, 2 * px_per_cm], n_trial)
elif condition == "blocked_low":
    su = 0.1 * px_per_cm * np.ones(n_trial)
elif condition == "blocked_high":
    su = 0.5 * px_per_cm * np.ones(n_trial)

su = np.random.choice([1 * px_per_cm, 2 * px_per_cm], n_trial)

rotation = np.zeros(n_trial)
rotation[n_trial // 3:2 * n_trial // 3] = 15 * np.pi / 180

pygame.init()

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
grey = (128, 128, 128)
white = (255, 255, 255)
cyan = (0, 255, 255)
magenta = (255, 0, 255)
yellow = (255, 255, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# cursor circle radius
cursor_radius = 8
start_radius = 15
target_radius = 15

# relevant coords
center_x = screen.get_width() // 2
center_y = screen.get_height() // 2

start_pos = (center_x, center_y + 2 * px_per_cm)
target_pos = (center_x, center_y - 6 * px_per_cm)

# create clocks to keep time
clock_state = pygame.time.Clock()
clock_exp = pygame.time.Clock()

t_state = 0.0
time_exp = 0.0

# initial state
state_init = "state_init"

# set the current state to the initial state
state_current = state_init

# behavioural measurements
rt = -1
mt = -1
ep = -1
resp = -1

# record keeping
trial_data = {
    'condition': [],
    'subject': [],
    'trial': [],
    'su': [],
    'rotation': [],
    'rt': [],
    'mt': [],
    'ep': []
}

# set trials / phases
trial = 1

running = True
while running:

    time_exp += clock_exp.tick()
    screen.fill((0, 0, 0))
    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                pygame.quit()
            else:
                resp = event.key

    if state_current == "state_init":
        t_state += clock_state.tick()
        text = font.render("Please press the space bar to begin", True,
                           (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

        if resp == pygame.K_SPACE:
            t_state = 0
            resp = -1
            state_current = "state_searching_ring"

    if state_current == "state_finished":
        t_state += clock_state.tick()
        text = font.render("You finished! Thank you for being awesome!", True,
                           (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

    if state_current == "state_iti":
        t_state += clock_state.tick()
        screen.fill(black)
        if t_state > 1000:
            resp = -1
            rt = -1
            t_state = 0
            trial += 1
            if trial == n_trial:
                state_current = "state_finished"
            else:
                state_current = "state_searching_ring"

    if state_current == "state_searching_ring":
        t_state += clock_state.tick()

        r = np.sqrt((mouse_pos[0] - start_pos[0])**2 +
                    (mouse_pos[1] - start_pos[1])**2)

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, white, start_pos, r, 2)

        if r < 2 * start_radius:
            t_state = 0
            state_current = "state_searching_cursor"

    if state_current == "state_searching_cursor":
        t_state += clock_state.tick()

        r = np.sqrt((mouse_pos[0] - start_pos[0])**2 +
                    (mouse_pos[1] - start_pos[1])**2)

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, white, mouse_pos, cursor_radius)

        if r < start_radius:
            t_state = 0
            state_current = "state_holding"
        elif r >= 2 * start_radius:
            t_state = 0
            state_current = "state_searching_ring"

    if state_current == "state_holding":
        t_state += clock_state.tick()

        r = np.sqrt((mouse_pos[0] - start_pos[0])**2 +
                    (mouse_pos[1] - start_pos[1])**2)

        # smoothly transition from blue to green with
        # increasing time until next state
        if t_state < 2000:
            proportion = t_state / 2000
            green_component = int(255 * proportion)
            blue_component = int(255 * (1 - proportion))
            state_color = (0, green_component, blue_component)
            pygame.draw.circle(screen, state_color, start_pos, start_radius)
            pygame.draw.circle(screen, white, mouse_pos, cursor_radius)

        if r >= start_radius:
            t_state = 0
            state_current = "state_searching_cursor"

        elif t_state > 2000:
            t_state = 0
            t_state_2 = 0
            state_current = "state_moving"

    if state_current == "state_moving":
        t_state += clock_state.tick()

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, green, target_pos, target_radius)

        r = np.sqrt((mouse_pos[0] - start_pos[0])**2 +
                    (mouse_pos[1] - start_pos[1])**2)

        r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                           (target_pos[1] - start_pos[1])**2)

        if r >= r_target:
            ep = mouse_pos

            ep_theta = np.arctan2(ep[1] - start_pos[1], ep[0] - start_pos[0])
            ep_target = (r_target * np.cos(ep_theta) + start_pos[0],
                         r_target * np.sin(ep_theta) + start_pos[1])

            mt = t_state

            cloud = np.random.multivariate_normal(
                ep_target, [[su[trial]**2, 0], [0, su[trial]**2]], 10)

            # rotate the cloud by the rotation angle
            rot_mat = np.array(
                [[np.cos(rotation[trial]), -np.sin(rotation[trial])],
                 [np.sin(rotation[trial]),
                  np.cos(rotation[trial])]])

            cloud_rot = np.dot(cloud - start_pos, rot_mat) + start_pos

            t_state = 0
            state_current = "state_feedback_ep"

    if state_current == "state_feedback_ep":
        t_state += clock_state.tick()

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, green, target_pos, target_radius)

        for i in range(10):
            pygame.draw.circle(screen, white, cloud[i], cursor_radius)
            pygame.draw.circle(screen, blue, cloud_rot[i], cursor_radius)

        if t_state > 1000:
            trial_data['condition'].append(condition)
            trial_data['subject'].append(subject)
            trial_data['trial'].append(trial)
            trial_data['su'].append(np.round(su[trial], 2))
            trial_data['rotation'].append(np.round(rotation[trial], 2))
            trial_data['rt'].append(rt)
            trial_data['mt'].append(mt)
            trial_data['ep'].append(np.round(ep_theta, 2))
            pd.DataFrame(trial_data).to_csv(full_path, index=False)
            t_state = 0
            state_current = "state_iti"

    pygame.display.flip()

pygame.quit()
