from imports import *
"""
- This project aims to followup Hewitson et al. MIS paper
"""

subject = 1
dir_data = "../data"
full_path = os.path.join(dir_data, f"sub_{subject}_data.csv")
full_path_move = os.path.join(dir_data, f"sub_{subject}_data_move.csv")

# # Uncomment to check if file already exists
# if os.path.exists(full_path):
#     print(f"File {f_name} already exists. Aborting.")
#     sys.exit()

# useful constants but need to change / verify on each computer
pixels_per_inch = 227 / 2
px_per_cm = pixels_per_inch / 2.54

n_trial = 374

rotation = np.zeros(n_trial)
rotation[132:] = 15 * np.pi / 180

endpoint_visible = np.ones(n_trial)

target_angles = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
target_angle = np.zeros(n_trial)

# target_angle[242:]

tmp = []

np.array([
    np.concatenate((tmp, np.random.permutation(target_angle_generalize)))
    for _ in range(6)
]).shape

# fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10, 5))
# ax[1, 0].plot(rotation / rotation.max(), label='rotation')
# ax[2, 0].plot(endpoint_visible, label='endpoint visible')
# [x.legend() for x in ax.flatten()]
# plt.show()

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
orange = (255, 165, 0)
green = (0, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)

# cursor circle radius
cursor_radius = 8
start_radius = 15
target_radius = 15

n_points = 20

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

trial_move = {
    'condition': [],
    'subject': [],
    'trial': [],
    'state': [],
    't': [],
    'x': [],
    'y': []
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

        # smoothly transition from blue to red with
        # increasing time until next state
        if t_state < 2000:
            proportion = t_state / 2000
            red_component = int(255 * proportion)
            blue_component = int(255 * (1 - proportion))
            state_color = (red_component, 0, blue_component)
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
        pygame.draw.circle(screen, red, target_pos, target_radius)

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
                ep_target, [[su[trial]**2, 0], [0, su[trial]**2]], n_points)

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
        pygame.draw.circle(screen, red, target_pos, target_radius)

        if endpoint_visible[trial]:
            for i in range(n_points):
                pygame.draw.circle(screen, white, cloud[i], cursor_radius)
                pygame.draw.circle(screen, white, cloud_rot[i], cursor_radius)

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
            pd.DataFrame(trial_move).to_csv(full_path_move, index=False)
            t_state = 0
            state_current = "state_iti"

    trial_move['condition'].append(condition)
    trial_move['subject'].append(subject)
    trial_move['trial'].append(trial)
    trial_move['state'].append(state_current)
    trial_move['t'].append(time_exp)
    trial_move['x'].append(mouse_pos[0])
    trial_move['y'].append(mouse_pos[1])

    pygame.display.flip()

pygame.quit()
