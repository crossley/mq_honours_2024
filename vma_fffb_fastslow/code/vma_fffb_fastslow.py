from imports import *

"""
- This project aims to follow Hewitson et al. (2023). There,
  we found that including midpoint feedback -- and thereby
  inducing feedback corrections to an ongoing movement --
  lead to some very bizarre behaviour. In particular, the
  effect of sensory uncertainty on adaptation appeared to be
  almost entirely independent of error magnitude. One
  possible explanation for this is rooted in explicit
  strategy use, and another is rooted in implicit motor
  adaptation.

- We previously sought to adjudicate between these
  possibilities by performing an experiment that attemted to
  control for explicit strategies by giving explicit
  instructions to "Please reach directly to the target. Do
  not aim off-target in order to get the cursor to land
  on-target."

- Surprisingly we failed to replicate Hewitson et al.
  (2023). Our suspicion is that Hewitson coached
  participants to move considerably more slowly than we did.
  This intuition is the motivation for the current project.

- We will compare a **slow** condition to a **fast**
  condition both inclduding midpoint feedback. The
  prediction is that the slow condition will show the
  stratification effect of the origincal Hewitson et al.
  paper but the fast condition will not (becuase movements
  will occur too quickly in incorporate the midpoint
  feedback into an online correction.

- The movement time manipulation is implemented with a
  timer. If the movement is deemed either too fast or two
  slow (depends on the condition where the cutoffs are set)
  then a too fast / too slow message is displayed and
  shortly after a demonstration of the desired speed is
  drawn to the screen.

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

# set up condition-specific variables on the basis of
# subject number
condition_list = ["slow", "fast"]
condition = condition_list[(subject - 1) % 2]

n_trial = 400

if condition == "slow":
    mt_too_slow = 3000
    mt_too_fast = 1500

elif condition == "fast":
    mt_too_slow = 1500
    mt_too_fast = 750

su_low = 0.01 * px_per_cm
su_mid = 0.25 * px_per_cm
su_high = 0.75 * px_per_cm
su_inf = np.nan
su = np.random.choice([su_low, su_mid, su_high, su_inf], n_trial)
su = np.random.choice([su_mid, su_mid, su_mid, su_mid], n_trial)

rotation = np.zeros(n_trial)
rotation[n_trial // 3:2 * n_trial // 3] = 15 * np.pi / 180

mpep_visible = np.zeros(n_trial)
mpep_visible[:2 * n_trial // 3:] = 1

n_cloud_dots = 20

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
# pygame.mouse.set_visible(False)

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

start_pos = (center_x, center_y + 5 * px_per_cm)
target_pos = (center_x, center_y - 5 * px_per_cm)

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
trial = -1

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

# calibration vars
screen_coord_upper_left = (0, 0)
screen_coord_upper_right = (screen_width, 0)
screen_coord_lower_right = (screen_width, screen_height)
screen_coord_lower_left = (0, screen_height)

rig_coord_upper_left = (0, 0)
rig_coord_upper_right = (0, 0)
rig_coord_lower_right = (0, 0)
rig_coord_lower_left = (0, 0)

running = True
while running:

    rot_mat = np.array([[np.cos(rotation[trial]), -np.sin(rotation[trial])],
                        [np.sin(rotation[trial]),
                         np.cos(rotation[trial])]])

    time_exp += clock_exp.tick()
    screen.fill((0, 0, 0))
    hand_pos = pygame.mouse.get_pos()
    cursor_pos = np.dot(np.array(hand_pos) - np.array(start_pos),
                        rot_mat) + start_pos

    # uncomment for debug
    # pygame.draw.circle(screen, magenta, cursor_pos, cursor_radius, 20)
    # pygame.draw.circle(screen, yellow, hand_pos, cursor_radius, 20)

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
            state_current = "state_iti"

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
            mt = -1
            ep = -1
            t_state = 0
            trial += 1

            if not np.isnan(su[trial]):
                cloud = np.random.multivariate_normal(
                    [0, 0], [[su[trial]**2, 0], [0, su[trial]**2]],
                    n_cloud_dots)

            if trial == n_trial:
                state_current = "state_finished"
            else:
                state_current = "state_searching_ring"

    if state_current == "state_searching_ring":
        t_state += clock_state.tick()

        r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                    (hand_pos[1] - start_pos[1])**2)

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, white, start_pos, r, 2)

        if r < 2 * start_radius:
            t_state = 0
            state_current = "state_searching_cursor"

    if state_current == "state_searching_cursor":
        t_state += clock_state.tick()

        r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                    (hand_pos[1] - start_pos[1])**2)

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, white, hand_pos, cursor_radius)

        if r < start_radius:
            t_state = 0
            state_current = "state_holding"
        elif r >= 2 * start_radius:
            t_state = 0
            state_current = "state_searching_ring"

    if state_current == "state_holding":
        t_state += clock_state.tick()

        r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                    (hand_pos[1] - start_pos[1])**2)

        # smoothly transition from blue to red with
        # increasing time until next state
        if t_state < 2000:
            proportion = t_state / 2000
            red_component = int(255 * proportion)
            blue_component = int(255 * (1 - proportion))
            state_color = (red_component, 0, blue_component)
            pygame.draw.circle(screen, state_color, start_pos, start_radius)
            pygame.draw.circle(screen, white, hand_pos, cursor_radius)

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

        r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                    (hand_pos[1] - start_pos[1])**2)

        r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                           (target_pos[1] - start_pos[1])**2)

        if r > (r_target / 2) * 0.9:
            mt = t_state
            t_state = 0

            # if not np.isnan(su[trial]):
            #     cloud = cloud - cloud.mean(axis=0) + cursor_pos

            state_current = "state_feedback_mp"

    if state_current == "state_feedback_mp":
        t_state += clock_state.tick()

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, red, target_pos, target_radius)

        r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                    (hand_pos[1] - start_pos[1])**2)

        r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                           (target_pos[1] - start_pos[1])**2)

        if mpep_visible[trial] == 1:
            if not np.isnan(su[trial]):
                for i in range(n_cloud_dots):
                    pygame.draw.circle(screen, white,
                                       cloud[i, :] - cloud.mean() + cursor_pos,
                                       cursor_radius)

        if t_state > 100:
            mt += t_state
            t_state = 0
            state_current = "state_moving_2"

    if state_current == "state_moving_2":
        t_state += clock_state.tick()

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, red, target_pos, target_radius)

        r = np.sqrt((hand_pos[0] - start_pos[0])**2 +
                    (hand_pos[1] - start_pos[1])**2)

        r_target = np.sqrt((target_pos[0] - start_pos[0])**2 +
                           (target_pos[1] - start_pos[1])**2)

        if r > r_target:
            mt += t_state
            t_state = 0

            ep_hand = hand_pos
            ep_theta_hand = np.arctan2(ep_hand[1] - start_pos[1],
                                       ep_hand[0] - start_pos[0])

            ep = cursor_pos
            ep_theta = np.arctan2(ep[1] - start_pos[1], ep[0] - start_pos[0])
            ep_target = (r_target * np.cos(ep_theta) + start_pos[0],
                         r_target * np.sin(ep_theta) + start_pos[1])

            if not np.isnan(su[trial]):
                cloud = cloud - cloud.mean(axis=0) + ep_target

            if mt > mt_too_slow:
                state_current = "too_slow"
            elif mt < mt_too_fast:
                state_current = "too_fast"
            else:
                state_current = "state_feedback_ep"

    if state_current == "state_feedback_ep":
        t_state += clock_state.tick()

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, red, target_pos, target_radius)

        # if mpep_visible[trial] == 1:
        #     if not np.isnan(su[trial]):
        #         for i in range(n_cloud_dots):
        #             pygame.draw.circle(screen, white, cloud[i], cursor_radius)

        if t_state > 1000:
            trial_data['condition'].append(condition)
            trial_data['subject'].append(subject)
            trial_data['trial'].append(trial)
            trial_data['rotation'].append(np.round(rotation[trial], 2))
            trial_data['rt'].append(rt)
            trial_data['mt'].append(mt)
            trial_data['ep'].append(np.round(ep_theta_hand, 2))

            if su[trial] == su_low:
                trial_data['su'].append("low")
            elif su[trial] == su_mid:
                trial_data['su'].append("mid")
            elif su[trial] == su_high:
                trial_data['su'].append("high")
            elif np.isnan(su[trial]):
                trial_data['su'].append("inf")

            pd.DataFrame(trial_data).to_csv(full_path, index=False)

            t_state = 0
            state_current = "state_iti"

    if state_current == "too_fast":
        t_state += clock_state.tick()
        text = font.render("Too fast!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

        if t_state > 2000:
            t_state = 0
            state_current = "demo_speed"

    if state_current == "too_slow":
        t_state += clock_state.tick()
        text = font.render("Too slow!", True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

        if t_state > 2000:
            t_state = 0
            state_current = "demo_speed"

    if state_current == "demo_speed":
        t_state += clock_state.tick()

        text = font.render("Please try to move at about the following speed",
                           True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, red, target_pos, target_radius)

        if t_state > 2000:
            t_state = 0
            state_current = "demo_speed_2"

    if state_current == "demo_speed_2":
        t_state += clock_state.tick()

        pygame.draw.circle(screen, blue, start_pos, start_radius)
        pygame.draw.circle(screen, red, target_pos, target_radius)

        a0 = a1 = a2 = 0
        a3 = 10
        a4 = -15
        a5 = 6

        total_time = 1000
        normalized_time = t_state / total_time
        if normalized_time < 1:
            t = normalized_time
            y = a0 + a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5
            position = [
                start_pos[0], start_pos[1] + (target_pos[1] - start_pos[1]) * y
            ]

            pygame.draw.circle(screen, (255, 0, 0), position, cursor_radius)

        if t_state > 2000:
            t_state = 0
            state_current = "state_searching_ring"

    # TODO: Implement calibration states
    if state_current == "calibrate_upper_left":
        t_state += clock_state.tick()
        text = font.render("Please to the upper left corner of the screen",
                           True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

    if state_current == "calibrate_upper_right":
        t_state += clock_state.tick()
        text = font.render("Please to the upper right corner of the screen",
                           True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

    if state_current == "calibrate_lower_right":
        t_state += clock_state.tick()
        text = font.render("Please to the lower right corner of the screen",
                           True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

    if state_current == "calibrate_lower_left":
        t_state += clock_state.tick()
        text = font.render("Please to the lower left corner of the screen",
                           True, (255, 255, 255))
        text_rect = text.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.fill(black)
        screen.blit(text, text_rect)

    pygame.display.flip()

pygame.quit()
