import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_subs_per_cnd = 16
conditions = ['low', 'high'] * n_subs_per_cnd
np.random.shuffle(conditions)

for i in range(len(conditions)):

    # Specify possible target angles
    target_angle = np.array([0, 45, 60, 75, 90, 105, 120, 135, 180, 225, 270, 315])
    target_train = target_angle[4]
    n_targets = target_angle.shape[0]

    # Specify the number of times you want to cycle through the targets. Note
    # that each phase can have a different set of targets to cycle between (see
    # below).
    n_cycle_baseline_no_fb = 2
    n_cycle_baseline_continuous_fb = 2
    n_cycle_baseline_endpoint_fb = 2
    n_cycle_baseline_mixed_fb = 2
    n_cycle_generalisation = 15
    n_cycle_washout_no_fb = 2
    n_cycle_washout_fb = 2

    n_gen_tops = n_targets - 1

    # Specify a single cycle's worth of targets for each phase
    targets_baseline_no_fb = target_angle
    targets_baseline_continuous_fb = target_angle
    targets_baseline_endpoint_fb = target_angle
    targets_baseline_mixed_fb = target_angle
    targets_generalisation = np.concatenate(
        (np.tile(target_angle[4], n_gen_tops), target_angle[0:4], target_angle[5:]))
    targets_washout_no_fb = target_angle
    targets_washout_fb = target_angle

    # Construct a target_angle array to be later added to the config dataframe
    target_angle = np.concatenate(
        (np.tile(targets_baseline_no_fb, n_cycle_baseline_no_fb),
         np.tile(targets_baseline_continuous_fb,
                 n_cycle_baseline_continuous_fb),
         np.tile(targets_baseline_endpoint_fb, n_cycle_baseline_endpoint_fb),
         np.tile(targets_baseline_mixed_fb, n_cycle_baseline_mixed_fb),
         np.tile(targets_generalisation, n_cycle_generalisation),
         np.tile(targets_washout_no_fb, n_cycle_washout_no_fb),
         np.tile(targets_washout_fb, n_cycle_washout_fb)))

    # For each phase, create an array to indicate the current cycle
    cycle_baseline_no_fb = np.repeat(
        np.arange(1, n_cycle_baseline_no_fb + 1, 1),
        targets_baseline_no_fb.shape[0])
    cycle_baseline_continuous_fb = np.repeat(
        np.arange(1, n_cycle_baseline_continuous_fb + 1, 1),
        targets_baseline_continuous_fb.shape[0])
    cycle_baseline_endpoint_fb = np.repeat(
        np.arange(1, n_cycle_baseline_endpoint_fb + 1, 1),
        targets_baseline_endpoint_fb.shape[0])
    cycle_baseline_mixed_fb = np.repeat(
        np.arange(1, n_cycle_baseline_mixed_fb + 1, 1),
        targets_baseline_mixed_fb.shape[0])
    cycle_generalisation = np.repeat(
        np.arange(1, n_cycle_generalisation + 1, 1),
        targets_generalisation.shape[0])
    cycle_washout_no_fb = np.repeat(np.arange(1, n_cycle_washout_no_fb + 1, 1),
                                    targets_washout_no_fb.shape[0])
    cycle_washout_fb = np.repeat(np.arange(1, n_cycle_washout_fb + 1, 1),
                                 targets_washout_fb.shape[0])

    # Combine the above into an array that can later be added to the config data frame
    cycle_phase = np.concatenate(
        (cycle_baseline_no_fb, cycle_baseline_continuous_fb,
         cycle_baseline_endpoint_fb, cycle_baseline_mixed_fb,
         cycle_generalisation, cycle_washout_no_fb, cycle_washout_fb))

    # Get the number of trials the previous two chunks yield for each phase
    n_trial_baseline_no_fb = n_cycle_baseline_no_fb * targets_baseline_no_fb.shape[
        0]
    n_trial_baseline_continuous_fb = n_cycle_baseline_continuous_fb * targets_baseline_continuous_fb.shape[
        0]
    n_trial_baseline_endpoint_fb = n_cycle_baseline_endpoint_fb * targets_baseline_endpoint_fb.shape[
        0]
    n_trial_baseline_mixed_fb = n_cycle_baseline_mixed_fb * targets_baseline_mixed_fb.shape[
        0]
    n_trial_generalisation = n_cycle_generalisation * targets_generalisation.shape[
        0]
    n_trial_washout_no_fb = n_cycle_washout_no_fb * targets_washout_no_fb.shape[
        0]
    n_trial_washout_fb = n_cycle_washout_fb * targets_washout_fb.shape[0]

    # Get the full number of trials combined across all phases
    n_trial = 0
    n_trial += n_trial_baseline_no_fb
    n_trial += n_trial_baseline_continuous_fb
    n_trial += n_trial_baseline_endpoint_fb
    n_trial += n_trial_baseline_mixed_fb
    n_trial += n_trial_generalisation
    n_trial += n_trial_washout_no_fb
    n_trial += n_trial_washout_fb

    # Construct a trial array to later add to the config dataframe
    trial = np.arange(1, n_trial + 1, 1)

    # Construct a phase indicator columns to be later added to the config dataframe
    phase = np.concatenate(
        (['baseline_no_fb'] * n_trial_baseline_no_fb,
         ['baseline_continuous_fb'] * n_trial_baseline_continuous_fb,
         ['baseline_endpoint_fb'] * n_trial_baseline_endpoint_fb,
         ['baseline_mixed_fb'] * n_trial_baseline_mixed_fb, 
         ['generalisation'] * n_trial_generalisation,
         ['washout_no_fb'] * n_trial_washout_no_fb,
         ['washout_fb'] * n_trial_washout_fb))

    # Specify the mean and standard deviation of the perturbation to be applied
    # during the clamp phase.
    if conditions[i] == 'low':
        rot_mean = 30
        rot_sig = 4
    elif conditions[i] == 'high':
        rot_mean = 30
        rot_sig = 12

    # Specify phase-specific instructions.
    instruct_phase = {
        'baseline_no_fb':
        'Please slice through the target as quickly and accurately as possible.\n'
        + 'You will not see the cursor during this phase.',
        'baseline_continuous_fb':
        'You will now only see the cursor throughout your entire reach.\n' +
        'Please continue to slice through the target as quickly and accurately as possible.',
        'baseline_endpoint_fb':
        'You will now only see the cursor only at the endpoint of your reach.\n'
        +
        'Please continue to slice through the target as quickly and accurately as possible.',
        'baseline_mixed_fb':
        'You will now only see the cursor at the endpoint of your reach on some trials.\n'
        + 'On the other trials you will not recieve feedback at all.\n'
        'Please continue to slice through the target as quickly and accurately as possible.',
        'clamp':
        'The cursor feedback is now clamped.\n' +
        'The location of the endpoint feedback is random.\n' +
        'It  has nothing to do with how accurately you reach.\n' +
        'Please do your best to ignore the cursor feedback and continue slicing directly through the target.',
        'generalisation':
        'You will now be asked to reach to targets that you have not yet reached to.\n'
        + 'You will not receive feedback of any kind for these reaches.' +
        'Please continue to slice through the target as quickly and accurately as possible.',
        'washout_no_fb':
        'You will not receive feedback of any kind for the following reaches.'
        +
        'Please continue to slice through the target as quickly and accurately as possible.',
        'washout_fb':
        'You will now only see the cursor throughout your entire reach.\n' +
        'Please continue to slice through the target as quickly and accurately as possible.',
    }

    # Create arrays that contain the phase-specific instructions once at the
    # start of each phase and nowhere else.
    instruct_baseline_no_fb = [instruct_phase['baseline_no_fb']
                               ] + [''] * (n_trial_baseline_no_fb - 1)
    instruct_baseline_continuous_fb = [
        instruct_phase['baseline_continuous_fb']
    ] + [''] * (n_trial_baseline_continuous_fb - 1)
    instruct_baseline_endpoint_fb = [
        instruct_phase['baseline_endpoint_fb']
    ] + [''] * (n_trial_baseline_endpoint_fb - 1)
    instruct_baseline_mixed_fb = [instruct_phase['baseline_mixed_fb']
                                  ] + [''] * (n_trial_baseline_mixed_fb - 1)
    instruct_generalisation = [instruct_phase['generalisation']
                               ] + [''] * (n_trial_generalisation - 1)
    instruct_washout_no_fb = [instruct_phase['washout_no_fb']
                              ] + [''] * (n_trial_washout_no_fb - 1)
    instruct_washout_fb = [instruct_phase['washout_fb']
                           ] + [''] * (n_trial_washout_fb - 1)

    # Combine each phase-specific array defined above into a larger array that
    # can later be added to the config dataframe.
    instruct_phase = np.concatenate(
        (instruct_baseline_no_fb, instruct_baseline_continuous_fb,
         instruct_baseline_endpoint_fb, instruct_baseline_mixed_fb, 
         instruct_generalisation, instruct_washout_no_fb,
         instruct_washout_fb))

    # The experiment code also defines instructions that are displayed for
    # every state. The following is an indicator column that should be used to
    # switch them on or off.
    instruct_state = np.ones(instruct_phase.shape)

    # Continuous cursor feedback
    cursor_vis = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         1 * np.ones(n_trial_baseline_continuous_fb),
         0 * np.ones(n_trial_baseline_endpoint_fb),
         0 * np.ones(n_trial_baseline_mixed_fb),
         0 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 0 * np.ones(n_trial_washout_fb)))

    # midpoint feedback
    midpoint_vis = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         0 * np.ones(n_trial_baseline_continuous_fb),
         0 * np.ones(n_trial_baseline_endpoint_fb),
         0 * np.ones(n_trial_baseline_mixed_fb),
         0 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 0 * np.ones(n_trial_washout_fb)))

    # endpoint feedback
    endpoint_vis = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         1 * np.ones(n_trial_baseline_continuous_fb),
         1 * np.ones(n_trial_baseline_endpoint_fb),
         np.random.permutation([0, 1] * (n_trial_baseline_mixed_fb // 2)), 
         0 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 1 * np.ones(n_trial_washout_fb)))

    # continuous cursor cloud standard deviation
    cursor_sig = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         0 * np.ones(n_trial_baseline_continuous_fb),
         0 * np.ones(n_trial_baseline_endpoint_fb),
         0 * np.ones(n_trial_baseline_mixed_fb),
         0 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 0 * np.ones(n_trial_washout_fb)))

    # midpoint cursor cloud standard deviation
    cursor_mp_sig = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         0 * np.ones(n_trial_baseline_continuous_fb),
         0 * np.ones(n_trial_baseline_endpoint_fb),
         0 * np.ones(n_trial_baseline_mixed_fb),
         0 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 0 * np.ones(n_trial_washout_fb)))

    # endpoint cursor cloud standard deviation
    cursor_ep_sig = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         0 * np.ones(n_trial_baseline_continuous_fb),
         0 * np.ones(n_trial_baseline_endpoint_fb),
         0 * np.ones(n_trial_baseline_mixed_fb),
         0 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 0 * np.ones(n_trial_washout_fb)))

    # whether or not cursor feedback of any kind is clamped
    clamp = np.concatenate(
        (0 * np.ones(n_trial_baseline_no_fb),
         0 * np.ones(n_trial_baseline_continuous_fb),
         0 * np.ones(n_trial_baseline_endpoint_fb),
         0 * np.ones(n_trial_baseline_mixed_fb),
         1 * np.ones(n_trial_generalisation),
         0 * np.ones(n_trial_washout_no_fb), 0 * np.ones(n_trial_washout_fb)))

    # cursor rotation
    rot = np.concatenate((
        np.random.normal(0, rot_sig, n_trial_baseline_no_fb),
        np.random.normal(0, rot_sig, n_trial_baseline_continuous_fb),
        np.random.normal(0, rot_sig, n_trial_baseline_endpoint_fb),
        np.random.normal(0, rot_sig, n_trial_baseline_mixed_fb),
        np.random.normal(rot_mean, rot_sig, n_trial_generalisation),
        np.random.normal(0, rot_sig, n_trial_washout_no_fb),
        np.random.normal(0, rot_sig, n_trial_washout_fb),
    ))

    # Construct the config dataframe
    d = pd.DataFrame({
        'condition': conditions[i],
        'subject': i,
        'trial': trial,
        'phase': phase,
        'cycle_phase': cycle_phase,
        'target_angle': target_angle,
        'cursor_vis': cursor_vis,
        'midpoint_vis': midpoint_vis,
        'endpoint_vis': endpoint_vis,
        'cursor_sig': cursor_sig,
        'cursor_mp_sig': cursor_mp_sig,
        'cursor_ep_sig': cursor_ep_sig,
        'clamp': clamp,
        'rot': rot,
        'instruct_phase': instruct_phase,
        'instruct_state': instruct_state
    })

    # Randomise target order within each phase and cycle_phase
    d['target_angle'] = d.groupby(
        ['phase',
         'cycle_phase'])['target_angle'].sample(frac=1).reset_index(drop=True)

    # Turn on endpoint feedback for training target during the generalisation
    # phase
    d.loc[(d['phase'] == 'generalisation') &
          (d['target_angle'] == target_train), 'endpoint_vis'] = 1

    # # NOTE: plot design
    # nn = [
    #     n_trial_baseline_no_fb, n_trial_baseline_continuous_fb,
    #     n_trial_baseline_endpoint_fb, n_trial_baseline_mixed_fb, n_trial_clamp,
    #     n_trial_generalisation, n_trial_washout_no_fb, n_trial_washout_fb
    # ]
    # labels = [
    #     'baseline_no_feedback', 'baseline_continuous_fb',
    #     'baseline_endpoint_fb', 'baseline_mixed_fb', 'clamp', 'generalisation',
    #     'washout_no_fb', 'washout_fb'
    # ]
    # labels_x = np.concatenate(([0], np.cumsum(nn)[:-1]))
    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # ax[0, 0].scatter(trial,
    #                  rot,
    #                  c=d['target_angle'],
    #                  alpha=d['endpoint_vis'] * 0.5 + 0.25)
    # ax[0, 0].vlines(labels_x, 0, rot_mean + 5, 'k', '--')
    # for i in range(len(labels)):
    #     ax[0, 0].text(labels_x[i], np.max(rot) + 5, labels[i], rotation=30)
    # ax[0, 0].set_ylabel('Rotation (degrees)')
    # ax[0, 0].set_xlabel('Trial')
    # ax[0, 0].set_xticks(np.arange(0, n_trial + 1, 20))
    # plt.show()

    # dd = d[['condition', 'subject', 'trial', 'phase', 'cycle_phase', 'target_angle',
    #    'cursor_vis', 'midpoint_vis', 'endpoint_vis', 'cursor_sig',
    #    'cursor_mp_sig', 'cursor_ep_sig', 'clamp', 'rot', 'instruct_phase',
    #    'instruct_state']]
    # dd.plot(subplots=True, layout=(4, 4))
    # plt.show()

    d.to_csv('../config/config_reach_' + str(i) + '.csv', index=False)