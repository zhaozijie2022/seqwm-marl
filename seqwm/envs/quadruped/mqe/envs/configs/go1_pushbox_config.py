import numpy as np
from seqwm.envs.quadruped.mqe.utils.helpers import merge_dict
from seqwm.envs.quadruped.mqe.envs.go1.go1 import Go1Cfg

class Go1PushboxCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1pushbox"
        num_envs = 1
        num_agents = 2
        num_npcs = 2
        episode_length_s = 15
    # rewards weight setting

    class rewards(Go1Cfg.rewards):
        expanded_ocb_reward = False # if True, the reward will be given based on Circular Arc Interpolation Trajectory
        class scales:
            target_reward_scale = 0.00325  # 0.00325
            approach_reward_scale = 0.0030  # 0.00075
            collision_punishment_scale = -0.0025
            push_reward_scale = 0.0030  # 0.0015
            ocb_reward_scale = 0.004
            reach_target_reward_scale = 10
            exception_punishment_scale = -5


    class asset(Go1Cfg.asset):
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/box.urdf"
        name_npc = "box"
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
        vertex_list = [[-0.60, -0.60], [ 0.60,-0.60],
                       [ 0.60,  0.60], [-0.60, 0.60]]
        # target area
        _terminate_after_contacts_on = []
        _file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/target.urdf"
        _name_npc = "target_area"
        _npc_collision = False
        _fix_npc_base_link = True
        _npc_gravity = True

    class goal:
        # static goal pos
        static_goal_pos = True
        goal_pos = [ 4.0, 0.73, 0.02]
        # goal_pos = [ 4.0, -0.45, 0.02]
        # goal_pos = [ 4.0, 0.17, 0.02]
        goal_rpy = [  0.0, 0.0, 0.0]

        new_random_goal_pos_xy = False
        new_random_goal_pos_x = [4.0, 4.0]
        new_random_goal_pos_y = [-0.7, -0.6]


        # random goal pos
        random_goal_pos = False
        random_goal_distance_from_init = [1.5 , 3.0]                    # target_pos_randomlization
        random_goal_theta_from_init = [0, 2 * np.pi] # [min, max]       # target_theta_randomlization
        random_goal_rpy_range = dict(                                   # target_yaw_randomlization
                                r= [-0.01, 0.01],
                                p= [-0.01, 0.01],
                                y= [-0.01, 0.01],
                            )

        # receive goal pos from the user or high layer
        received_goal_pos = False
        received_final_pos = [ 9.0, 0.0, 0.1]

        # use only for test mode(middle layer sequential task)
        sequential_goal_pos = False
        goal_poses = [
            [ 3.0, 0.0, 0.1],
            [ 4.0, 0.0, 0.1],
            [ 5.0, 0.0, 0.1],
            [ 6.0, 0.0, 0.1],
            [ 7.0, 0.0, 0.1]]
        general_dist = False   # if True, orientation of the goal
        yaw_active = True      # if True, the general_dist will calculated based on the yaw angle
        THRESHOLD = 0.3        # the threshold of completion

        # check the goal setting
        check_setting = [static_goal_pos, new_random_goal_pos_xy, random_goal_pos, received_goal_pos, sequential_goal_pos]
        if check_setting.count(True) != 1:
            raise ValueError("Only one of static_goal_pos, random_goal_pos, received_goal_pos, sequential_goal_pos can be True")

    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1
        map_size = [5.0, 4.0]
        BarrierTrack_kwargs = merge_dict(
            Go1Cfg.terrain.BarrierTrack_kwargs,
            dict(
                options = [
                    "init",
                    "plane",
                    "wall",
                ],
                # wall_thickness= 0.2,
                track_width = map_size[1],
                init = dict(
                    block_length = 0.1,
                    room_size = (0.1, map_size[1]),
                    border_width = 0.0,
                    offset = (0, 0),
                ),
                plane=dict(
                    block_length=map_size[0],
                ),
                wall = dict(
                    block_length = 0.1
                ),
                wall_height= 0.5,
                virtual_terrain = False, # Change this to False for real terrain
                no_perlin_threshold = 0.06,
                add_perlin_noise = False,
            )
        )


    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [1.0, 0.75, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [1.0, -0.75, 0.42],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            # physical box
            init_state_class(
                pos = [2.0, 0.0, 0.50],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            # target area
            init_state_class(
                pos = [4.0, 0.0, 0.1],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    class control(Go1Cfg.control):
        control_type = 'C'

    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
            "roll",
            "pitch",
        ]

    class domain_rand(Go1Cfg.domain_rand):
        push_robots = False  # use for non-virtual training
        random_base_init_state = False
        # init_base_tiny_pos_range = dict(
        #     x=[-0.25, 0.25],
        #     y=[-0.5, 0.5],
        # )
        # init_npc_pos_range = dict(
        #     x= [11.5, 13.0],
        #     y= [-0.5, 0.5],
        # )
        # init_npc_rpy_range = dict(
        #     r=[-0.00001, 0.00001],
        #     p=[-0.00001, 0.00001],
        #     y=[-0.01, 2 * np.pi],
        # )



    class viewer(Go1Cfg.viewer):
        pos = [12.5, -2., 4.]  # [m]
        lookat = [12.5, 3., 0.]  # [m]
