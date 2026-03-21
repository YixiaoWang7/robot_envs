import numpy as np
from PIL import Image


def quaternion_multiply(q1, q2):
    """Multiply two quaternions: result = q1 * q2."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    
    return np.stack([x, y, z, w], axis=-1)


def quaternion_inverse(q):
    """Compute quaternion inverse (conjugate for unit quaternions)."""
    q_inv = q.copy()
    q_inv[..., :3] = -q_inv[..., :3]  # Negate x, y, z components
    return q_inv


def canonicalize_quaternion(q):
    """Ensure quaternion has positive w component (canonical form)."""
    q_canon = q.copy()
    if q[..., 3] < 0:
        q_canon = -q_canon
    return q_canon


def transform_to_relative_coordinates(eef_pos, eef_quat, obj_data):
    """Transform object pose from world frame to robot EEF frame.
    
    Args:
        eef_pos: End-effector position, shape (3,)
        eef_quat: End-effector quaternion, shape (4,) [qx, qy, qz, qw]
        obj_data: Object data in world frame, shape (7,) [pos(3), quat(4)]
    
    Returns:
        Object data in EEF frame, shape (7,) [pos_rel(3), quat_rel(4)]
    """
    # Extract object position and quaternion
    obj_pos = obj_data[:3]
    obj_quat = obj_data[3:7]
    
    # Transform position: relative_pos = obj_pos - eef_pos
    pos_relative = obj_pos - eef_pos
    
    # Transform orientation: quat_relative = eef_quat_inv * obj_quat
    eef_quat_inv = quaternion_inverse(eef_quat)
    quat_relative = quaternion_multiply(eef_quat_inv, obj_quat)
    
    # Canonicalize quaternion to have w >= 0
    quat_relative = canonicalize_quaternion(quat_relative)
    
    # Concatenate back to 7D
    return np.concatenate([pos_relative, quat_relative])


class ImageBasedCGWrapper:
    def __init__(self, make_env_fn, num_envs=1, policy=None, use_relative_coordinates=False, gripper_types="PandaGripper"):
        self.envs = [make_env_fn() for _ in range(num_envs)]
        self.policy = policy
        self.num_envs = num_envs
        self.obs = [None] * num_envs
        self.step_counter = [0] * num_envs
        self.use_relative_coordinates = use_relative_coordinates
        self.gripper_types = gripper_types
        # Store previous robot states for comparison
        self.prev_gripper_qpos = [None] * num_envs
        self.prev_eef_pos = [None] * num_envs
        self.prev_eef_quat = [None] * num_envs
        self.prev_joint_pos = [None] * num_envs
        self.prev_joint_vel = [None] * num_envs
        
        self.is_success = [False] * num_envs
        
        self.train_task = None
        
        # Debug settings
        self.debug_enabled = False # 改 true 好像有 bug 报错
        self.gripper_threshold = 0.0  # Configurable threshold
        self.track_robot_state = False  # Enable comprehensive robot state tracking

    def check_gripper_compliance(self, env_idx, action, obs, verbose=True):
        """
        Check if gripper obeyed the action.
        
        Args:
            env_idx: Environment index
            action: Action taken (7D array where last element is gripper action)
            obs: Observation after step
            verbose: Whether to print detailed debug info
            
        Returns:
            bool: True if gripper obeyed action, False otherwise
        """
        gripper_action = action[6]  # Last element is gripper action
        current_gripper_qpos = obs["robot0_gripper_qpos"]
        prev_gripper_qpos = self.prev_gripper_qpos[env_idx]
        
        # Calculate gripper movement
        gripper_movement = current_gripper_qpos - prev_gripper_qpos
        
        # Check if gripper moved in the expected direction
        # Based on Panda gripper XML:
        # - finger_joint1 (left): range="0.0 0.04" (decreases when closing, increases when opening)
        # - finger_joint2 (right): range="-0.04 0.0" (increases when closing, decreases when opening)
        if gripper_action > self.gripper_threshold:  # Closing action
            # For closing: finger_joint1 should decrease, finger_joint2 should increase
            expected_movement = np.array([-1, 1])  # [left_finger_decrease, right_finger_increase]
            actual_movement_direction = np.sign(gripper_movement)
            compliance = np.allclose(actual_movement_direction, expected_movement, atol=0.1)
            
        elif gripper_action < -self.gripper_threshold:  # Opening action
            # For opening: finger_joint1 should increase, finger_joint2 should decrease
            expected_movement = np.array([1, -1])  # [left_finger_increase, right_finger_decrease]
            actual_movement_direction = np.sign(gripper_movement)
            compliance = np.allclose(actual_movement_direction, expected_movement, atol=0.1)
            
        else:  # No action (gripper_action ≈ 0)
            # Should not move significantly
            compliance = np.allclose(gripper_movement, 0, atol=1e-4)
        

        # Print debug information if verbose
        if verbose:
            print(f"=== Gripper Debug (Env {env_idx}) ===")
            print(f"Gripper action: {gripper_action:.4f}")
            print(f"Threshold: {self.gripper_threshold}")
            print(f"Previous gripper qpos: {prev_gripper_qpos}")
            print(f"Current gripper qpos: {current_gripper_qpos}")
            print(f"Gripper movement: {gripper_movement}")
            print(f"Actual movement direction: {actual_movement_direction}")
            print(f"Expected movement direction: {expected_movement}")
            print(f"Movement magnitude: {np.linalg.norm(gripper_movement):.6f}")
            print(f"Gripper obeyed action: {compliance}")
            print("=" * 40)
        
        return compliance

    def calculate_object_distances(self, env_idx, obs, verbose=True):
        """
        Calculate distances between end-effector and objects/containers.
        
        Args:
            env_idx: Environment index
            obs: Current observation
            verbose: Whether to print detailed debug info
            
        Returns:
            dict: Dictionary containing distance information
        """
        # Extract positions
        eef_pos = obs["robot0_eef_pos"]
        obj_A_pos = obs["object_A_pos"]
        obj_B_pos = obs["object_B_pos"]
        
        # Calculate distances
        eef_to_obj_A_distance = np.linalg.norm(eef_pos - obj_A_pos)
        eef_to_obj_B_distance = np.linalg.norm(eef_pos - obj_B_pos)
        obj_A_to_obj_B_distance = np.linalg.norm(obj_A_pos - obj_B_pos)
        
        # Calculate relative positions (already computed in _compute_observation)
        rel_obj_A = obj_A_pos - eef_pos
        rel_obj_B = obj_B_pos - eef_pos
        
        distance_info = {
            "eef_to_obj_A_distance": eef_to_obj_A_distance,
            "eef_to_obj_B_distance": eef_to_obj_B_distance,
            "obj_A_to_obj_B_distance": obj_A_to_obj_B_distance,
            "rel_obj_A": rel_obj_A,
            "rel_obj_B": rel_obj_B,
            "eef_pos": eef_pos,
            "obj_A_pos": obj_A_pos,
            "obj_B_pos": obj_B_pos,
        }
        
        if verbose:
            print(f"📏 Object Distance Tracking (Env {env_idx})")
            print(f"  🎯 EEF to Object A: {eef_to_obj_A_distance:.6f}")
            print(f"  🎯 EEF to Object B: {eef_to_obj_B_distance:.6f}")
            print(f"  🔗 Object A to Object B: {obj_A_to_obj_B_distance:.6f}")
            print(f"  📍 EEF Position: {eef_pos}")
            print(f"  📦 Object A Position: {obj_A_pos}")
            print(f"  📦 Object B Position: {obj_B_pos}")
            print(f"  ➡️  Relative A: {rel_obj_A}")
            print(f"  ➡️  Relative B: {rel_obj_B}")
            
            # Check for close proximity
            if eef_to_obj_A_distance < 0.05:  # 5cm threshold
                print(f"  ✅ EEF is close to Object A!")
            if eef_to_obj_B_distance < 0.05:  # 5cm threshold
                print(f"  ✅ EEF is close to Object B!")
            if obj_A_to_obj_B_distance < 0.1:  # 10cm threshold
                print(f"  ✅ Objects A and B are close to each other!")
                
            print("-" * 40)
        
        return distance_info

    def track_robot_state_changes(self, env_idx, action, obs, verbose=True):
        """
        Track comprehensive robot state changes after env.step(action).
        
        Args:
            env_idx: Environment index
            action: Action taken (7D array)
            obs: Observation after step
            verbose: Whether to print detailed debug info
            
        Returns:
            dict: Dictionary containing state change information
        """
        if not self.track_robot_state:
            return {}
            
        # Extract current robot states
        current_eef_pos = obs["robot0_eef_pos"]
        current_eef_quat = obs["robot0_eef_quat"]
        current_gripper_qpos = obs["robot0_gripper_qpos"]
        current_joint_pos = obs["robot0_joint_pos"]
        current_joint_vel = obs.get("robot0_joint_vel", np.zeros_like(current_joint_pos))
        
        # Get previous states
        prev_eef_pos = self.prev_eef_pos[env_idx]
        prev_eef_quat = self.prev_eef_quat[env_idx]
        prev_joint_pos = self.prev_joint_pos[env_idx]
        prev_joint_vel = self.prev_joint_vel[env_idx]
        
        # Calculate state changes
        eef_pos_change = current_eef_pos - prev_eef_pos
        eef_pos_distance = np.linalg.norm(eef_pos_change)
        
        # Calculate quaternion difference (simplified)
        quat_diff = np.linalg.norm(current_eef_quat - prev_eef_quat)
        
        joint_pos_change = current_joint_pos - prev_joint_pos
        joint_vel_change = current_joint_vel - prev_joint_vel
        
        gripper_movement = current_gripper_qpos - self.prev_gripper_qpos[env_idx]
        
        # Extract action components
        pos_action = action[:3]  # Position action
        rot_action = action[3:6]  # Rotation action (if any)
        gripper_action = action[6]  # Gripper action
        
        # Calculate action magnitude
        pos_action_magnitude = np.linalg.norm(pos_action)
        rot_action_magnitude = np.linalg.norm(rot_action)
        
        # Calculate object distances
        distance_info = self.calculate_object_distances(env_idx, obs, verbose=verbose)
        
        # Store state change information
        state_changes = {
            "eef_pos_change": eef_pos_change,
            "eef_pos_distance": eef_pos_distance,
            "quat_diff": quat_diff,
            "joint_pos_change": joint_pos_change,
            "joint_vel_change": joint_vel_change,
            "gripper_movement": gripper_movement,
            "pos_action": pos_action,
            "rot_action": rot_action,
            "gripper_action": gripper_action,
            "pos_action_magnitude": pos_action_magnitude,
            "rot_action_magnitude": rot_action_magnitude,
            "total_action_magnitude": np.linalg.norm(action),
            # Add distance information
            "eef_to_obj_A_distance": distance_info["eef_to_obj_A_distance"],
            "eef_to_obj_B_distance": distance_info["eef_to_obj_B_distance"],
            "obj_A_to_obj_B_distance": distance_info["obj_A_to_obj_B_distance"],
            "rel_obj_A": distance_info["rel_obj_A"],
            "rel_obj_B": distance_info["rel_obj_B"],
        }
        
        if verbose:
            print(f"🤖 Robot State Tracking (Env {env_idx}) - Step {self.step_counter[env_idx]}")
            print(f"  📍 EEF Position Change: {eef_pos_change} (distance: {eef_pos_distance:.6f})")
            print(f"  🔄 EEF Quaternion Diff: {quat_diff:.6f}")
            print(f"  🦾 Joint Position Change: {joint_pos_change}")
            print(f"  ⚡ Joint Velocity Change: {joint_vel_change}")
            print(f"  🦀 Gripper Movement: {gripper_movement}")
            print(f"  🎯 Action Components:")
            print(f"    - Position: {pos_action} (magnitude: {pos_action_magnitude:.6f})")
            print(f"    - Rotation: {rot_action} (magnitude: {rot_action_magnitude:.6f})")
            print(f"    - Gripper: {gripper_action:.6f}")
            print(f"    - Total Action Magnitude: {state_changes['total_action_magnitude']:.6f}")
            
            # Check for unusual movements
            if eef_pos_distance > 0.1:  # Large position change
                print(f"  ⚠️  WARNING: Large EEF position change detected!")
            if np.any(np.abs(joint_pos_change) > 0.1):  # Large joint change
                print(f"  ⚠️  WARNING: Large joint position change detected!")
            if np.any(np.abs(joint_vel_change) > 1.0):  # Large velocity change
                print(f"  ⚠️  WARNING: Large joint velocity change detected!")
            
            # Render and save image for debugging
            try:
                frames = self.render()
                if frames and len(frames) > env_idx:
                    frame = frames[env_idx]
                    print(f"  📸 Rendered image shape: {frame.shape}")
                    
                    # Save the image to file
                    try:
                        import matplotlib.pyplot as plt
                        import os
                        
                        # Create debug_images directory if it doesn't exist
                        debug_dir = "debug_images"
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # Save image with step number
                        filename = f"{debug_dir}/env_{env_idx}_step_{self.step_counter[env_idx]}.png"
                        
                        plt.figure(figsize=(8, 6))
                        plt.imshow(frame)
                        plt.title(f"Environment {env_idx} - Step {self.step_counter[env_idx]}")
                        plt.axis('off')
                        plt.tight_layout()
                        plt.savefig(filename, dpi=150, bbox_inches='tight')
                        plt.close()  # Close the figure to free memory
                        print(f"  💾 Image saved to: {filename}")
                        
                    except ImportError:
                        print(f"  📸 Image rendered but matplotlib not available for saving")
                    except Exception as e:
                        print(f"  ⚠️  Failed to save image: {e}")
                        
            except Exception as e:
                print(f"  ⚠️  Failed to render image: {e}")
                
            print("-" * 60)
        
        return state_changes

    def enable_gripper_debug(self, threshold=0.0, verbose=False):
        """Enable gripper action compliance debugging."""
        self.debug_enabled = True
        self.gripper_threshold = threshold
        self.debug_verbose = verbose

    def disable_gripper_debug(self):
        """Disable gripper action compliance debugging."""
        self.debug_enabled = False

    def enable_robot_state_tracking(self, enabled=True):
        """Enable or disable comprehensive robot state tracking."""
        self.track_robot_state = enabled

    def reset(self, **kwargs):
        print("old wrapper")
        kwargs.pop("seed", None)
        self.obs = []
        
        if self.num_envs % 2 != 0:
            raise ValueError("num_envs must be even")


        # "L2":[[0,1],[1,0],[1,1],[1,2],[2,1]],
        # "Sfull":[[0,0],[0,1],[1,1],[1,2],[2,2],[2,0]],
        # "diagmid":[[0,0],[1,1],[2,2],[0,1]],
        # "diagcorner":[[0,0],[1,1],[2,2],[0,2]],


        if self.train_task == "S":
            # cross to plate
            # cube to bin
            # cylinder to bin
            # cylinder to cup
            ood_list = [(0, 2), (1, 0), (2, 0), (2, 1)]
        elif self.train_task == "L":
            ood_list = [(1, 1), (1, 2), (2, 1), (2, 2)]
        elif self.train_task == "only-00":
            ood_list = [(0, 0)]
        elif self.train_task == "diag":
            ood_list = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        elif self.train_task == "L2":
            ood_list = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
        elif self.train_task == "Sfull":
            ood_list = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)]
        elif self.train_task == "diagmid":
            ood_list = [(0, 0), (1, 1), (2, 2), (0, 1)]
        elif self.train_task == "diagcorner":
            ood_list = [(0, 0), (1, 1), (2, 2), (0, 2)]
        elif self.train_task == "all":
            ood_list = []
        else:
            raise ValueError(f"Invalid train task: {self.train_task}")


        if len(ood_list) > 0:
            if self.num_envs % 2 != 0:
                raise ValueError("num_envs must be even for ood and id separation.")
            target_per_group = self.num_envs // 2
            id_num = 0   # count of envs with y < -0.15
            ood_num = 0  # count of envs with y > -0.12

            for i, env in enumerate(self.envs):
                while True:
                    obs_i = env.reset(**kwargs)
                    task_id = get_task_ood_label(env.task)
                    
                    if id_num >= target_per_group:
                        if task_id in ood_list:
                            ood_num += 1
                            break
                    elif ood_num >= target_per_group:
                        if task_id not in ood_list:
                            id_num += 1
                            break
                    else:
                        if task_id in ood_list:
                            ood_num += 1
                            break
                        else:
                            id_num += 1
                            break
                        
                self.obs.append(obs_i)
                self.step_counter[i] = 0
                # Initialize previous robot states
                self.prev_gripper_qpos[i] = obs_i["robot0_gripper_qpos"].copy()
                self.prev_eef_pos[i] = obs_i["robot0_eef_pos"].copy()
                self.prev_eef_quat[i] = obs_i["robot0_eef_quat"].copy()
                self.prev_joint_pos[i] = obs_i["robot0_joint_pos"].copy()
                self.prev_joint_vel[i] = obs_i.get("robot0_joint_vel", np.zeros_like(obs_i["robot0_joint_pos"])).copy()            
        else:
            for i, env in enumerate(self.envs):
                obs_i = env.reset(**kwargs)
                
                self.obs.append(obs_i)
                self.step_counter[i] = 0
                # Initialize previous robot states
                self.prev_gripper_qpos[i] = obs_i["robot0_gripper_qpos"].copy()
                self.prev_eef_pos[i] = obs_i["robot0_eef_pos"].copy()
                self.prev_eef_quat[i] = obs_i["robot0_eef_quat"].copy()
                self.prev_joint_pos[i] = obs_i["robot0_joint_pos"].copy()
                self.prev_joint_vel[i] = obs_i.get("robot0_joint_vel", np.zeros_like(obs_i["robot0_joint_pos"])).copy()

        obs_dicts = [self._compute_observation(i) for i in range(self.num_envs)]
        batched_obs = {
            k: np.stack([obs[k] for obs in obs_dicts], axis=0)
            for k in obs_dicts[0]
        }
        
        self.is_success = [False] * self.num_envs
        
        return batched_obs, [{} for _ in range(self.num_envs)]

    def step(self, actions):
        next_obs, rewards, terminateds, truncateds, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            # Store previous states before step
            if self.track_robot_state and self.obs[i] is not None:
                self.prev_eef_pos[i] = self.obs[i]["robot0_eef_pos"].copy()
                self.prev_eef_quat[i] = self.obs[i]["robot0_eef_quat"].copy()
                self.prev_joint_pos[i] = self.obs[i]["robot0_joint_pos"].copy()
                self.prev_joint_vel[i] = self.obs[i].get("robot0_joint_vel", np.zeros_like(self.obs[i]["robot0_joint_pos"])).copy()
            
            # # Execute environment step
            # if self.is_success[i]:
            #     obs, reward, terminated, info = env.step(np.zeros_like(actions[i].copy()))
            # else:
            #     obs, reward, terminated, info = env.step(actions[i])
            
            if self.is_success[i]:
                obs, reward, terminated, info = env.step(np.zeros_like(actions[i].copy()))
            else:
                tmp_actions = actions[i].copy()
                # tmp_actions[:3] = (tmp_actions[:3] - self.obs[i]["robot0_eef_pos"]) * 0.05
                obs, reward, terminated, info = env.step(tmp_actions)
            
            # Track robot state changes
            if self.track_robot_state:
                state_changes = self.track_robot_state_changes(i, actions[i], obs, verbose=True)
                # Store state changes in info for potential use
                info = info or {}
                info["robot_state_changes"] = state_changes
            
            # Debug gripper action compliance if enabled
            if self.debug_enabled:
                gripper_compliance = self.check_gripper_compliance(
                    i, actions[i], obs
                )
                
                # Print warning if gripper didn't obey action
                if not gripper_compliance:
                    print(f"⚠️  WARNING: Gripper in environment {i} did NOT obey the action!")
            
            self.obs[i] = obs
            # Update previous gripper position for next step
            self.prev_gripper_qpos[i] = obs["robot0_gripper_qpos"].copy()
            self.step_counter[i] += 1
            truncated = False

            info["is_success"] = bool(env._check_success())

            if not self.is_success[i]:
                self.is_success[i] = info["is_success"]

            infos.append(info)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            next_obs.append(self._compute_observation(i, env))

        batched_obs = {
            k: np.stack([obs[k] for obs in next_obs], axis=0)
            for k in next_obs[0]
        }
        
        # At the end of the episode, include final_info
        final_info = [
            infos[i] if terminateds[i] or truncateds[i] else None
            for i in range(self.num_envs)
        ]

        return (
            batched_obs,
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            {"is_success": self.is_success, "final_info": final_info}
        )
        
    

    def _compute_observation(self, idx, env=None):
        # keys in self.obs[idx] are:
        # odict_keys(['robot0_joint_pos', 
        # 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 
        # 'robot0_joint_vel', 'robot0_eef_pos', 'robot0_eef_quat', 
        # 'robot0_eef_quat_site', 'robot0_gripper_qpos', 
        # 'robot0_gripper_qvel', 'agentview_image', 'cross_pos', 
        # 'cross_quat', 'cube_pos', 'cube_quat', 'cylinder_pos', 
        # 'cylinder_quat', 'bin_pos', 'bin_quat', 'cup_pos', 'cup_quat', 
        # 'plate_pos', 'plate_quat', 'gripper_to_object_A_pos', 
        # 'object_A_pos', 'object_A_quat', 'object_B_pos', 'object_B_quat', 
        # 'language_vector', 'robot0_proprio-state', 
        # 'object-state', 'object_A-state', 'object_B-state', 
        # 'language-state'])
        
        # target batch keys:
        # - observation.state.robot0_eef_pos: (B, T, 3)
        # - observation.state.robot0_eef_quat: (B, T, 4)
        # - observation.state.robot0_gripper_qpos: (B, T, 2)
        # - observation.state.object.cross: (B, T, 7)
        # - observation.state.object.cube: (B, T, 7)
        # - observation.state.object.cylinder: (B, T, 7)
        # - observation.state.object.bin: (B, T, 7)
        # - observation.state.object.cup: (B, T, 7)
        # - observation.state.object.plate: (B, T, 7)
        

        robot0_eef_pos = self.obs[idx]["robot0_eef_pos"]
        robot0_eef_quat = self.obs[idx]["robot0_eef_quat"]
        robot0_gripper_qpos = self.obs[idx]["robot0_gripper_qpos"]
        if self.gripper_types == "RethinkGripper":
            panda_close = np.array([ 0.00049761, -0.00049912])
            panda_open = np.array([ 0.03948026, -0.03948177])
            rethink_close = np.array([-0.0118367,   0.01183658])
            rethink_open = np.array([ 0.01106267, -0.01106308])
            
            robot0_gripper_qpos = panda_close + (robot0_gripper_qpos - rethink_close) / (rethink_open - rethink_close) * (panda_open - panda_close) 
            
        
        
        joint_pos = self.obs[idx]["robot0_joint_pos"]
        
        # Concatenate position and quaternion for each object (7D each)
        cross = np.concatenate([self.obs[idx]["cross_pos"], self.obs[idx]["cross_quat"]]) # (7,) ((3, ) + (4, ))
        cube = np.concatenate([self.obs[idx]["cube_pos"], self.obs[idx]["cube_quat"]]) # (7,)
        cylinder = np.concatenate([self.obs[idx]["cylinder_pos"], self.obs[idx]["cylinder_quat"]]) # (7,)
        bin = np.concatenate([self.obs[idx]["bin_pos"], self.obs[idx]["bin_quat"]]) # (7,)
        cup = np.concatenate([self.obs[idx]["cup_pos"], self.obs[idx]["cup_quat"]]) # (7,)
        plate = np.concatenate([self.obs[idx]["plate_pos"], self.obs[idx]["plate_quat"]]) # (7,)
        
        # Transform to relative coordinates if requested
        if self.use_relative_coordinates:
            # print("Transforming to relative coordinates")
            cross = transform_to_relative_coordinates(robot0_eef_pos, robot0_eef_quat, cross)
            cube = transform_to_relative_coordinates(robot0_eef_pos, robot0_eef_quat, cube)
            cylinder = transform_to_relative_coordinates(robot0_eef_pos, robot0_eef_quat, cylinder)
            bin = transform_to_relative_coordinates(robot0_eef_pos, robot0_eef_quat, bin)
            cup = transform_to_relative_coordinates(robot0_eef_pos, robot0_eef_quat, cup)
            plate = transform_to_relative_coordinates(robot0_eef_pos, robot0_eef_quat, plate)

        # state = np.concatenate([robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, joint_pos])
        state = np.concatenate([robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos])

        environment_state = np.concatenate([cross, cube, cylinder, bin, cup, plate])
        
        
        # get the image
        # Get camera image and resize to standard size
        agentview_image = self.obs[idx]["agentview_image"]
        # Convert image to PIL for resizing, then back to numpy
        agentview_image = Image.fromarray(agentview_image)
        # print('from env, before wrapper',pil_image.size)
        # resized_image = pil_image.resize((224, 224), Image.BILINEAR)
        agentview_image = np.array(agentview_image)
        agentview_image = np.flipud(agentview_image)
        
        # print(self.obs[idx].keys())
        
        robot0_eye_in_hand_image = self.obs[idx]["robot0_eye_in_hand_image"]
        robot0_eye_in_hand_image = Image.fromarray(robot0_eye_in_hand_image)
        robot0_eye_in_hand_image = np.array(robot0_eye_in_hand_image)
        robot0_eye_in_hand_image = np.flipud(robot0_eye_in_hand_image)

        

        return {
            "observation.state": state,
            "observation.environment_state": environment_state,
            "observation.images.agentview": agentview_image,
            "observation.images.robot0_eye_in_hand": robot0_eye_in_hand_image,
        }

    def render(self):
        frames = []
        for env in self.envs:
            try:
                frame = env.sim.render(
                    camera_name="agentview",
                    width=256,
                    height=256,
                )
                frame = np.flipud(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Render failed: {e}")
                frames.append(np.zeros((256, 256, 3), dtype=np.uint8))
        return frames

    def close(self):
        for env in self.envs:
            env.close()

    def __getattr__(self, attr):
        if attr == "num_envs":
            return self.num_envs
        if hasattr(self.envs[0], attr):
            return getattr(self.envs[0], attr)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def call(self, name, *args, **kwargs):
        results = []
        for env in self.envs:
            if name in {"_max_episode_steps", "horizon"}:
                results.append(env.horizon)
            elif name == "render":
                results.append(env.render())
            elif name == "task":
                results.append(env.task)
            else:
                attr = getattr(env, name)
                results.append(attr(*args, **kwargs) if callable(attr) else attr)
        return results

    def get_attr(self, name):
        return [getattr(env, name) for env in self.envs]

    def set_attr(self, name, values):
        for env, val in zip(self.envs, values):
            setattr(env, name, val)

    @property
    def unwrapped(self):
        return self

    @property
    def metadata(self):
        return {"render_fps": 20}
    
    def one_hot_tokenize(self, language_instructions):
        one_hot_dict = {"place": 0, "the": 1, "into": 2, "cross": 3, "cube": 4, "cylinder": 5, "bin": 6, "cup": 7, "plate": 8}
        words = language_instructions.split()
        language_vector = np.zeros(len(one_hot_dict))
        for word in words:
            language_vector[one_hot_dict[word]] = 1
        return language_vector



def get_task_ood_label(task: str) -> str:
    object_A_index = {"cross": 0, "cube": 1, "cylinder": 2}
    object_B_index = {"bin": 0, "cup": 1, "plate": 2}
    
    for key, value in object_A_index.items():
        if key in task:
            object_A_index = value
            break
    for key, value in object_B_index.items():
        if key in task:
            object_B_index = value
            break
    
    task_id = (object_A_index, object_B_index)
    return task_id