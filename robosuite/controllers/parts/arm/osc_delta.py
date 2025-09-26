from robosuite.controllers.parts.controller import Controller
from robosuite.utils.control_utils import *
import robosuite.utils.transform_utils as T
import numpy as np
from scipy.spatial.transform import Rotation

# Set VERBOSE to True to debug
np.set_printoptions(precision=3)
VERBOSE = False

# Supported impedance modes
IMPEDANCE_MODES = {"fixed", "variable", "variable_kp"}


def angle_diff(vec1, vec2, degree=True):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    prod = np.dot(vec1, vec2)
    prod = np.clip(prod, -1, 1)
    angle = np.arccos(prod)
    if degree:
        angle = angle / np.pi * 180
    return angle


class OSCDelta(Controller):
    """
    Controller for controlling robot arm via operational space control. Allows position and / or orientation control
    of the robot's end effector. For detailed information as to the mathematical foundation for this controller, please
    reference http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

    NOTE: Control input actions can either be taken to be relative to the current position / orientation of the
    end effector or absolute values. In either case, a given action to this controller is assumed to be of the form:
    (x, y, z, ax, ay, az) if controlling pos and ori or simply (x, y, z) if only controlling pos

    Args:
        sim (MjSim): Simulator instance this controller will pull robot state updates from

        eef_name (str): Name of controlled robot arm's end effector (from robot XML)

        joint_indexes (dict): Each key contains sim reference indexes to relevant robot joint information, namely:

            :`'joints'`: list of indexes to relevant robot joints
            :`'qpos'`: list of indexes to relevant robot joint positions
            :`'qvel'`: list of indexes to relevant robot joint velocities

        actuator_range (2-tuple of array of float): 2-Tuple (low, high) representing the robot joint actuator range

        input_max (float or Iterable of float): Maximum above which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        input_min (float or Iterable of float): Minimum below which an inputted action will be clipped. Can be either be
            a scalar (same value for all action dimensions), or a list (specific values for each dimension). If the
            latter, dimension should be the same as the control dimension for this controller

        output_max (float or Iterable of float): Maximum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        output_min (float or Iterable of float): Minimum which defines upper end of scaling range when scaling an input
            action. Can be either be a scalar (same value for all action dimensions), or a list (specific values for
            each dimension). If the latter, dimension should be the same as the control dimension for this controller

        kp (float or Iterable of float): positional gain for determining desired torques based upon the pos / ori error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

        impedance_mode (str): Impedance mode with which to run this controller. Options are {"fixed", "variable",
            "variable_kp"}. If "fixed", the controller will have fixed kp and damping_ratio values as specified by the
            @kp and @damping_ratio arguments. If "variable", both kp and damping_ratio will now be part of the
            controller action space, resulting in a total action space of (6 or 3) + 6 * 2. If "variable_kp", only kp
            will become variable, with damping_ratio fixed at 1 (critically damped). The resulting action space will
            then be (6 or 3) + 6.

        kp_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is set to either
            "variable" or "variable_kp". This sets the corresponding min / max ranges of the controller action space
            for the varying kp values. Can be either be a 2-list (same min / max for all kp action dims), or a 2-list
            of list (specific min / max for each kp dim)

        damping_ratio_limits (2-list of float or 2-list of Iterable of floats): Only applicable if @impedance_mode is
            set to "variable". This sets the corresponding min / max ranges of the controller action space for the
            varying damping_ratio values. Can be either be a 2-list (same min / max for all damping_ratio action dims),
            or a 2-list of list (specific min / max for each damping_ratio dim)

        policy_freq (int): Frequency at which actions from the robot policy are fed into this controller

        position_limits (2-list of float or 2-list of Iterable of floats): Limits (m) below and above which the
            magnitude of a calculated goal eef position will be clipped. Can be either be a 2-list (same min/max value
            for all cartesian dims), or a 2-list of list (specific min/max values for each dim)

        orientation_limits (2-list of float or 2-list of Iterable of floats): Limits (rad) below and above which the
            magnitude of a calculated goal eef orientation will be clipped. Can be either be a 2-list
            (same min/max value for all joint dims), or a 2-list of list (specific min/mx values for each dim)

        interpolator_pos (Interpolator): Interpolator object to be used for interpolating from the current position to
            the goal position during each timestep between inputted actions

        interpolator_ori (Interpolator): Interpolator object to be used for interpolating from the current orientation
            to the goal orientation during each timestep between inputted actions

        control_ori (bool): Whether inputted actions will control both pos and ori or exclusively pos

        uncouple_pos_ori (bool): Whether to decouple torques meant to control pos and torques meant to control ori

        **kwargs: Does nothing; placeholder to "sink" any additional arguments so that instantiating this controller
            via an argument dict that has additional extraneous arguments won't raise an error

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(self,
                 sim,
                 ref_name,  # ref_name
                 joint_indexes,
                 actuator_range,
                 input_max=1,
                 input_min=-1,
                 output_max=1,
                 output_min=-1,
                 max_translation=0.03,
                 max_rotation=0.1,
                 kp=150,
                 damping_ratio=1,
                 impedance_mode="fixed",
                 input_ref_frame="base",
                 kp_limits=(0, 300),
                 damping_ratio_limits=(0, 100),
                 policy_freq=20,
                 position_limits=None,
                 orientation_limits=None,
                 interpolator_pos=None,
                 interpolator_ori=None,
                 control_ori=True,
                 uncouple_pos_ori=True,
                 use_lambda=False,
                 use_nullspace=False,
                 type=None,
                 lite_physics=True,
                 input_type="delta",


                 **kwargs  # does nothing; used so no error raised when dict is passed with extra terms used previously
                 ):

        super().__init__(
            sim,
            ref_name=ref_name,
            joint_indexes=joint_indexes,
            actuator_range=actuator_range,
            lite_physics=lite_physics,
            part_name=kwargs.get("part_name", None),
            naming_prefix=kwargs.get("naming_prefix", None),


        )

        self.use_ori = control_ori

        self.input_type = input_type
        assert self.input_type in [
            "delta", "absolute"], f"Input type must be delta or absolute, got: {self.input_type}"

        self.input_ref_frame = input_ref_frame
        assert self.input_ref_frame in [
            "world",
            "base",
        ], f"Input reference frame must be world or base, got: {self.input_ref_frame}"

        self.controller_type = type
        self.control_dim = 6 if self.use_ori else 3

        # input and output max and min (allow for either explicit lists or single numbers)
        self.input_max = self.nums2array(input_max, self.control_dim)
        self.input_min = self.nums2array(input_min, self.control_dim)
        self.output_max = self.nums2array(output_max, self.control_dim)
        self.output_min = self.nums2array(output_min, self.control_dim)
        self.max_translation = max_translation
        self.max_rotation = max_rotation

        # kp kd
        self.kp = self.nums2array(kp, 6)
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio

        # kp and kd limits
        self.kp_min = self.nums2array(kp_limits[0], 6)
        self.kp_max = self.nums2array(kp_limits[1], 6)
        self.damping_ratio_min = self.nums2array(damping_ratio_limits[0], 6)
        self.damping_ratio_max = self.nums2array(damping_ratio_limits[1], 6)

        # Verify the proposed impedance mode is supported
        assert impedance_mode in IMPEDANCE_MODES, "Error: Tried to instantiate OSC controller for unsupported " \
                                                  "impedance mode! Inputted impedance mode: {}, Supported modes: {}". \
            format(impedance_mode, IMPEDANCE_MODES)

        # Impedance mode
        self.impedance_mode = impedance_mode

        # Add to control dim based on impedance_mode
        if self.impedance_mode == "variable":
            self.control_dim += 12
        elif self.impedance_mode == "variable_kp":
            self.control_dim += 6

        # limits
        self.position_limits = np.array(
            position_limits) if position_limits is not None else position_limits
        self.orientation_limits = np.array(
            orientation_limits) if orientation_limits is not None else orientation_limits

        # control frequency
        self.control_freq = policy_freq

        # interpolator
        self.interpolator_pos = interpolator_pos
        self.interpolator_ori = interpolator_ori

        # whether or not pos and ori want to be uncoupled
        self.uncoupling = uncouple_pos_ori
        self.use_lambda = use_lambda
        self.use_nullspace = use_nullspace

        # initialize goals based on initial pos / ori
        self.goal_pos = None
        self.goal_ori = None

        self.relative_ori = np.zeros(3)
        self.ori_ref = None

        self.origin_ori = None
        self.origin_pos = None

    def set_goal(self, action, set_pos=None, set_ori=None):
        """
        Sets goal based on input @action. If self.impedance_mode is not "fixed", then the input will be parsed into the
        delta values to update the goal position / pose and the kp and/or damping_ratio values to be immediately updated
        internally before executing the proceeding control loop.

        Note that @action expected to be in the following format, based on impedance mode!

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Args:
            action (Iterable): Desired relative joint position goal state
            set_pos (Iterable): If set, overrides @action and sets the desired absolute eef position goal state
            set_ori (Iterable): IF set, overrides @action and sets the desired absolute eef orientation goal state
        """
        # Update state
        self.update()

        # # # TODO: parse action for variable impedance mode
        # # # Parse action based on the impedance mode, and update kp / kd as necessary
        # if self.impedance_mode == "variable":
        #     damping_ratio, kp, delta = action[:6], action[6:12], action[12:]
        #     self.kp = np.clip(kp, self.kp_min, self.kp_max)
        #     self.kd = 2 * np.sqrt(self.kp) * np.clip(damping_ratio,
        #                                              self.damping_ratio_min, self.damping_ratio_max)
        # elif self.impedance_mode == "variable_kp":
        #     kp, delta = action[:6], action[6:]
        #     self.kp = np.clip(kp, self.kp_min, self.kp_max)
        #     self.kd = 2 * np.sqrt(self.kp)  # critically damped
        # else:   # This is case "fixed"
        #     delta = action
        delta = np.zeros(self.control_dim)
        for i in range(self.control_dim):
            delta[i] = action[i]
            
        assert self.control_dim == len(action) -1 

        # Align actions to corresponding the enabled axis.
        # For example, 3D action space in XZPlane will be assigned to x-translation, z-translation and y-rotation.
        # delta = np.zeros_like(self.control_axis)
        # dim_count = 0
        # for i in range(len(self.control_axis)):
            # if self.control_axis[i] == 1:
            # delta[i] = action[dim_count]
            # dim_count += 1
        assert self.control_dim == len(delta)

        pos, ori = self._get_desired_pose(delta)

        if set_ori is not None:
            ori = set_ori
        if set_pos is not None:
            pos = set_pos
        # If the desired pose is reaching orientation limit or joint limit,
        # it will use the previous goal
        # if self.orientation_feasible(ori) and self.joint_limit_feasible(pos, ori):
        self.goal_ori = ori
        self.goal_pos = pos

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            # reference is the current orientation at start
            self.ori_ref = np.array(self.ref_ori_mat)
            # goal is the total orientation error
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref))
            # relative orientation always starts at 0
            self.relative_ori = np.zeros(3)

    def _get_desired_pose(self, action):
        # Get base pose
        if self.input_type == "delta":
            delta = action
            scaled_delta = self.scale_action(delta)
            pos = self.compute_goal_pos(scaled_delta[0:3])
            if self.use_ori is True:
                orn = self.compute_goal_ori(scaled_delta[3:6])
            else:
                orn = self.compute_goal_ori(np.zeros(3))
        # Else, interpret actions as absolute values
        elif self.input_type == "absolute":
            abs_action = action
            pos = abs_action[0:3]
            if self.use_ori is True:
                orn = Rotation.from_rotvec(
                    abs_action[3:6]).as_matrix()
            else:
                orn = self.compute_goal_ori(np.zeros(3))
        else:
            raise ValueError(f"Unsupport input_type {self.input_type}")

        # # Limit desired pose within the bounding box
        if self.position_limits is not None:
            if np.any(pos > self.position_limits[1]) or np.any(pos < self.position_limits[0]):
                if VERBOSE:
                    print('Reaching translation limit.')
                pos = np.minimum(self.position_limits[1], pos)
                pos = np.maximum(self.position_limits[0], pos)

        if self.use_ori:
            # divide by sqrt(3) because the norm of the max action is larger than dim=1.
            orn = (T.quat2mat(T.axisangle2quat(
                action[3:] * self.max_rotation/np.sqrt(3)))).dot(orn)

        return pos, orn

    def run_controller(self):
        """
        Calculates the torques required to reach the desired setpoint.

        Executes Operational Space Control (OSC) -- either position only or position and orientation.

        A detailed overview of derivation of OSC equations can be seen at:
        http://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf

        Returns:
             np.array: Command torques
        """
        # Update state
        self.update()

        desired_pos = None
        # Only linear interpolator is currently supported
        if self.interpolator_pos is not None:
            # Linear case
            if self.interpolator_pos.order == 1:
                desired_pos = self.interpolator_pos.get_interpolated_goal()
            else:
                # Nonlinear case not currently supported
                pass
        else:
            if self.input_ref_frame == "base":
                # compute goal based on current base position and orientation
                desired_world_pos = self.origin_pos + \
                    np.dot(self.origin_ori, self.goal_pos)
            elif self.input_ref_frame == "world":
                desired_world_pos = self.goal_pos
            else:
                raise ValueError

        if self.interpolator_ori is not None:
            # relative orientation based on difference between current ori and ref
            self.relative_ori = orientation_error(
                self.ref_ori_mat, self.ori_ref)

            ori_error = self.interpolator_ori.get_interpolated_goal()
        else:
            if self.input_ref_frame == "base":
                # compute goal based on current base orientation
                desired_world_ori = np.dot(self.origin_ori, self.goal_ori)
            elif self.input_ref_frame == "world":
                desired_world_ori = self.goal_ori
            else:
                raise ValueError
            ori_error = orientation_error(desired_world_ori, self.ref_ori_mat)

        position_error = desired_world_pos - self.ref_pos
        base_pos_vel = np.array(self.sim.data.get_site_xvelp(
            f"{self.naming_prefix}{self.part_name}_center"))
        vel_pos_error = -(self.ref_pos_vel - base_pos_vel)

        # F_r = kp * pos_err + kd * vel_err
        desired_force = np.multiply(np.array(position_error), np.array(self.kp[0:3])) + np.multiply(
            vel_pos_error, self.kd[0:3]
        )

        base_ori_vel = np.array(self.sim.data.get_site_xvelr(
            f"{self.naming_prefix}{self.part_name}_center"))
        vel_ori_error = -(self.ref_ori_vel - base_ori_vel)

        # Tau_r = kp * ori_err + kd * vel_err
        desired_torque = np.multiply(np.array(ori_error), np.array(self.kp[3:6])) + np.multiply(
            vel_ori_error, self.kd[3:6]
        )

        # Compute nullspace matrix (I - Jbar * J) and lambda matrices ((J * M^-1 * J^T)^-1)
        lambda_full, lambda_pos, lambda_ori, nullspace_matrix = opspace_matrices(
            self.mass_matrix, self.J_full, self.J_pos, self.J_ori
        )

        # Decouples desired positional control from orientation control
        if self.uncoupling:
            decoupled_force = np.dot(lambda_pos, desired_force)
            decoupled_torque = np.dot(lambda_ori, desired_torque)
            decoupled_wrench = np.concatenate(
                [decoupled_force, decoupled_torque])
        else:
            desired_wrench = np.concatenate([desired_force, desired_torque])
            decoupled_wrench = np.dot(lambda_full, desired_wrench)

        # Gamma (without null torques) = J^T * F + gravity compensations
        self.torques = np.dot(
            self.J_full.T, decoupled_wrench) + self.torque_compensation
        # Calculate and add nullspace torques (nullspace_matrix^T * Gamma_null) to final torques
        # Note: Gamma_null = desired nullspace pose torques, assumed to be positional joint control relative
        #                     to the initial joint positions
        self.torques += nullspace_torques(
            self.mass_matrix, nullspace_matrix, self.initial_joint, self.joint_pos, self.joint_vel
        )

        # Always run superclass call for any cleanups at the end
        super().run_controller()

        return self.torques

    def world_to_origin_frame(self, vec):
        """
        transform vector from world to reference coordinate frame
        """

        # world rotation matrix is just identity
        world_frame = np.eye(4)
        world_frame[:3, 3] = vec

        origin_frame = T.make_pose(self.origin_pos, self.origin_ori)
        origin_frame_inv = T.pose_inv(origin_frame)
        vec_origin_pose = T.pose_in_A_to_pose_in_B(
            world_frame, origin_frame_inv)
        vec_origin_pos, _ = T.mat2pose(vec_origin_pose)
        return vec_origin_pos

    def goal_origin_to_eef_pose(self):
        origin_pose = T.make_pose(self.origin_pos, self.origin_ori)
        ee_pose = T.make_pose(self.ref_pos, self.ref_ori_mat)
        origin_pose_inv = T.pose_inv(origin_pose)
        return T.pose_in_A_to_pose_in_B(ee_pose, origin_pose_inv)

    def compute_goal_pos(self, delta, goal_update_mode=None):
        """
        Compute new goal position, given a delta to update. Can either update the new goal based on
        current achieved position or current deisred goal. Updating based on current deisred goal can be useful
        if we want the robot to adhere with a sequence of target poses as closely as possible,
        without lagging or overshooting.

        Args:
            delta (np.array): Desired relative change in position [x, y, z]
            goal_update_mode (str): either "achieved" (achieved position) or "desired" (desired goal)

        Returns:
            np.array: updated goal position in the controller frame
        """
        if goal_update_mode is None:
            goal_update_mode = self._goal_update_mode
        assert goal_update_mode in ["achieved", "desired"]

        if self.goal_pos is None:
            # if goal is not already set, set it to current position (in controller ref frame)
            if self.input_ref_frame == "base":
                self.goal_pos = self.world_to_origin_frame(self.ref_pos)
            elif self.input_ref_frame == "world":
                self.goal_pos = self.ref_pos
            else:
                raise ValueError

        if goal_update_mode == "desired":
            # update new goal wrt current desired goal
            goal_pos = self.goal_pos + delta
        elif goal_update_mode == "achieved":
            # update new goal wrt current achieved position
            if self.input_ref_frame == "base":
                goal_pos = self.world_to_origin_frame(self.ref_pos) + delta
            elif self.input_ref_frame == "world":
                goal_pos = self.ref_pos + delta
            else:
                raise ValueError

        if self.position_limits is not None:
            # to be implemented later
            raise NotImplementedError

        return goal_pos

    def compute_goal_ori(self, delta, goal_update_mode=None):
        """
        Compute new goal orientation, given a delta to update. Can either update the new goal based on
        current achieved position or current deisred goal. Updating based on current deisred goal can be useful
        if we want the robot to adhere with a sequence of target poses as closely as possible,
        without lagging or overshooting.

        Args:
            delta (np.array): Desired relative change in orientation, in axis-angle form [ax, ay, az]
            goal_update_mode (str): either "achieved" (achieved position) or "desired" (desired goal)

        Returns:
            np.array: updated goal orientation in the controller frame
        """
        if goal_update_mode is None:
            goal_update_mode = self._goal_update_mode
        assert goal_update_mode in ["achieved", "desired"]

        if self.goal_ori is None:
            # if goal is not already set, set it to current orientation (in controller ref frame)
            if self.input_ref_frame == "base":
                self.goal_ori = self.goal_origin_to_eef_pose()[:3, :3]
            elif self.input_ref_frame == "world":
                self.goal_ori = self.ref_ori_mat
            else:
                raise ValueError

        # convert axis-angle value to rotation matrix
        quat_error = T.axisangle2quat(delta)
        rotation_mat_error = T.quat2mat(quat_error)

        if self._goal_update_mode == "desired":
            # update new goal wrt current desired goal
            goal_ori = np.dot(rotation_mat_error, self.goal_ori)
        elif self._goal_update_mode == "achieved":
            # update new goal wrt current achieved orientation
            if self.input_ref_frame == "base":
                curr_goal_ori = self.goal_origin_to_eef_pose()[:3, :3]
            elif self.input_ref_frame == "world":
                curr_goal_ori = self.ref_ori_mat
            else:
                raise ValueError
            goal_ori = np.dot(rotation_mat_error, curr_goal_ori)
        else:
            raise ValueError

        # check for orientation limits
        if np.array(self.orientation_limits).any():
            # to be implemented later
            raise NotImplementedError
        return goal_ori

    def update_initial_joints(self, initial_joints):
        # First, update from the superclass method
        super().update_initial_joints(initial_joints)

        # We also need to reset the goal in case the old goals were set to the initial confguration
        self.reset_goal()

    def reset_goal(self, goal_update_mode="achieved"):
        """
        Resets the goal to the current state of the robot.

        Args:
            goal_update_mode (str): set mode for updating controller goals,
                either "achieved" (achieved position) or "desired" (desired goal).
        """
        self.goal_ori = np.array(self.ref_ori_mat)
        self.goal_pos = np.array(self.ref_pos)

        assert goal_update_mode in ["achieved", "desired"]
        self._goal_update_mode = goal_update_mode

        # Also reset interpolators if required

        if self.interpolator_pos is not None:
            self.interpolator_pos.set_goal(self.goal_pos)

        if self.interpolator_ori is not None:
            # reference is the current orientation at start
            self.ori_ref = np.array(self.ref_ori_mat)
            self.interpolator_ori.set_goal(
                orientation_error(self.goal_ori, self.ori_ref)
            )  # goal is the total orientation error
            # relative orientation always starts at 0
            self.relative_ori = np.zeros(3)

    @property
    def control_limits(self):
        """
        Returns the limits over this controller's action space, overrides the superclass property
        Returns the following (generalized for both high and low limits), based on the impedance mode:

            :Mode `'fixed'`: [joint pos command]
            :Mode `'variable'`: [damping_ratio values, kp values, joint pos command]
            :Mode `'variable_kp'`: [kp values, joint pos command]

        Returns:
            2-tuple:

                - (np.array) minimum action values
                - (np.array) maximum action values
        """
        if self.impedance_mode == "variable":
            low = np.concatenate(
                [self.damping_ratio_min, self.kp_min, self.input_min])
            high = np.concatenate(
                [self.damping_ratio_max, self.kp_max, self.input_max])
        elif self.impedance_mode == "variable_kp":
            low = np.concatenate([self.kp_min, self.input_min])
            high = np.concatenate([self.kp_max, self.input_max])
        else:  # This is case "fixed"
            low, high = self.input_min, self.input_max
        return low, high

    @property
    def name(self):
        return self.controller_type

    def orientation_feasible(self, orn):
        quat = T.convert_quat(T.mat2quat(
            orn.dot(self.initial_ee_ori_mat)), to='wxyz')
        angle = np.arccos(min(abs(quat[0]), 1))/np.pi*180*2
        assert self.orientation_limits.shape == (), 'Only taking a scalar limit for now.'
        if angle > self.orientation_limits:
            if VERBOSE:
                print('Desired_pose reaching orientation limit')
            return False

        return True

    def joint_limit_feasible(self, desired_pos, desired_ori):
        # joint limit from frankapy
        JOINT_LIMITS_MIN = np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]) + 0.1
        JOINT_LIMITS_MAX = np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]) - 0.1

        delta_pos = desired_pos - self.ee_pos
        delta_ori = orientation_error(desired_ori, self.ee_ori_mat)

        J_inv = np.linalg.pinv(self.J_full)
        new_joint_pos = self.joint_pos + \
            np.dot(J_inv, np.concatenate([delta_pos, delta_ori]))

        if np.any(new_joint_pos < JOINT_LIMITS_MIN) or np.any(new_joint_pos > JOINT_LIMITS_MAX):
            if VERBOSE:
                print('Desired_pose reaching joint limit:')
                print('min:\t', JOINT_LIMITS_MIN)
                print('new:\t', new_joint_pos)
                print('max:\t', JOINT_LIMITS_MAX)
            return False

        return True
