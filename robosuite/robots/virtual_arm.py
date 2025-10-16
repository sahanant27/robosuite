import numpy as np

from collections import OrderedDict

import robosuite.utils.transform_utils as T

from robosuite.models.grippers import gripper_factory
# TODO check if this is correct
from robosuite.controllers import controller_factory, load_part_controller_config

from robosuite.robots import FixedBaseRobot
from robosuite.utils.buffers import DeltaBuffer, RingBuffer
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.mjcf_utils import array_to_string, find_elements
import robosuite
import os
import copy
from robosuite.controllers.parts.arm.osc_delta import OSCDelta


class VirtualArm(FixedBaseRobot):
    """
    Initializes a single-armed robot simulation object.

    Args:
        robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

        idn (int or str): Unique ID of this robot. Should be different from others

        controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
            Else, uses the default controller for this specific task

        initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
            instantiated for the task

        initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
            corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            :Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

        mount_type (str): type of mount, used to instantiate mount models from mount factory.
            Default is "default", which is the default mount associated with this robot's corresponding model.
            None results in no mount, and any other (valid) model overrides the default mount.

        gripper_type (str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default gripper associated
            within the 'robot' specification. None removes the gripper, and any other (valid) model overrides the
            default gripper

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        composite_controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        base_type="default",
        gripper_type="default",
        control_freq=20,
        control_gripper=True,
        gripper_friction=(3, 0.015, 0.0003),
        close_gripper=False,
        lite_physics=True
    ):
        self.robot_type = robot_type
        self.controller = None
        self.controller_config = copy.deepcopy(composite_controller_config)
        self.gripper_type = gripper_type
        self.has_gripper = self.gripper_type is not None
        self.control_gripper = control_gripper and self.has_gripper
        self.gripper_friction = np.array(gripper_friction)

        self.gripper = None                                 # Gripper class
        # xml joint names for gripper
        self.gripper_joints = None
        # keep the gripper close during the manipulation
        self.close_gripper = close_gripper
        # xml gripper joint position indexes in mjsim
        self._ref_gripper_joint_pos_indexes = None
        # xml gripper joint velocity indexes in mjsim
        self._ref_gripper_joint_vel_indexes = None
        # xml gripper (pos) actuator indexes for robot in mjsim
        self._ref_joint_gripper_actuator_indexes = None
        # Current torques being applied
        self.torques = None

        # Current and last forces / torques sensed at eef
        self.recent_ee_forcetorques = None
        # Current and last eef pose (pos + ori (quat))
        self.recent_ee_pose = None
        # Current and last eef velocity
        self.recent_ee_vel = None
        # RingBuffer holding prior 10 values of velocity values
        self.recent_ee_vel_buffer = None
        # Current and last eef acceleration
        self.recent_ee_acc = None

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
            base_type=base_type,
            control_freq=control_freq,
            composite_controller_config=composite_controller_config,
            gripper_type=gripper_type,
            lite_physics=lite_physics,
        )

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

        # Verify that the loaded model is of the correct type for this robot
        if self.robot_model.arm_type != "single":
            raise TypeError("Error loading robot model: Incompatible arm type specified for this robot. "
                            "Requested model arm type: {}, robot arm type: {}"
                            .format(self.robot_model.arm_type, type(self)))

        # Make the gripper stiff so that it won't move after collision
        if not self.control_gripper and self.gripper_type["right"] == 'default':
            self.gripper["right"].actuator[0].set("kp", str(1e6))
            self.gripper["right"].actuator[1].set("kp", str(1e6))

        # Increase default friction
        if self.gripper_type["right"] == 'panda_narrow':
            gripper_geoms = find_elements(
                self.gripper["right"].root, 'geom', return_first=False)
            for geom in gripper_geoms:
                # if "collision" in geom.get("name"):
                if 'pad_outer_collision' in geom.get("name") or 'finger1_collision' in geom.get("name") or 'finger2_collision' in geom.get("name"):
                    geom.set("friction", array_to_string(
                        self.gripper_friction))
        # https://github.com/marcocognetti/FrankaEmikaPandaDynModel/blob/master/pdf/RA-L_2019_PandaDynIdent_SUPPLEMENTARY_MATERIAL.pdf
        if self.robot_type == 'Panda':
            damping = np.array(
                [0.0628, 0.2088, 0.0361, 0.2174, 0.1021, 1.6128e-04, 0.0632])
            frictionloss = np.array(
                [5.4615e-01, 0.87224, 6.4068e-01, 1.2794e+00, 8.3904e-01, 3.0301e-01, 5.6489e-01])/2
            self.robot_model.set_joint_attribute(
                attrib="damping", values=damping, force=True)
            self.robot_model.set_joint_attribute(
                attrib="frictionloss", values=frictionloss, force=True)
        # TODO: https://github.com/frankaemika/franka_ros/blob/develop/franka_description/robots/panda_gazebo.xacro

        # Add gripper to this robot model
        # self.robot_model.add_gripper(self.gripper)

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and controller
        # deterministic = True
        super().reset(deterministic)

      
    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()


    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should be
                the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        # """

        super().control(action, policy_step)

    def grip_action(self, gripper, gripper_action):
        """
        Executes @gripper_action for specified @gripper

        Args:
            gripper (GripperModel): Gripper to execute action for
            gripper_action (float): Value between [-1,1] to send to gripper
        """
        actuator_idxs = [self.sim.model.actuator_name2id(
            actuator) for actuator in gripper.actuators]
        gripper_action_actual = gripper.format_action(gripper_action)
        # rescale normalized gripper action to control ranges
        ctrl_range = self.sim.model.actuator_ctrlrange[actuator_idxs]
        bias = 0.5 * (ctrl_range[:, 1] + ctrl_range[:, 0])
        weight = 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        applied_gripper_action = bias + weight * gripper_action_actual
        self.sim.data.ctrl[actuator_idxs] = applied_gripper_action

    def visualize(self, vis_settings):
        """
        Do any necessary visualization for this manipulator

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "robots" and "grippers" keyword as well as any other
                robot-specific options specified.
        """
        super().visualize(vis_settings=vis_settings)
        self._visualize_grippers(visible=vis_settings["grippers"])

    def _visualize_grippers(self, visible):
        """
        Visualizes the gripper site(s) if applicable.

        Args:
            visible (bool): True if visualizing the gripper for this arm.
        """
        self.gripper["right"].set_sites_visibility(
            sim=self.sim, visible=visible)

    def setup_observables(self):
        """
        Sets up observables to be used for this robot

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        # Get general robot observables first
        observables = super().setup_observables()

        # Get prefix from robot model to avoid naming clashes for multiple robots and define observables modality
        pf = self.robot_model.naming_prefix
        modality = f"{pf}proprio"

        # # finger-tip
        # @sensor(modality=modality)
        # def eef_pos(obs_cache):
        #     return self.part_controllers["right"].ee_pos

        # @sensor(modality=modality)
        # def eef_quat(obs_cache):
        #     return T.convert_quat(T.mat2quat(self.part_controllers["right"].ee_ori_mat), to='wxyz')

        # @sensor(modality=modality)
        # def eef_pos_vel(obs_cache):
        #     return self.part_controllers["right"].ee_pos_vel

        # @sensor(modality=modality)
        # def eef_ori_vel(obs_cache):
        #     return self.part_controllers["right"].ee_ori_vel

        # @sensor(modality=modality)
        # def joint_torque(obs_cache):
        #     return self.part_controllers["right"].torques

        # sensors = [eef_pos, eef_quat, eef_pos_vel, eef_ori_vel, joint_torque]
        # names = [f"{pf}eef_pos", f"{pf}eef_quat", f"{pf}eef_pos_vel",
        #          f"{pf}eef_ori_vel", f"{pf}joint_torque"]

        if 'OSC' in self.part_controllers["right"].name:
            @sensor(modality=modality)
            def osc_desired_pos(obs_cache):
                return self.part_controllers["right"].goal_pos

            @sensor(modality=modality)
            def osc_desired_quat(obs_cache):
                return T.convert_quat(T.mat2quat(self.part_controllers["right"].goal_ori), to='wxyz')

            # sensors += [osc_desired_pos, osc_desired_quat]4
            sensors = [osc_desired_pos, osc_desired_quat]

            # names += [f"{pf}osc_desired_pos", f"{pf}osc_desired_quat"]
            names = [f"{pf}osc_desired_pos", f"{pf}osc_desired_quat"]

        # Create observables for this robot
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    @property
    def dof(self):
        """
        Returns:
            int: degrees of freedom of the robot (with grippers).
        """
        # Get the dof of the base robot model
        dof = super().dof
        if self.control_gripper:
            for gripper in self.robot_model.grippers.values():
                dof += gripper.dof
        return dof
