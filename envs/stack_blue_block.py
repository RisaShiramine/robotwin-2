from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_blue_block(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        red_center_pose = sapien.Pose([0, -0.13, 0.75 + self.table_z_bias], [0, 1, 0, 0])
        green_on_red_pose = sapien.Pose([0, -0.13, 0.75 + self.table_z_bias + 0.05], [0, 1, 0, 0])

        blue_pose = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.08, 0.05],
            zlim=[0.741 + block_half_size],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        while (abs(blue_pose.p[0]) < 0.05
               or np.sum(pow(blue_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
               or np.sum(pow(blue_pose.p[:2] - np.array([0, -0.13]), 2)) < 0.01):
            blue_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.red_block = create_block(red_center_pose, (1, 0, 0))
        self.green_block = create_block(green_on_red_pose, (0, 1, 0))
        self.blue_block = create_block(blue_pose, (0, 0, 1))

        self.blocks = [self.red_block, self.green_block, self.blue_block]

        for block in self.blocks:
            self.add_prohibit_area(block, padding=0.07)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def play_once(self):
        block_pose = self.blue_block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        self.move(self.grasp_actor(self.blue_block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        green_pose = self.green_block.get_pose().p
        target_pose = [green_pose[0], green_pose[1], green_pose[2] + 0.05, 0, 1, 0, 0]
        self.move(
            self.place_actor(
                self.blue_block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        red_pose = self.red_block.get_pose().p
        green_pose = self.green_block.get_pose().p
        blue_pose = self.blue_block.get_pose().p
        center_target = np.array([0.0, -0.13])
        center_eps = [0.03, 0.03]
        stack_eps = [0.025, 0.025, 0.012]

        red_on_center = np.all(abs(red_pose[:2] - center_target) < center_eps)
        green_on_red = np.all(abs(green_pose - np.array(red_pose[:2].tolist() + [red_pose[2] + 0.05])) < stack_eps)
        blue_on_green = np.all(abs(blue_pose - np.array(green_pose[:2].tolist() + [green_pose[2] + 0.05])) < stack_eps)

        return (red_on_center and green_on_red and blue_on_green
                and self.is_left_gripper_open() and self.is_right_gripper_open())
