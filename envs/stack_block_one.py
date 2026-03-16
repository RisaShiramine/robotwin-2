from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_block_one(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        block_pose = rand_pose(
            xlim=[-0.28, 0.28],
            ylim=[-0.08, 0.05],
            zlim=[0.741 + block_half_size],
            qpos=[1, 0, 0, 0],
            ylim_prop=True,
            rotate_rand=True,
            rotate_lim=[0, 0, 0.75],
        )

        while (abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

        color_options = [
            ((1, 0, 0), "red"),
            ((0, 1, 0), "green"),
            ((0, 0, 1), "blue"),
        ]
        color_rgb, self.block_color_name = color_options[np.random.randint(0, 3)]

        self.block1 = create_box(
            scene=self,
            pose=block_pose,
            half_size=(block_half_size, block_half_size, block_half_size),
            color=color_rgb,
            name="box",
        )
        self.add_prohibit_area(self.block1, padding=0.07)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def play_once(self):
        block_pose = self.block1.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        self.move(self.grasp_actor(self.block1, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        self.move(
            self.place_actor(
                self.block1,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.info["info"] = {
            "{A}": f"{self.block_color_name} block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        target = np.array([0.0, -0.13])
        eps = [0.03, 0.03]

        return (np.all(abs(block1_pose[:2] - target) < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
