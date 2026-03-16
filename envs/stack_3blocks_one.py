from ._base_task import Base_Task
from .utils import *
import sapien
import math


class stack_3blocks_one(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        block_pose_lst = []

        for i in range(3):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(_block_pose):
                for j in range(len(block_pose_lst)):
                    if np.sum(pow(_block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01:
                        return False
                return True

            while (abs(block_pose.p[0]) < 0.05
                   or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
                   or not check_block_pose(block_pose)):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            block_pose_lst.append(deepcopy(block_pose))

        color_options = [
            ((1, 0, 0), "red"),
            ((0, 1, 0), "green"),
            ((0, 0, 1), "blue"),
        ]
        color_ids = np.random.permutation(len(color_options))
        block_colors = [color_options[i] for i in color_ids]

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], block_colors[0][0])
        self.block2 = create_block(block_pose_lst[1], block_colors[1][0])
        self.block3 = create_block(block_pose_lst[2], block_colors[2][0])
        self.block_color_names = [block_colors[0][1], block_colors[1][1], block_colors[2][1]]

        self.blocks = [self.block1, self.block2, self.block3]
        self.target_block_index = np.random.randint(0, 3)

        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        self.add_prohibit_area(self.block3, padding=0.07)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def play_once(self):
        target_block = self.blocks[self.target_block_index]
        target_color_name = self.block_color_names[self.target_block_index]

        block_pose = target_block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        self.move(self.grasp_actor(target_block, arm_tag=arm_tag, pre_grasp_dis=0.09))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        self.move(
            self.place_actor(
                target_block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.info["info"] = {
            "{A}": f"{target_color_name} block",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        target = np.array([0.0, -0.13])
        eps = [0.03, 0.03]

        block1_on_center = np.all(abs(block1_pose[:2] - target) < eps)
        block2_on_center = np.all(abs(block2_pose[:2] - target) < eps)
        block3_on_center = np.all(abs(block3_pose[:2] - target) < eps)

        return ((block1_on_center or block2_on_center or block3_on_center)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
