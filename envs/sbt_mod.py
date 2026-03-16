from ._base_task import Base_Task
from .utils import *


class sbt_mod(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        block_pose_lst = []
        for _ in range(3):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(pose):
                for exist_pose in block_pose_lst:
                    if np.sum(pow(pose.p[:2] - exist_pose.p[:2], 2)) < 0.01:
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

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.block3 = create_block(block_pose_lst[2], (0, 0, 1))

        self.block_colors = {
            self.block1: (1.0, 0.0, 0.0),
            self.block2: (0.0, 1.0, 0.0),
            self.block3: (0.0, 0.0, 1.0),
        }

        self.add_prohibit_area(self.block1, padding=0.05)
        self.add_prohibit_area(self.block2, padding=0.05)
        self.add_prohibit_area(self.block3, padding=0.05)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def _iter_render_bodies(self, block: Actor):
        entity = block.actor if hasattr(block, "actor") else block
        for component in entity.get_components():
            comp_name = type(component).__name__
            if "RenderBodyComponent" not in comp_name:
                continue
            yield component

    def _iter_render_shapes(self, block: Actor):
        for component in self._iter_render_bodies(block):
            if hasattr(component, "get_render_shapes"):
                shapes = component.get_render_shapes()
            elif hasattr(component, "render_shapes"):
                shapes = component.render_shapes
            else:
                shapes = []

            for shape in shapes:
                yield shape

    def _set_block_alpha(self, block: Actor, alpha: float):
        base_rgb = self.block_colors[block]
        target_color = [base_rgb[0], base_rgb[1], base_rgb[2], alpha]

        for component in self._iter_render_bodies(block):
            if hasattr(component, "set_visibility"):
                component.set_visibility(alpha)
            elif hasattr(component, "visibility"):
                component.visibility = alpha

        for shape in self._iter_render_shapes(block):
            material = None
            if hasattr(shape, "material"):
                material = shape.material
            elif hasattr(shape, "get_material"):
                material = shape.get_material()

            if material is None:
                continue

            if hasattr(material, "base_color"):
                material.base_color = target_color
            elif hasattr(material, "set_base_color"):
                material.set_base_color(target_color)

            if hasattr(shape, "set_material"):
                shape.set_material(material)

    def _show_blocks(self, visible_blocks: list[Actor]):
        visible_ids = {id(block) for block in visible_blocks}
        for block in [self.block1, self.block2, self.block3]:
            self._set_block_alpha(block, 1.0 if id(block) in visible_ids else 0.0)

    def play_once(self):
        self.last_gripper = None
        self.last_actor = None

        self._show_blocks([self.block1])
        arm_tag1 = self.pick_and_place_block(self.block1)

        self._show_blocks([self.block1, self.block2])
        arm_tag2 = self.pick_and_place_block(self.block2)

        self._show_blocks([self.block1, self.block2, self.block3])
        arm_tag3 = self.pick_and_place_block(self.block3)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{C}": "blue block",
            "{a}": str(arm_tag1),
            "{b}": str(arm_tag2),
            "{c}": str(arm_tag3),
        }
        return self.info

    def pick_and_place_block(self, block: Actor):
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(
                self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                self.back_to_origin(arm_tag=arm_tag.opposite),
            )
        else:
            self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        if self.last_actor is None:
            target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        else:
            target_pose = self.last_actor.get_functional_point(1)

        self.move(
            self.place_actor(
                block,
                target_pose=target_pose,
                arm_tag=arm_tag,
                functional_point_id=0,
                pre_dis=0.05,
                dis=0.,
                pre_dis_axis="fp",
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        block3_pose = self.block3.get_pose().p
        eps = [0.025, 0.025, 0.012]

        return (np.all(abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2] + 0.05])) < eps)
                and np.all(abs(block3_pose - np.array(block2_pose[:2].tolist() + [block2_pose[2] + 0.05])) < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
