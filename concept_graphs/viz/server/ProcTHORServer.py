from typing import Dict
from scipy.spatial.transform import Rotation as Rsc
import numpy as np
import viser

from concept_graphs.inference.toolbox.PerpetuaMapToolbox import PerpetuaMapToolbox
from concept_graphs.viz.server.PerpetuaMapServer import PerpetuaMapServer
from concept_graphs.utils import split_camel_preserve_acronyms, procthor_to_ros

import logging

log = logging.getLogger(__name__)


class PrivilegedServer(PerpetuaMapServer):
    def __init__(self, toolbox: PerpetuaMapToolbox, downsample_factor: int = 2, point_shape: str = "circle", stop_server: bool = False):
        super().__init__(toolbox, point_shape)

        # Hyperparams
        self.downsample_factor = downsample_factor
        self.stop_server = stop_server  
        
        # Containers
        self.open_vocab_query = None
        self.receptacle_id = None
        self.clip_query = ""

        # Handles
        self.agent_handle: Dict[str, viser.FrameHandle] = {}

        # Reset GUI
        self.server.gui.reset()
        # GUI
        self.tab_group = self.server.gui.add_tab_group()
        with self.tab_group.add_tab("User"):
            with self.server.gui.add_folder("Open-Vocab"):
                self.open_vocab_query_gui_text = self.server.gui.add_text(
                    "Query",
                    initial_value="",
                )
                self.open_vocab_query_button = self.server.gui.add_button(
                    "Submit Query",
                    icon=viser.Icon.MOUSE,
                )
                self.open_vocab_query_button.on_click(self.on_open_vocab_query_submit)

        with self.tab_group.add_tab("Agent"):
            # User controls to test specific functions
            self.object_dropdown = self.server.gui.add_dropdown(
                "Pickupable",
                options=self.object_names,
                initial_value=None,
            )
            self.object_dropdown.on_update(self.on_object_dropdown_update)
            self.receptacle_dropdown = self.server.gui.add_dropdown(
                "Receptacle",
                options=[""],
                initial_value=None,
            )
            # Allow to modify the content here easily
            self._setup_user_tab_header()

            self.map_gui_reset_button = self.server.gui.add_button(
                "Reset Map",
                icon=viser.Icon.MOUSE,
            )
            self.map_gui_reset_button.on_click(self.on_map_time_reset_click)

        if self.stop_server:
            log.info("Viewer disabled (stop_server=true).")
            self.server.stop()

    def _setup_user_tab_header(self):
        # CLIP Query
        with self.server.gui.add_folder("CLIP Query"):
            self.clip_gui_text = self.server.gui.add_text(
                "Object Query",
                initial_value="",
            )
            self.clip_gui_button = self.server.gui.add_button(
                "CLIP Similarities",
                icon=viser.Icon.MOUSE,
            )
            self.clip_gui_button.on_click(self.on_clip_query_submit)
        # Object Time Prediction
        with self.server.gui.add_folder("Predict Object at Time"):
            self.object_time_gui_number = self.server.gui.add_number(
                "Time (hours)", initial_value=0, min=0
            )
            self.object_time_gui_button = self.server.gui.add_button(
                "Predict at Time",
                icon=viser.Icon.MOUSE,
            )
            self.object_time_gui_button.on_click(self.on_object_time_query_submit)
        # Go to Object
        with self.server.gui.add_folder("Go to Object"):
            self.go_to_object_button = self.server.gui.add_button(
                "Go to Receptacle",
                icon=viser.Icon.MOUSE,
            )
            self.go_to_object_button.on_click(self.on_go_to_object_submit)

    def reset(self):
        super().reset()
        self.agent_handle = {}

    # Gui callbacks and inputs
    def on_open_vocab_query_submit(self, data):
        self.open_vocab_query = self.open_vocab_query_gui_text.value

    def on_object_dropdown_update(self, data):
        pickupable_name = self.object_dropdown.value
        pickupable = self.object_map.get_pickupable(pickupable_name)
        self.receptacle_dropdown.options = pickupable.receptacles

    def on_map_time_reset_click(self, data):
        self.toolbox.reset_temporal_edges()
        self._update_server_state()

    def on_go_to_object_submit(self, data):
        self.selected_object_id = self.object_dropdown.value
        self.receptacle_id = self.receptacle_dropdown.value

    def spin(self):
        # The agent is the one that spins
        pass

    def display_query_object(
        self,
        pickupable_name: str,
        receptacle_name: str,
        color: np.ndarray,
    ):
        # First display everything in RGB to color previous moved objects
        self.clear_labels()
        self.display_object_rgb()
        # Get object
        obj = self.object_map.get_pickupable(pickupable_name)
        # Get handles of object if they exist
        object_handle = self.object_handles.get(f"objects/{pickupable_name}", None)
        hitbox_handle = self.hitbox_handles.get(f"hitbox/{pickupable_name}", None)

        # Common data
        points = np.asarray(obj.pcd.points)
        bbox = obj.pcd.get_oriented_bounding_box()
        wxyz = Rsc.from_matrix(np.array(bbox.R, copy=True)).as_quat(scalar_first=True)
        visibility = obj.visibility

        # Update object point cloud
        if object_handle:
            object_handle.points = points
            object_handle.colors = color
            object_handle.visible = visibility
        else:
            obj_handle = self.server.scene.add_point_cloud(
                f"objects/{pickupable_name}",
                points,
                color=color,
                point_size=self.pcd_size_gui_slider.value,
                point_shape=self.point_shape,
                visible=visibility,
            )
            self.object_handles[f"objects/{pickupable_name}"] = obj_handle
        # Update hitbox position
        if hitbox_handle:
            hitbox_handle.position = bbox.center
            hitbox_handle.dimensions = bbox.extent.tolist()
            hitbox_handle.wxyz = wxyz
            hitbox_handle.visible = visibility
        else:
            hitbox = self.server.scene.add_box(
                name=f"hitbox/{pickupable_name}",
                dimensions=bbox.extent.tolist(),
                position=bbox.center,
                wxyz=wxyz,
                color=(255, 255, 255),
                opacity=0.0,
                visible=visibility,
            )
            self.hitbox_handles[f"hitbox/{pickupable_name}"] = hitbox

        # Display name
        if visibility:
            centroid = obj.centroid
            label_text = f"{split_camel_preserve_acronyms(pickupable_name.split('|')[0])} @ {split_camel_preserve_acronyms(receptacle_name.split('|')[0])}"
            label_handle = self.server.scene.add_label(
                name=f"label/{pickupable_name}",
                text=label_text,
                position=centroid,
                visible=obj.visibility,
            )
            self.label_handles[f"label/{pickupable_name}"] = label_handle

    def display_agent(self, pose: Dict[str, float], img: np.ndarray = None, intrinsics: Dict[str, float] = None):
        # Show clean RGB in case something changes in the map
        self.display_object_rgb()
        ros_pose = procthor_to_ros(pose)
        position = ros_pose[0:3, 3]
        wxyz = Rsc.from_matrix(ros_pose[0:3, 0:3]).as_quat(scalar_first=True)
        frame = self.server.scene.add_frame(
            "tree/agent_frame",
            wxyz=wxyz,
            position=position,
            axes_length=0.1,
            axes_radius=0.005,
        )
        H, W = img.shape[:2]
        downsampled_img = img[::self.downsample_factor, ::self.downsample_factor]
        frustum = self.server.scene.add_camera_frustum(
            f"tree/agent_frame/frustum",
            fov=intrinsics["fov_degrees"] * np.pi / 180.0,
            aspect=W / H,
            scale=0.15,
            image=downsampled_img,
        )
        self.agent_handle["tree/agent_frame"] = frame

        @frustum.on_click
        def _(_, frame=frame) -> None:
            for client in self.server.get_clients().values():
                client.camera.wxyz = frame.wxyz
                client.camera.position = frame.position


    def display_llm_response(self, msg: str):
        for client in self.server.get_clients().values():
            client.add_notification(
                title="LLM Response",
                body=msg,
                auto_close_seconds=10.0,
            )

        
