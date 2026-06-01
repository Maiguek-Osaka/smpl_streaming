#!/usr/bin/env python3

from __future__ import annotations

import os
import zmq
import time
import torch
import argparse
import threading
import numpy as np
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R, Rotation as sRot

from utils.rotation_conversion import decompose_rotation_aa
from utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up
from utils.g1_gripper_ik_solver import G1GripperInverseKinematicsSolver

from kimodo.kimodo.exports.smplx import get_amass_parameters, amass_arrays_to_kimodo_motion
from kimodo.kimodo.skeleton.registry import build_skeleton
from kimodo.kimodo.postprocess import post_process_motion

from utils.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

from utils.torch_transform import (
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)

# Import the SMPL processing function from pico_manager_thread_server
def process_smpl_joints(body_pose, global_orient, transl, disable_ytoz=True):
    """Process SMPL parameters to compute local joints."""
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    
    # Toggleable Y-to-Z up conversion based on generator source
    if smpl_root_ytoz_up is not None and not disable_ytoz:
        global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
        
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )

    if remove_smpl_base_rot is not None:
        global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)

    return {
        "smpl_pose": body_pose,
        "joints": joints,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "adjusted_transl": transl,
    }


def load_smpl_raw(input_path: str, source_fps: str, post_process:bool = True, no_legs:bool = False, disable_ytoz:bool = True) -> dict[str, np.ndarray]:

    if input_path[-3:] == 'pkl':
        data = np.load(input_path, allow_pickle=True)
        smpl_poses, smpl_trans, full_pose = data['smpl_poses'], data['smpl_trans'], data['full_pose']

        global_orient = torch.from_numpy(smpl_poses[:, :3]).float()
        body_pose = torch.from_numpy(smpl_poses[:, 3:66]).float()
        transl = torch.from_numpy(smpl_trans).float()
    elif input_path[-2:] == 'pt':
        data = torch.load(input_path)
        body_pose = data['body_params_incam']['body_pose']
        global_orient = data['body_params_incam']['global_orient']
        transl = data['body_params_incam']['transl']
    else:
        data = np.load(input_path, allow_pickle=True)
        global_orient = torch.from_numpy(data['poses'][:, :3]).float()
        body_pose = torch.from_numpy(data['poses'][:, 3:66]).float()
        transl = torch.from_numpy(data['trans']).float()

    if post_process:
        sk = build_skeleton(22) # smplx22
        
        kimodo_motion_dict = amass_arrays_to_kimodo_motion(
            trans=transl,
            root_orient=global_orient,
            pose_body=body_pose,
            skeleton=sk,
            source_fps=source_fps,
            z_up=True,
        )

        local_rot_mats = kimodo_motion_dict['local_rot_mats'].unsqueeze(0)
        root_positions = kimodo_motion_dict['root_positions'].unsqueeze(0)
        contacts = kimodo_motion_dict['foot_contacts'].unsqueeze(0)
        
        post_processed_dict = post_process_motion(
            local_rot_mats=local_rot_mats,
            root_positions=root_positions,
            contacts=contacts,
            skeleton=sk,
        )

        transl, global_orient, body_pose = get_amass_parameters(
            local_rot_mats=post_processed_dict['local_rot_mats'],
            root_positions=post_processed_dict['root_positions'],
            skeleton=sk,
            z_up=True,
        )
        
        body_pose = torch.from_numpy(body_pose).float().squeeze(0)
        global_orient = torch.from_numpy(global_orient).float().squeeze(0)
        transl = transl.squeeze(0)

    # --- NO LEGS IMPLEMENTATION ---
    if no_legs:
        # Reshape body pose to access individual joints (T, 21, 3)
        body_pose_reshaped = body_pose.view(-1, 21, 3)
        
        # SMPL leg joint indices inside body_pose (excluding root):
        # Hips: 0, 1 | Knees: 3, 4 | Ankles: 6, 7 | Feet: 9, 10
        leg_indices = [0, 1, 3, 4, 6, 7, 9, 10]
        
        # Zero out the axis-angle rotations for all leg joints
        body_pose_reshaped[:, leg_indices, :] = 0.0
        
        # Flatten back to (T, 63)
        body_pose = body_pose_reshaped.view(-1, 63)
        print("--- Leg motions have been disabled (--no_legs active) ---")
    
    processed_smpl = process_smpl_joints(body_pose, global_orient, transl, disable_ytoz)
    return processed_smpl

def init_hand_ik_solvers():
    if G1GripperInverseKinematicsSolver is not None:
        left_solver = G1GripperInverseKinematicsSolver(side="left")
        right_solver = G1GripperInverseKinematicsSolver(side="right")
        return left_solver, right_solver
    return None, None

def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q

def _interp_pose_axis_angle(prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float) -> np.ndarray:
    prev_quats = sRot.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()
    curr_quats = sRot.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    out_pose = sRot.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)
    return out_pose

class PoseStreamer:
    def __init__(self, socket, processed_data: dict, num_frames_to_send: int, target_fps: int, use_cuda: bool, record_dir: str, record_format: str, log_prefix: str = "PoseLoop"):
        self.socket = socket
        self.num_frames_to_send = num_frames_to_send
        self.target_fps = target_fps
        self.record_dir = record_dir
        self.log_prefix = log_prefix
        
        self.num_frames = processed_data["smpl_pose"].shape[0]
        self.smpl_pose_seq = (processed_data["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(self.num_frames, 21, 3)).astype(np.float32)
        self.smpl_joints_seq = (processed_data["smpl_joints_local"].detach().cpu().numpy()).astype(np.float32)
        self.body_quat_seq = (processed_data["global_orient_quat"].detach().cpu().numpy()).astype(np.float32)
        
        self.device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")

        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        self.record_idx = 0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()
        
        self.step = 0
        self.last_fps_report = time.time()
        self.fps_counter = 0
        self.frame_time = 0.95 / max(1, target_fps)
        self.frame_buffer = defaultdict(lambda: deque(maxlen=num_frames_to_send))

        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.frame_start = time.time()
        self.buffer_cleared = True

    def on_mode_exit(self):
        self.frame_buffer.clear()
        self.prev_smpl_pose_np = None
        self.prev_smpl_joints_np = None
        self.prev_body_quat_np = None
        self.buffer_cleared = True
        self.step = 0

    def run_once(self):
        frame_idx = self.step % self.num_frames
        
        smpl_pose_np = self.smpl_pose_seq[frame_idx]
        smpl_joints_np = self.smpl_joints_seq[frame_idx]
        body_quat_np = self.body_quat_seq[frame_idx]

        left_hand_joints = np.zeros((1, 7), dtype=np.float32)
        right_hand_joints = np.zeros((1, 7), dtype=np.float32)

        alpha = 1.0 
        
        if self.prev_smpl_joints_np is not None:
            use_joints = (1.0 - alpha) * self.prev_smpl_joints_np + alpha * smpl_joints_np
            use_pose = _interp_pose_axis_angle(self.prev_smpl_pose_np, smpl_pose_np, alpha).astype(np.float32)
            use_body_quat = _quat_lerp_normalized(self.prev_body_quat_np, body_quat_np, alpha).astype(np.float32)
        else:
            use_joints = smpl_joints_np
            use_pose = smpl_pose_np
            use_body_quat = body_quat_np
        
        joint_pos = np.zeros(29)
        body_pose = use_pose.reshape(-1, 21, 3)

        SMPL_L_ELBOW_IDX, SMPL_L_WRIST_IDX = 17, 19
        SMPL_R_ELBOW_IDX, SMPL_R_WRIST_IDX = 18, 20

        G1_L_WRIST_ROLL_IDX, G1_L_WRIST_PITCH_IDX, G1_L_WRIST_YAW_IDX = 23, 25, 27
        G1_R_WRIST_ROLL_IDX, G1_R_WRIST_PITCH_IDX, G1_R_WRIST_YAW_IDX = 24, 26, 28
        
        smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
        smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
        smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
        smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

        g1_l_elbow_axis = np.array([0, 1, 0])
        g1_l_elbow_q_twist, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, g1_l_elbow_axis)
        g1_r_elbow_axis = np.array([0, 1, 0])
        g1_r_elbow_q_twist, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, g1_r_elbow_axis)

        l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
        r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler("XYZ", degrees=False)
        l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
        r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

        joint_pos[G1_L_WRIST_ROLL_IDX] = (l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0])[0]
        joint_pos[G1_L_WRIST_PITCH_IDX] = -(-l_wrist_euler[:, 1])[0]
        joint_pos[G1_L_WRIST_YAW_IDX] = (l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2])[0]

        joint_pos[G1_R_WRIST_ROLL_IDX] = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])[0]
        joint_pos[G1_R_WRIST_PITCH_IDX] = (-r_wrist_euler[:, 1])[0]
        joint_pos[G1_R_WRIST_YAW_IDX] = (r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2])[0]

        self.frame_buffer["smpl_pose"].append(use_pose)
        self.frame_buffer["smpl_joints"].append(use_joints)
        self.frame_buffer["body_quat_w"].append(use_body_quat)
        self.frame_buffer["frame_index"].append(int(self.step))
        self.frame_buffer["joint_pos"].append(joint_pos)

        N = len(self.frame_buffer["frame_index"])
        buffer_is_full = N >= self.num_frames_to_send
        if buffer_is_full and self.buffer_cleared:
            self.buffer_cleared = False

        if buffer_is_full and not self.buffer_cleared:
            numpy_data = {
                "smpl_pose": np.stack((self.frame_buffer["smpl_pose"]), axis=0),
                "smpl_joints": np.stack((self.frame_buffer["smpl_joints"]), axis=0),
                "body_quat_w": np.stack((self.frame_buffer["body_quat_w"]), axis=0),
                "joint_pos": np.stack((self.frame_buffer["joint_pos"]), axis=0),
                "joint_vel": np.zeros((N, 29)),
                "frame_index": np.array((self.frame_buffer["frame_index"]), dtype=np.int64),
                "left_hand_joints": left_hand_joints.reshape(-1).astype(np.float32),
                "right_hand_joints": right_hand_joints.reshape(-1).astype(np.float32),
            }

            packed_message = pack_pose_message(numpy_data, topic="pose")
            self.socket.send(packed_message)

            if self.record_dir:
                out_path = os.path.join(self.record_dir, f"pose_{self.record_idx:06d}.npz")
                np.savez_compressed(out_path, **numpy_data)
                self.record_idx += 1

        self.step += 1
        self.prev_smpl_pose_np = smpl_pose_np
        self.prev_smpl_joints_np = smpl_joints_np
        self.prev_body_quat_np = body_quat_np
        
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_report >= 5.0:
            fps = self.fps_counter / (current_time - self.last_fps_report)
            print(f"[{self.log_prefix}] FPS: {fps:.2f}, Step: {self.step}, Frame: {frame_idx}/{self.num_frames}")
            self.fps_counter = 0
            self.last_fps_report = current_time
            
        elapsed = time.time() - self.frame_start
        if elapsed < self.frame_time:
            time.sleep(self.frame_time - elapsed)
        self.frame_start = time.time()

def _pose_stream_common(processed_data, socket, buffer_size: int, num_frames_to_send: int, target_fps: int, use_cuda: bool, record_dir: str, record_format: str, stop_event: threading.Event | None = None, log_prefix: str = "PoseLoop", enable_smpl_vis: bool = False):
    streamer = PoseStreamer(socket=socket, processed_data=processed_data, num_frames_to_send=num_frames_to_send, target_fps=target_fps, use_cuda=use_cuda, record_dir=record_dir, record_format=record_format, log_prefix=log_prefix)
    if stop_event is None: stop_event = threading.Event()
    try:
        while not stop_event.is_set(): streamer.run_once()
    except KeyboardInterrupt: pass

def run_pico(processed_data, host="0.0.0.0", port=5556, target_fps=50, buffer_size: int = 15, num_frames_to_send: int = 5, record_dir: str = "", record_format: str = "npz", use_cuda: bool = False):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://{host}:{port}")
    time.sleep(0.1)

    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(processed_data=processed_data, socket=socket, buffer_size=buffer_size, num_frames_to_send=num_frames_to_send, target_fps=target_fps, use_cuda=use_cuda, record_dir=record_dir, record_format=record_format, stop_event=None, log_prefix="Main")
    except KeyboardInterrupt:
        print("\nStreaming stopped by user.")
    finally:
        socket.close()
        context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input SMPL data (.npz or .pkl)")
    parser.add_argument("--output", type=str, default="sonic_stream_payload.npz", help="Output .npz for inspection")
    parser.add_argument("--fps", type=float, default=50.0, help="Source framerate")
    parser.add_argument("--post_process", type=bool, default=True, help="Use Kimodo's post processing")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    
    # NEW ARGUMENTS ADDED HERE
    parser.add_argument("--no_legs", action="store_true", help="Zero out leg joints to isolate upper body motion")
    parser.add_argument("--disable_ytoz", action="store_true", help="Disable Y-to-Z up conversion (useful for EDGE pkls if they are already Z-up)")
    
    args = parser.parse_args()

    # Prepare streaming payload
    processed_data = load_smpl_raw(
        input_path=args.input, 
        source_fps=args.fps, 
        post_process=args.post_process,
        no_legs=args.no_legs,
        disable_ytoz=args.disable_ytoz
    )

    run_pico(processed_data, host=args.host, port=args.port, target_fps=args.fps)
