import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_ntu_frame(file_path, frame_idx=0):
    # Load the data
    data = np.load(file_path, allow_pickle=True)
    
    # Handle dictionary vs raw array (depending on your converter)
    if isinstance(data, np.ndarray) and data.dtype == 'O':
        skeleton = data.item()['skel_body0']
    else:
        skeleton = data

    # NTU Joint Connections (Bones)
    # These pairs connect the 25 joints into a human skeleton
    bones = [
        (1, 2), (2, 21), (21, 3), (3, 4),           # Spine & Head
        (21, 5), (5, 6), (6, 7), (7, 8),           # Left Arm
        (21, 9), (9, 10), (10, 11), (11, 12),      # Right Arm
        (1, 13), (13, 14), (14, 15), (15, 16),     # Left Leg
        (1, 17), (17, 18), (18, 19), (19, 20),     # Right Leg
        (22, 23), (22, 8), (24, 25), (24, 12)      # Hands (Standard V2)
    ]
    # Note: NTU joints are 1-indexed in docs, 0-indexed in arrays
    bones = [(i - 1, j - 1) for i, j in bones]

    # Get frame data (Shape: [Joints, Coordinates])
    joints = skeleton[frame_idx]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Plot the joints
    ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c='red', s=20)

    # 2. Plot the bones
    for b in bones:
        if b[0] < len(joints) and b[1] < len(joints):
            ax.plot([joints[b[0], 0], joints[b[1], 0]],
                    [joints[b[0], 2], joints[b[1], 2]],
                    [joints[b[0], 1], joints[b[1], 1]], c='blue')

    # Setting labels and fixing perspective
    ax.set_xlabel('X')
    ax.set_ylabel('Z (Depth)')
    ax.set_zlabel('Y (Height)')
    ax.view_init(elev=10, azim=100) # Adjust view for best perspective
    plt.title(f"NTU Skeleton Frame {frame_idx}")
    plt.show()

# Run it
visualize_ntu_frame('./NPU_raw_npy/S001C001P001R001A001.skeleton.npy', frame_idx=10)