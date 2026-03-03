import os
import glob
import cv2
from typing import List, Dict, Tuple, Optional, Generator
from pathlib import Path

class UPNALoader:
    """
    A robust loader for the UPNA Head Pose Database.
    
    Directory Structure Expected:
    ./Head_Pose_Database_UPNA/
        ├── User_01/
        │   ├── user_01_video_01.mp4
        │   └── ...
        ├── User_02/
        └── ...
    """
    
    def __init__(self, base_dir: str = "./Head_Pose_Database_UPNA"):
        self.base_dir = Path(base_dir)
        
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {self.base_dir.absolute()}")
        
        # Cache available users and videos upon initialization
        self.available_users = self._scan_users()
        self.video_map = self._scan_videos()

    def _scan_users(self) -> List[str]:
        """Identify all valid User_[id] folders."""
        users = []
        if not self.base_dir.exists():
            return users
            
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("User_"):
                users.append(item.name.replace("User_", ""))
        
        return sorted(users)

    def _scan_videos(self) -> Dict[str, List[str]]:
        """
        Map user_ids to a list of absolute video paths.
        Returns: {'01': ['/path/to/user_01_video_01.mp4', ...], ...}
        """
        video_map = {}
        for user_id in self.available_users:
            user_folder = self.base_dir / f"User_{user_id}"
            # Pattern: user_[user_id]_video_[video_id].mp4
            pattern = str(user_folder / f"user_{user_id}_video_*.mp4")
            videos = sorted(glob.glob(pattern))
            if videos:
                video_map[user_id] = videos
        return video_map

    def get_video_path(self, user_id: str, video_id: str) -> Optional[str]:
        """
        Get the absolute path for a specific video.
        
        Args:
            user_id: 2-digit string (e.g., '01')
            video_id: 2-digit string (e.g., '05')
            
        Returns:
            Absolute path to the .mp4 file or None if not found.
        """
        user_id = user_id.zfill(2)
        video_id = video_id.zfill(2)
        
        if user_id not in self.video_map:
            return None
            
        target_filename = f"user_{user_id}_video_{video_id}.mp4"
        
        for path in self.video_map[user_id]:
            if path.endswith(target_filename):
                return path
        return None

    def load_video_cv2(self, user_id: str, video_id: str) -> Optional[cv2.VideoCapture]:
        """
        Directly returns a cv2.VideoCapture object for immediate processing.
        
        Args:
            user_id: 2-digit string
            video_id: 2-digit string
            
        Returns:
            cv2.VideoCapture object or None if file missing.
        """
        path = self.get_video_path(user_id, video_id)
        if not path:
            print(f"Error: Video user_{user_id}_video_{video_id}.mp4 not found.")
            return None
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video at {path}")
            return None
        return cap

    def iterate_videos(self, user_ids: Optional[List[str]] = None, video_ids: Optional[List[str]] = None) -> Generator[Tuple[str, str, cv2.VideoCapture], None, None]:
        """
        Generator to yield (user_id, video_id, capture_object) for batch processing.
        Useful for testing models across multiple samples without loading everything into memory.
        
        Args:
            user_ids: List of user IDs to filter (e.g., ['01', '02']). If None, loads all.
            video_ids: List of video IDs to filter (e.g., ['01', '03']). If None, loads all available for the user.
            
        Yields:
            Tuple: (user_id_str, video_id_str, cv2.VideoCapture object)
        """
        target_users = user_ids if user_ids else self.available_users
        
        for uid in target_users:
            uid = uid.zfill(2)
            if uid not in self.video_map:
                continue
                
            # Determine which videos to load for this user
            if video_ids:
                # Filter based on provided video_ids
                vids_to_load = [vid.zfill(2) for vid in video_ids]
            else:
                # Extract video_ids from filenames dynamically
                vids_to_load = []
                for path in self.video_map[uid]:
                    fname = os.path.basename(path)
                    # Expecting format user_XX_video_YY.mp4
                    parts = fname.replace('.mp4', '').split('_')
                    if len(parts) >= 4:
                        vids_to_load.append(parts[3])
                vids_to_load = sorted(list(set(vids_to_load)))

            for vid in vids_to_load:
                cap = self.load_video_cv2(uid, vid)
                if cap:
                    yield uid, vid, cap

# Example Usage Helper (Optional)
def demo_usage():
    loader = UPNALoader()
    
    # 1. Load a single specific video
    print("Loading single video...")
    cap = loader.load_video_cv2("01", "01")
    if cap:
        ret, frame = cap.read()
        if ret:
            print(f"Successfully read frame shape: {frame.shape}")
        cap.release()

    # 2. Iterate over multiple users/videos for model testing
    print("\nBatch processing example...")
    count = 0
    for uid, vid, cap in loader.iterate_videos(user_ids=["01", "02"], video_ids=["01", "02"]):
        # Insert your pose detection logic here
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     if not ret: break
        #     model.predict(frame)
        count += 1
        cap.release()
        print(f"Processed User {uid}, Video {vid}")
        
    print(f"Total videos processed: {count}")

if __name__ == "__main__":
    demo_usage()