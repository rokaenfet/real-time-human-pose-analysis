from typing import List, Tuple, Optional, Dict
import tkinter as tk
import cv2
import numpy as np

class ScreenUtility():
    """
    Returns screen size
    """
    def __init__(self, screen_margin : int = 100):
        self.root = tk.Tk()
        self.root.withdraw()
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.available_w = self.screen_w - screen_margin
        self.available_h = self.screen_h - screen_margin

    def get_available_screen_size(self) -> List[int]:
        """Return device's maximum screen res in pixel, minus the screen margin

        Returns
        -------
        List[int,int]
            available screen width, available screen height
        """
        return self.available_w, self.available_h
    
    def resize_with_padding(self, img, target_size):
        """Resizes keeping aspect ratio and pads with black."""
        h, w = img.shape[:2]
        tw, th = target_size
        scale = min(tw/w, th/h)
        nw, nh = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        
        # Center the image on the black canvas
        dx, dy = (tw - nw) // 2, (th - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized
        return canvas