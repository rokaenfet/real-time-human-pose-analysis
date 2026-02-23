from typing import List, Tuple, Optional, Dict
import tkinter as tk


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