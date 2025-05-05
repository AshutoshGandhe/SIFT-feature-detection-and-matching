import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import main

class SIFTGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SIFT Feature Matching GUI")
        
        # Initialize variables
        self.img1 = None
        self.img2 = None
        self.kp1 = None
        self.kp2 = None
        self.des1 = None
        self.des2 = None
        self.good_matches = None
        
        # Parameters with default values
        self.ratio_threshold = 0.75
        self.num_matches_to_show = 15
        self.flann_trees = 5
        self.flann_checks = 50
        self.match_line_thickness = 2  # New parameter for line thickness
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Frame for controls
        control_frame = Frame(self.root)
        control_frame.pack(side=TOP, fill=X, padx=5, pady=5)
        
        # Image loading buttons
        load_frame = LabelFrame(control_frame, text="Image Loading")
        load_frame.pack(side=LEFT, padx=5, pady=5)
        
        Button(load_frame, text="Load Image 1", command=self.load_image1).pack(padx=5, pady=2)
        Button(load_frame, text="Load Image 2", command=self.load_image2).pack(padx=5, pady=2)
        Button(load_frame, text="Use Rotated Version", command=self.use_rotated_version).pack(padx=5, pady=2)
        
        # Parameters frame
        param_frame = LabelFrame(control_frame, text="Parameters")
        param_frame.pack(side=LEFT, padx=5, pady=5)
        
        # Ratio threshold
        Label(param_frame, text="Lowe's Ratio:").grid(row=0, column=0, sticky=W)
        self.ratio_slider = Scale(param_frame, from_=0.1, to=1.0, resolution=0.05, 
                                 orient=HORIZONTAL, command=self.update_params)
        self.ratio_slider.set(self.ratio_threshold)
        self.ratio_slider.grid(row=0, column=1, padx=5)
        
        # Number of matches to show
        Label(param_frame, text="Matches to show:").grid(row=1, column=0, sticky=W)
        self.matches_spin = Spinbox(param_frame, from_=1, to=100, command=self.update_params)
        self.matches_spin.delete(0, "end")
        self.matches_spin.insert(0, self.num_matches_to_show)
        self.matches_spin.grid(row=1, column=1, padx=5)
        
        # FLANN parameters
        Label(param_frame, text="FLANN Trees:").grid(row=2, column=0, sticky=W)
        self.trees_spin = Spinbox(param_frame, from_=1, to=20, command=self.update_params)
        self.trees_spin.delete(0, "end")
        self.trees_spin.insert(0, self.flann_trees)
        self.trees_spin.grid(row=2, column=1, padx=5)
        
        Label(param_frame, text="FLANN Checks:").grid(row=3, column=0, sticky=W)
        self.checks_spin = Spinbox(param_frame, from_=1, to=100, command=self.update_params)
        self.checks_spin.delete(0, "end")
        self.checks_spin.insert(0, self.flann_checks)
        self.checks_spin.grid(row=3, column=1, padx=5)
        
        # Line thickness parameter
        Label(param_frame, text="Line Thickness:").grid(row=4, column=0, sticky=W)
        self.thickness_slider = Scale(param_frame, from_=1, to=10, resolution=1, 
                                    orient=HORIZONTAL, command=self.update_params)
        self.thickness_slider.set(self.match_line_thickness)
        self.thickness_slider.grid(row=4, column=1, padx=5)
        
        # Process button
        Button(control_frame, text="Run SIFT Matching", command=self.run_sift_matching, 
               bg="#4CAF50", fg="white").pack(side=LEFT, padx=5, pady=5)
        
        # Matplotlib figure for display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        
        # Status bar
        self.status = Label(self.root, text="Ready", bd=1, relief=SUNKEN, anchor=W)
        self.status.pack(side=BOTTOM, fill=X)
    
    def load_image1(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.img1 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            self.update_status(f"Loaded Image 1: {filepath}")
            self.show_images()
    
    def load_image2(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.img2 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            self.update_status(f"Loaded Image 2: {filepath}")
            self.show_images()
    
    def use_rotated_version(self):
        if self.img1 is not None:
            rows, cols = self.img1.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle=45, scale=1.2)
            self.img2 = cv2.warpAffine(self.img1, M, (cols, rows))
            self.update_status("Created rotated version of Image 1 as Image 2")
            self.show_images()
        else:
            self.update_status("Error: Load Image 1 first")
    
    def update_params(self, *args):
        try:
            self.ratio_threshold = float(self.ratio_slider.get())
            self.num_matches_to_show = int(self.matches_spin.get())
            self.flann_trees = int(self.trees_spin.get())
            self.flann_checks = int(self.checks_spin.get())
            self.match_line_thickness = int(self.thickness_slider.get())
        except:
            pass
    
    def run_sift_matching(self):
        if self.img1 is None or self.img2 is None:
            self.update_status("Error: Please load both images first")
            return
            
        try:
            # Initialize SIFT detector
            self.kp1, self.des1 = main.computeKeypointsAndDescriptors(self.img1)
            self.kp2, self.des2 = main.computeKeypointsAndDescriptors(self.img2)
            
            # Match descriptors using FLANN
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=self.flann_trees)
            search_params = dict(checks=self.flann_checks)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(self.des1, self.des2, k=2)
            
            # Apply Lowe's ratio test
            self.good_matches = [m for m, n in matches if m.distance < self.ratio_threshold * n.distance]
            
            self.update_status(f"Found {len(self.good_matches)} good matches")
            self.show_matches()
            
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
    
    def show_images(self):
        if self.img1 is not None and self.img2 is not None:
            self.ax1.clear()
            self.ax2.clear()
            
            self.ax1.imshow(self.img1, cmap='gray')
            self.ax1.set_title("Image 1")
            self.ax1.axis('off')
            
            self.ax2.imshow(self.img2, cmap='gray')
            self.ax2.set_title("Image 2")
            self.ax2.axis('off')
            
            self.fig.tight_layout()
            self.canvas.draw()
    
    def show_matches(self):
        if (self.img1 is not None and self.img2 is not None and 
            self.kp1 is not None and self.kp2 is not None and 
            self.good_matches is not None):
            
            # Draw matches with thicker lines
            matched_img = cv2.drawMatches(
                self.img1, self.kp1, 
                self.img2, self.kp2, 
                self.good_matches[:self.num_matches_to_show], 
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 255, 0),  # Green color for matches
                singlePointColor=None,
                matchesMask=None,
                matchesThickness=self.match_line_thickness  # Set line thickness here
            )
            
            self.ax1.clear()
            self.ax2.clear()
            
            # Show original images
            self.ax1.imshow(self.img1, cmap='gray')
            self.ax1.set_title("Image 1")
            self.ax1.axis('off')
            
            self.ax2.imshow(self.img2, cmap='gray')
            self.ax2.set_title("Image 2")
            self.ax2.axis('off')
            
            # Create a new figure for matches
            match_fig, match_ax = plt.subplots(figsize=(15, 8))
            match_ax.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
            match_ax.set_title(f"Top {min(self.num_matches_to_show, len(self.good_matches))} SIFT Matches")
            match_ax.axis('off')
            match_fig.tight_layout()
            
            # Show the match figure in a new window
            match_fig.show()
    
    def update_status(self, message):
        self.status.config(text=message)
        self.root.update_idletasks()

# Create and run the GUI
if __name__ == "__main__":
    root = Tk()
    app = SIFTGUI(root)
    root.mainloop()