# SIFT-feature-detection-and-matching# SIFT Feature Matching Application

## Overview

- This application provides a graphical user interface (GUI) for Scale-Invariant Feature Transform (SIFT) feature matching between images. SIFT is a powerful algorithm for detecting and describing local features in images that can be used to find correspondences between different views of an object or scene.

## Features

- Load and compare two images
- Create rotated versions of images for testing
- Adjust matching parameters in real-time
- Visualize keypoints and matches between images
- Fine-tune FLANN matcher parameters

## Requirements

```text
Python 3.x
OpenCV (cv2)
NumPy
Matplotlib
tkinter
PIL (Pillow)
```

## Installation

1. Clone the repository:

```text
git clone https://github.com/yourusername/sift-feature-matching.git
cd sift-feature-matching
```

2. Create a virtual environment (recommended):

```text
python -m venv venv
```

3. Activate the virtual environment:

- Windows:

```text
venv\Scripts\activate
```

- macOS/Linux:

```text
source venv/bin/activate
```

4. Install dependencies:

```text
pip install opencv-python numpy matplotlib pillow
```

## Usage

Run the application:

```text
python gnr602_final.py
```

The GUI provides the following options:

## Image Loading:

1. Load Image 1: Select the first image
2. Load Image 2: Select the second image
3. Use Rotated Version: Create a rotated version of Image 1
4. Parameters:

- Lowe's Ratio (0.1-1.0): Controls match filtering strictness (lower values = stricter filtering)
- Matches to show (1-100): Number of top matches to display
- FLANN Trees (1-20): Number of trees for the FLANN algorithm
- FLANN Checks (1-100): Number of checks for the FLANN algorithm
- Line Thickness (1-10): Thickness of the lines connecting matched features
- Click "Run SIFT Matching" to process the images and visualize the matches.

## How It Works

The application uses a custom SIFT implementation to detect keypoints and compute descriptors for both images.
FLANN (Fast Library for Approximate Nearest Neighbors) is used to match descriptors between images.
Lowe's ratio test is applied to filter good matches.
The top matches are visualized, connecting corresponding features between the two images.

## Implementation Details

main.py: Contains the core SIFT implementation with keypoint detection, orientation assignment, and descriptor generation
gnr602_final.py: Implements the GUI and visualization components

## Example

After loading two images (or creating a rotated version), adjust the parameters as needed and click "Run SIFT Matching." The application will display:

The original images in the main window
A separate window showing the matched features with connecting lines
License
This project is available under the MIT License.
