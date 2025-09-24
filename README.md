# OpenCV Horizon Leveling 

Using OpenCV to stabilize climbing footage by detecting and correcting rotational drift using the intersection between the climbing wall and the mat.

<p align="center">
  <img src="demo.gif" alt="Demo"/>
</p>


## Inspiration
While reviewing climbing footage, I noticed that the video was not properly leveled and had a gradual rotation throughtout the entire clip. Unlike an rotated image/video, a simple rotational offset could not be used to straighten the video as the angle of rotation varied continuously throughtout the video.

Standard video editor stabilization tools like CapCut were not able to fixed this type of gradual drift. I suspect this is because most stabilization algorithms are used to reduce fast, erratic movements rather than a slow, continuous rotation. 

Since climbing gyms have a distinct visual feature of a horizontal line between the climbing wall and the mat. I decided to use this as a reference point for stabilizing my climbing footage using computer vision.


## Approach
Using OpenCV, I implemented a stabilization pipeline that detects the wall-mat intersection line and rotates each frame to keep this reference line horizontal.

### Algorithm
1. **Image Masking**: Applies a mask to enhance colour contrast between the climbing wall and floor mats, making edge detection more reliable.

2. **Edge Detection**: Uses OpenCV's Canny edge detector followed by Hough Line Transform to identify potential lines in the image.

3. **Line Filtering**: Filters detected lines, retaining only lines that are approximately horizontal for further processing.

4. **Outlier Removal**: RANSAC algorithm to remove outlier points from the detected line segments.

5. **Line Fitting**: Uses OpenCV's fitline function to fit a single line through the RANSAC filtered points.

6. **Rotation Calculation**: Compute the rotation angle required for the frame.

7. **Frame Correction**: Apply calculated rotation to each frame to maintain a horizontal reference line.


## Results
The algorithm successfully corrected the rotation drift throughtout the climbing footage. The wall-mat reference line originally had a slightly negative slope at the beginning, but a significantly positive slope towards the end of the clip. As far as this clip is concerned, the video was successfully straighten. As for future improvements, more videos are required to test edge cases and various scenarios where the wall-mat line is not clearly visible. 


