import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_vanishing_point_refined(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #  Pre-processing: Blur more to ignore small tree branches
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) # Here blurring done by gaussian blurring
    edges = cv2.Canny(blurred, 50, 150)
    
    #  Line Detection: Increase minLineLength to ignore "noisy" tree bits
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                            minLineLength=120, maxLineGap=15)
    
    intersections = []
    filtered_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle. If the line is too horizontal (near 0°) 
            # or perfectly vertical, ignore it.
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if 10 < angle < 80 or 100 < angle < 170:
                filtered_lines.append(line[0])

        # Find Intersections of filtered lines
        for i in range(len(filtered_lines)):
            for j in range(i + 1, len(filtered_lines)):
                l1, l2 = filtered_lines[i], filtered_lines[j]
                
                a1, b1 = l1[3] - l1[1], l1[0] - l1[2]
                c1 = a1 * l1[0] + b1 * l1[1]
                a2, b2 = l2[3] - l2[1], l2[0] - l2[2]
                c2 = a2 * l2[0] + b2 * l2[1]
                
                det = a1 * b2 - a2 * b1
                if abs(det) > 100:
                    x = (b2 * c1 - b1 * c2) / det
                    y = (a1 * c2 - a2 * c1) / det
                    # Only keep intersections in the upper-middle of the image AS INITIALLY IT WAS GETTING CONFUSED WITH THE LINEAR TREE BRANCHES.
                    if 0 < x < img.shape[1] and 0 < y < img.shape[0] * 0.7:
                        intersections.append((x, y))

    #  Result Visualization
    if intersections:
        intersections = np.array(intersections)
        vx = int(np.median(intersections[:, 0]))
        vy = int(np.median(intersections[:, 1]))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        
        # Only draw lines that actually point toward our estimated VP
        for line in filtered_lines[:15]: 
            plt.plot([line[0], line[2]], [line[1], line[3]], color='lime', linewidth=3)
            
        plt.scatter([vx], [vy], color='red', s=2000, zorder=5) 
        plt.title("Refined Vanishing Point Estimation")
        plt.savefig('refined_output.png') # Saves the clean version
        plt.show()

if __name__ == "__main__":
    process_vanishing_point_refined("data/pexels-photo-10622719.jpeg")
