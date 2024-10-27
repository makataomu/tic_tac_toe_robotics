import cv2
import numpy as np

# Function to perform perspective transformation
def apply_perspective_transform(image, pts, grid_width=600, grid_height=600):
    # Destination points (the corners of a perfect 3x3 grid)
    dst_pts = np.array([[0, 0], [grid_width, 0], [grid_width, grid_height], [0, grid_height]], dtype="float32")
    
    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    # Apply the perspective transformation to warp the image
    warped = cv2.warpPerspective(image, M, (grid_width, grid_height))

    return warped

# Function to detect the grid and apply perspective correction
def detect_grid_and_cells(image_path, image=None, resize_dim=(600, 600), blur_kernel=(5, 5), 
                          adaptive_thresh_block_size=11, adaptive_thresh_C=2, 
                          min_area=10000, show_image=True):
    if image is None:
        if image_path is None:
            raise ValueError("if image isn't provided, you should provide image_path")
        image = cv2.imread(image_path)
    image = cv2.resize(image, resize_dim)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, adaptive_thresh_block_size, adaptive_thresh_C)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:  # Assuming the grid is large enough
            if area > max_area:
                max_area = area
                best_cnt = c

    # If grid found, find its four corner points
    if best_cnt is not None:
        epsilon = 0.02 * cv2.arcLength(best_cnt, True)
        approx = cv2.approxPolyDP(best_cnt, epsilon, True)

        if len(approx) == 4:
            # есть границы поля для игры
            # Sort the corner points to be top-left, top-right, bottom-right, bottom-left
            pts = np.array([point[0] for point in approx], dtype="float32")
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]  # Top-left
            rect[2] = pts[np.argmax(s)]  # Bottom-right
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]  # Top-right
            rect[3] = pts[np.argmax(diff)]  # Bottom-left

            # Apply perspective transformation to correct the grid
            image = apply_perspective_transform(image, rect, resize_dim[0], resize_dim[1])

            # Now you can divide the warped grid into 9 cells
            cells = divide_into_cells((0, 0, resize_dim[0], resize_dim[1]), image)
        else:
            # если нету границы поля для игры
            x, y, w, h = cv2.boundingRect(best_cnt)
            cv2.drawContours(image, [best_cnt], -1, (0, 255, 0), 3)
            cells = divide_into_cells((x, y, w, h), image)

            print("Could not find four corner points of the grid.")

        # Optionally show the final image with cells
        if show_image:
            cv2.imshow("Final Image with Cells", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return cells

    else:
        print("No grid detected")

# Function to divide the grid into 9 cells (top-left and bottom-right corners)
def divide_into_cells(grid_bounding_box, image, color_cells=True, grid_size=3, transparency=0.4):
    x, y, w, h = grid_bounding_box
    cell_width = w // grid_size
    cell_height = h // grid_size

    # Define colors for each cell (only used if color_cells is True)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (128, 128, 0), (0, 128, 128)
    ]

    overlay = image.copy()  # Create an overlay for transparency
    alpha = transparency  # Transparency factor

    cells = []
    color_index = 0

    for i in range(grid_size):
        for j in range(grid_size):
            x0 = x + j * cell_width
            y0 = y + i * cell_height
            x1 = x0 + cell_width
            y1 = y0 + cell_height

            cell = {
                'x0': x0, 'y0': y0,  # Top-left corner
                'x1': x1, 'y1': y1   # Bottom-right corner
            }
            cells.append(cell)

            if color_cells:  # Only color the cells if the flag is True
                cell_color = colors[color_index % len(colors)]
                color_index += 1

                # Draw the cell with transparency
                cv2.rectangle(overlay, (cell['x0'], cell['y0']),
                              (cell['x1'], cell['y1']),
                              cell_color, -1)
                cv2.rectangle(overlay, (cell['x0'], cell['y0']),
                              (cell['x1'], cell['y1']),
                              (255, 255, 255), 2)

    # If coloring is enabled, blend the overlay with the original image
    if color_cells:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    return cells
