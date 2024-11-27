from typing import Dict, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import time
from gholb import tictactoe as ttt 

X = "x"
O = "o"
EMPTY = None

CELLS_COORDS = [{'x0': 207, 'y0': 121, 'x1': 298, 'y1': 215},
                {'x0': 298, 'y0': 122, 'x1': 392, 'y1': 216},
                {'x0': 393, 'y0': 123, 'x1': 488, 'y1': 217},
                {'x0': 205, 'y0': 213, 'x1': 296, 'y1': 309},
                {'x0': 298, 'y0': 215, 'x1': 391, 'y1': 311},
                {'x0': 392, 'y0': 216, 'x1': 488, 'y1': 311},
                {'x0': 204, 'y0': 307, 'x1': 295, 'y1': 405},
                {'x0': 296, 'y0': 309, 'x1': 390, 'y1': 407},
                {'x0': 391, 'y0': 311, 'x1': 487, 'y1': 409}]

def load_yolo_model(model_path):
    return YOLO(model_path)

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


def detect_xo(frame, model):
    """
    детекция крестиков и ноликов для картинки/фрейма 
    """
    results = model.predict(source=frame, conf=0.5, show=False, verbose=False)  # Adjust `conf` if necessary
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            label = result.names[int(box.cls)]
            confidence = float(box.conf)

            # Determine the cell that contains this object
            detected_objects.append({
                'label': label,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
    return detected_objects

def send_move_to_robot(robot_move):
    pass

# Helper function to find cell containing the center of the bounding box
def find_cell_for_object(center, cells):
    for index, cell in enumerate(cells):
        if cell['x0'] <= center[0] <= cell['x1'] and cell['y0'] <= center[1] <= cell['y1']:
            return index  # Return the cell index if the center is within cell bounds
    return None

# Function to detect X/O and identify corresponding cell
def detect_xo_and_identify_cells(frame, model, cells, show_yolo=False):
    results = model.predict(source=frame, conf=0.2, show=show_yolo, verbose=False)  # Adjust `conf` if necessary
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            label = result.names[int(box.cls)]
            confidence = float(box.conf)

            # Determine the cell that contains this object
            cell_index = find_cell_for_object((center_x, center_y), cells)
            
            if cell_index is not None:
                detected_objects.append({
                    'label': label,
                    'confidence': confidence,
                    'cell_index': cell_index,
                    'bbox': (x1, y1, x2, y2)
                })
    return detected_objects


# Modify get_original_image_cells to accept an image array instead of path
def get_original_image_cells(image, resize_dim=(600, 600), show_image=False):
    """
    Detect grid and cells in an image (either as a path or image array) and return cell coordinates.
    """
    # Detect the grid and get the perspective transformation matrix
    cells = detect_grid_and_cells(image_path=None, image=image, resize_dim=resize_dim, show_image=show_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area, best_cnt = 0, None
    for c in contours:
        area = cv2.contourArea(c)
        if area > 10000 and area > max_area:
            max_area = area
            best_cnt = c

    # Find four corners of the grid for transformation
    try: 
        epsilon = 0.02 * cv2.arcLength(best_cnt, True)
        approx = cv2.approxPolyDP(best_cnt, epsilon, True)
        pts = np.array([point[0] for point in approx], dtype="float32")
        rect = np.zeros((4, 2), dtype="float32")
        s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        
        # Compute perspective transform and inverse transform matrices
        dst_pts = np.array([[0, 0], [resize_dim[0], 0], [resize_dim[0], resize_dim[1]], [0, resize_dim[1]]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst_pts)
        M_inv = cv2.getPerspectiveTransform(dst_pts, rect)

        # Apply perspective transform to get cells in transformed space
        warped_image = apply_perspective_transform(image, rect, resize_dim[0], resize_dim[1])
        transformed_cells = divide_into_cells((0, 0, resize_dim[0], resize_dim[1]), warped_image)

        # Transform cells back to original coordinates using M_inv
        original_cells = []
        for cell in transformed_cells:
            top_left = np.dot(M_inv, np.array([cell['x0'], cell['y0'], 1]))
            bottom_right = np.dot(M_inv, np.array([cell['x1'], cell['y1'], 1]))
            original_cells.append({
                'x0': int(top_left[0] / top_left[2]),
                'y0': int(top_left[1] / top_left[2]),
                'x1': int(bottom_right[0] / bottom_right[2]),
                'y1': int(bottom_right[1] / bottom_right[2])
            })

        return original_cells
    
    except:
        raise ValueError("grid not found")

def convert_ai_move_for_robot(cell_index, cells: Dict): 
    x_center = (cells[cell_index]['x0'] + cells[cell_index]['x1']) / 2
    y_center = (cells[cell_index]['y0'] + cells[cell_index]['y1']) / 2
    return x_center, y_center

def draw_cells(frame, cells, color=(0, 255, 0), thickness=2, show_index=True):
    """
    Draws rectangles on the frame for each cell in cells and optionally labels them with their index.
    """
    for idx, cell in enumerate(cells):
        top_left = (cell['x0'], cell['y0'])
        bottom_right = (cell['x1'], cell['y1'])
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        
        if show_index:
            # Calculate the position for the index label (top-left corner with some padding)
            text_position = (cell['x0'] + 5, cell['y0'] + 25)
            # Define the font, scale, color, and thickness
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            text_color = (0, 0, 255)  # Red color for visibility
            text_thickness = 2
            # Put the index text on the frame
            cv2.putText(frame, str(idx), text_position, font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    
    return frame

# Track the position stability for detected objects
def track_positions(detected_objects, frame_time, stable_positions, duration=2):
    confirmed_moves = []
    for obj in detected_objects:
        cell_index = obj['cell_index']
        label = obj['label']

        if (label, cell_index) in stable_positions:
            last_seen_time = stable_positions[(label, cell_index)]
            if frame_time - last_seen_time >= duration:
                confirmed_moves.append(obj)
        else:
            stable_positions[(label, cell_index)] = frame_time
    
    # Remove entries not seen in this frame
    stable_positions = {key: stable_positions[key] for key in stable_positions if key in {(obj['label'], obj['cell_index']) for obj in detected_objects}}
    return confirmed_moves, stable_positions

def initial_state():
    """ Returns starting state of the board. """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]
            ]

def update_board(board, confirmed_moves):
    for move in confirmed_moves:
        row, col = move['cell_index'] // 3, move['cell_index'] % 3
        board[row][col] = X if move['label'] == X else O
    return board 

# Process live video feed and detect objects
def process_live_video(video_path, model_path, show_image=False, user=X, hardcode_cells=True, move_duration=2, show_yolo=False):

    ai_turn = True
    ai_sign = X
    if user == X:
        ai_turn = False
        ai_sign = O

    model = load_yolo_model(model_path)

    # Open live video feed (0 is usually the default webcam)
    video = cv2.VideoCapture(video_path)
    
    # Get grid cells from the first frame for reference
    ret, first_frame = video.read()
    if not ret:
        print("Error reading video.")
        return
    
    # NOTE those are bboxes for now
    if hardcode_cells:
        original_cells = CELLS_COORDS
    else:
        original_cells = get_original_image_cells(first_frame, show_image=show_image) # order is correct

    if show_image:
        first_frame_with_cells = draw_cells(first_frame.copy(), original_cells, color=(0, 255, 0), thickness=2)
        cv2.imshow('Original Cells Verification', first_frame_with_cells)
        print("Press any key to continue...")
        cv2.waitKey(0)

    # Initialize stable positions with the first frame detections
    initial_detections = detect_xo_and_identify_cells(first_frame, model, original_cells, show_yolo=show_yolo)
    frame_time = time.time()
    current_positions = {(obj['label'], obj['cell_index']): frame_time for obj in initial_detections}
    board = initial_state()

    # Loop over frames
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_time = time.time()  # Current frame time
        
        detected_objects = detect_xo_and_identify_cells(frame, model, original_cells, show_yolo=show_yolo)
        confirmed_moves, current_positions = track_positions(
            detected_objects, 
            frame_time, 
            stable_positions=current_positions, 
            duration=move_duration
        )

        print(confirmed_moves, current_positions)
        
        if confirmed_moves:
            board = update_board(board, confirmed_moves)
            # Display the updated board and moves if needed
            print("Board updated:", board)
            
            # Here we call Gholibs function to get next move. This will take some time. 
            game_over = ttt.terminal(board)
            player = ttt.player(board)

            print(f'next move {player}')

            if game_over:
                winner = ttt.winner(board)
                if winner is None:
                    title = f"Game Over: Tie."
                else:
                    title = f"Game Over: {winner} wins."
                print(title)
                break
            elif user == player:
                title = f"Play as {user}"
            else:
                title = f"Computer thinking..."
            print(title)

            if user != player and not game_over:
                print('robot enter')
                if ai_turn:
                    print("robot maces moves")
                    time.sleep(0.5)
                    move = ttt.minimax(board) # это (i, j) - индексы ячейки
                    board = ttt.result(board, move)

                    # Send cell center to robot
                    cell_index = move[0] * 3 + move[1]
                    robot_move = convert_ai_move_for_robot(cell_index, original_cells)
                    send_move_to_robot(robot_move)

                    frame_time = time.time()
                    current_positions[ai_sign, cell_index] = frame_time

                    ai_turn = False
                else:
                    print("robot doesnt mace moves")

                    ai_turn = True
            
        if show_image:
            cv2.imshow('Real-Time Tic Tac Toe', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Press 'q' to exit the loop

    video.release()
    cv2.destroyAllWindows()
