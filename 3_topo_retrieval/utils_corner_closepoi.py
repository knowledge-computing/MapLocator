import cv2
import numpy as np
import os 
import glob
import json



def read_map_content_area_from_json(legend_json_path, use_bbox = True):#
    map_area_bbox = None
    
    try:   
        with open(legend_json_path, 'r', encoding='utf-8') as file:
            legend_dict = json.load(file)
    except:
        return FileNotFoundError(f'{legend_json_path} does not exist')

    for item in legend_dict['segments']:
        if 'map' == item['class_label']:

            if use_bbox:
                map_area_bbox = item['bbox'] 
            else:
                map_area_bbox = item['poly_bounds']
        
    return map_area_bbox

def read_map_area(args, use_bbox = True):
    if args.segmentation_json_path is  None or not os.path.isfile(args.segmentation_json_path):
        return -1 # segmentation file does not exist 
    
    segmentation_json_path = args.segmentation_json_path 
    
    map_area = read_map_content_area_from_json(segmentation_json_path, use_bbox = use_bbox)

    if map_area is None:
        return -2 # map_area does not exist
    else:
        
        return map_area 




def crop_patches_around_points(args, image, map_area_bbox, patch_size = (500, 500), if_write_file = True , bbox_format='xyxy'):

    assert bbox_format == 'xywh' or bbox_format == 'xyxy'
    
    if if_write_file:
        output_folder = os.path.join(args.output_dir, 'georef')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    if bbox_format == 'xywh':
        # x: height, y: width
        map_content_y, map_content_x,  map_content_w, map_content_h = map_area_bbox[0]

        bounding_box = [(map_content_y, map_content_x ),
            (map_content_y, map_content_x + map_content_h ),
            (map_content_y + map_content_w, map_content_x + map_content_h ),
            (map_content_y + map_content_w, map_content_x )]
    elif bbox_format == 'xyxy':
        xmin, ymin, xmax, ymax = map_area_bbox

        # Four corner points (x, y)
        bounding_box = [
            (xmin, ymin),  # top-left
            (xmax, ymin),  # top-right
            (xmax, ymax),  # bottom-right
            (xmin, ymax)   # bottom-left
        ]
    else:
        raise NotImplementedError 

    image_h, image_w = image.shape[:2]

    patch_list = []
    coords_list = []
    # Iterate through each point in the bounding box
    for idx in range(len(bounding_box)):
        point = bounding_box[idx]
        # Convert the point coordinates to integers
        y, x = map(int, point)
        
        # Calculate the top-left corner of the patch
        top_left_y = max(0, y - patch_size[0] // 2)
        top_left_x = max(0, x - patch_size[1] // 2)
        
        # Calculate the bottom-right corner of the patch
        bottom_right_y = min(image_w, y + patch_size[0] // 2)
        bottom_right_x = min(image_h, x + patch_size[1] // 2)
        
        # Crop the patch from the image
        patch = image[top_left_x:bottom_right_x, top_left_y:bottom_right_y]
        
        
        if if_write_file:
            
            output_path = os.path.join(output_folder, f'{args.map_name}_{str(idx)}.jpg' )

            cv2.imwrite(output_path, patch)
            

        else:
            # Add the patch to the list
            patch_list.append(patch)
            coords_list.append({'top_left_x':top_left_x, 'top_left_y': top_left_y, 'bottom_right_x':bottom_right_x, 'bottom_right_y':bottom_right_y })


    if if_write_file:
        logger.info(f'Saved the cropped images for {args.map_name} in {output_folder}')
        
        return 
    else:
        return patch_list, coords_list



# Function to find intersection of two lines
def line_intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    return int(np.round(x0)), int(np.round(y0))


def keep_two_largest_components(binary_mask):
    # Find all connected components (white blobs in the binary mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    # If there are fewer than 3 components (background + 2), return the mask as is
    if num_labels <= 2:
        return binary_mask

    # Sort components by area (excluding the background, which is component 0)
    sorted_stats = sorted(range(1, num_labels), key=lambda x: stats[x, cv2.CC_STAT_AREA], reverse=True)

    # Create an output mask
    output_mask = np.zeros_like(binary_mask)

    # Keep the two largest components
    for i in sorted_stats[:2]:
        output_mask[labels == i] = 255

    return output_mask

def create_border_mask(height, width, border_width=10):


    # Create a binary mask of the same size as the input image
    mask = np.zeros((height, width), dtype=np.uint8)

    # Set the border to white
    mask[:border_width, :] = 255  # Top border
    mask[-border_width:, :] = 255  # Bottom border
    mask[:, :border_width] = 255  # Left border
    mask[:, -border_width:] = 255  # Right border

    return mask

def create_hough_mask(gray, thresh = 3): 
    # darker than 3 will be considered as pure black 
    # Create a mask where black pixels are marked
    # thresh = 3
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # only keep the two largest components, avoid masking out the middle regions 
    mask = keep_two_largest_components(mask)
    

    # dilate the mask
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel
    mask = cv2.dilate(mask, kernel, iterations=1)


    height, width = gray.shape
    # create a border mask to avoid finding edges around the border
    border_mask = create_border_mask(height, width, border_width=10)

    mask = cv2.bitwise_or(mask, border_mask)

    # cv2.imwrite(os.path.join(output_dir, os.path.basename(image_name).split('.')[0] + '_mask.jpg'), mask)

    mask = cv2.bitwise_not(mask)

    return mask

def threshold_dark_pixels_rgb(image, threshold_value=50):
   
    # Split the image into its B, G, and R channels
    b_channel, g_channel, r_channel = cv2.split(image)

    # Apply a binary threshold to each channel
    _, b_thresh = cv2.threshold(b_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)
    _, g_thresh = cv2.threshold(g_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)
    _, r_thresh = cv2.threshold(r_channel, threshold_value, 255, cv2.THRESH_BINARY_INV)

    # Combine the thresholds to get the final mask
    dark_pixels_mask = cv2.bitwise_and(b_thresh, cv2.bitwise_and(g_thresh, r_thresh))

    return dark_pixels_mask

def is_close(point1, point2, threshold):
    """Check if two points are close to each other within a given threshold."""
    return np.linalg.norm(np.array(point1) - np.array(point2)) < threshold


def is_near_line(point, line, threshold):
    """Check if a point is near a given line within a threshold distance."""
    # point should be in (col, row)
    x0, y0 = point
    x1, y1, x2, y2 = line
    distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    return distance < threshold

def process_image(image, point_of_interest = None, threshold_distance = 100, if_visualize = False ):
    """
    Point_of_interset in the format of (height, width)
    # Distance threshold is to only consider lines near the point
    """

    height, width = image.shape[:2]
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    # Edge detection
    edges = cv2.Canny(gray, 100, 300)

    # To deal with the black border artifact after image rotation 
    mask = create_hough_mask(gray) 
    
    # Apply the mask to the edges
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    


    output_dir = '.'
    cv2.imwrite(os.path.join(output_dir, os.path.basename('test').split('.')[0] + '_edge.jpg'), edges)

    try:
        assert len(lines) != 0, "No line detected"
    except:

        print("No line detected")
        # print(image_name, "No line detected")
        return -1

    # # Find the first vertical and horizontal lines
    vertical_lines = []
    horizontal_lines = []


    for line in lines:

        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        line_coords = (x1, y1, x2, y2)
        
        if point_of_interest:
            if np.abs(np.sin(theta)) < 0.2 and is_near_line(point_of_interest, line_coords, threshold_distance):
                # horizontal_lines.append((rho, theta))
                vertical_lines.append((rho, theta))
            
            elif np.abs(np.cos(theta)) < 0.2 and is_near_line(point_of_interest, line_coords, threshold_distance):
                # vertical_lines.append((rho, theta))
                horizontal_lines.append((rho, theta))
        else: # point of interest is none 
            if np.abs(np.sin(theta)) < 0.2 :
                # horizontal_lines.append((rho, theta))
                vertical_lines.append((rho, theta))
            
            elif np.abs(np.cos(theta)) < 0.2 :
                # vertical_lines.append((rho, theta))
                horizontal_lines.append((rho, theta))


    print(len(horizontal_lines),len(vertical_lines)) 
    try:
        assert len(vertical_lines)!=0 and len(horizontal_lines)!=0 , "Not enough lines detected"
    except:
        print("Not enough lines detected")
        return -1

    corners = []
    corners.append(line_intersection(vertical_lines[0], horizontal_lines[0]))


    if if_visualize:
        # Draw the lines
        for line in [vertical_lines[0], horizontal_lines[0]]:

            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Draw corners
        for corner in corners:
            x, y = corner
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

        cv2.circle(image, (point_of_interest[0], point_of_interest[1]), 10, (255, 0, 0), -1)
        
        # Save and display the result
        # cv2.imwrite(os.path.join(output_dir, os.path.basename(image_name).split('.')[0] + '_corner.jpg'), image)


    return corners[0]


def process_image_lsd(image, point_of_interest=None, threshold_distance=100, if_visualize=False):
    """
    Detect vertical and horizontal lines using LSD.
    Point_of_interest in (row, col) = (y, x)
    """

    if if_visualize:
        cv2.polylines(image, [np.array(polygon).astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 300)

    mask = create_hough_mask(gray)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # -----------------------------
    # LSD line detection
    # -----------------------------
    lsd = cv2.createLineSegmentDetector(0)
    lines, _, _, _ = lsd.detect(edges)

    if lines is None or len(lines) == 0:
        print("No line detected")
        return -1

    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        theta = np.arctan2((y2 - y1), (x2 - x1))

        # Vertical-ish vs Horizontal-ish
        if point_of_interest:
            if abs(np.cos(theta)) < 0.2 and is_near_line(point_of_interest, (x1, y1, x2, y2), threshold_distance):
                vertical_lines.append((x1, y1, x2, y2))
            elif abs(np.sin(theta)) < 0.2 and is_near_line(point_of_interest, (x1, y1, x2, y2), threshold_distance):
                horizontal_lines.append((x1, y1, x2, y2))
        else:
            if abs(np.cos(theta)) < 0.2:
                vertical_lines.append((x1, y1, x2, y2))
            elif abs(np.sin(theta)) < 0.2:
                horizontal_lines.append((x1, y1, x2, y2))

    if len(vertical_lines) == 0 or len(horizontal_lines) == 0:
        print("Not enough lines detected")
        return -1
        
    view_lines(image, vertical_lines, horizontal_lines)
    # -----------------------------
    # Use first vertical & horizontal lines for intersection
    # -----------------------------
    # Convert LSD line endpoints to rho/theta to reuse your intersection function
    def endpoints_to_rho_theta(line):
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        theta = np.arctan2(dy, dx)
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        return rho, theta

    rho_theta_vertical = endpoints_to_rho_theta(vertical_lines[0])
    rho_theta_horizontal = endpoints_to_rho_theta(horizontal_lines[0])

    corners = [line_intersection(rho_theta_vertical, rho_theta_horizontal)]

    if if_visualize:
        # Draw lines
        for line in [vertical_lines[0], horizontal_lines[0]]:
            x1, y1, x2, y2 = line
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw corners
        for corner in corners:
            x, y = corner
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1)

        if point_of_interest:
            cv2.circle(image, (point_of_interest[1], point_of_interest[0]), 10, (255, 0, 0), -1)

    return corners[0]


def remove_all_files_in_directory(directory_path):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(directory_path, '*'))
    
    for file in files:
        try:
            os.remove(file)
            print(f"Removed file: {file}")
        except Exception as e:
            print(f"Error removing file {file}: {e}")



# def main():
    # tif_dir = '/home/yaoyi/shared/critical-maas/12month/nickel_maps/raw_geomaps_cogs/'
    # previous_module_dir = '/home/yaoyi/li002666/critical_maas/corner_detection/input/georef-inputs-12-month-hackathon/'


    # map_name = 'ece586c606067e6545357439320910e5c8cd4e6d4a6d4b6f4a6d4f6d4b6c34ca'
    # mapkurator_coords_dir = os.path.join(previous_module_dir, map_name, 'coordinate_spotting', map_name, 'spotter',map_name)
    # segmentation_json_path = os.path.join(previous_module_dir, map_name, 'legend_segment', map_name + '_map_segmentation.json')
    # image_path = os.path.join(tif_dir, map_name + '.tif')

    # assert os.path.isfile(image_path)
    # assert os.path.isdir(mapkurator_coords_dir)
    # assert os.path.isfile(segmentation_json_path)


    # corner_texts_dict = dict()
    # json_file_paths = sorted(os.listdir(mapkurator_coords_dir))
    # map_id = os.path.basename(mapkurator_coords_dir)
    # for json_path in json_file_paths:
        
    #     crop_id = json_path.split('_')[1].split('.json')[0]
    #     with open(os.path.join(mapkurator_coords_dir, json_path), 'r') as f:
    #         mapkurator_json = json.load(f)
        
    #     if 'text' not in mapkurator_json: 
    #         continue 
    #     else:
    #         text = mapkurator_json['text'] 
    #         polygon_x = mapkurator_json['polygon_x']
    #         polygon_y = mapkurator_json['polygon_y']
    #     if len(text) == 0: 
    #         continue 

    #     corner_texts_dict[crop_id] = text 

    # import pdb 
    # pdb.set_trace()




if __name__ == '__main__': 
    # main()

    image_dir = '/home/yaoyi/shared/critical-maas/9month/validation_corner_crops_1000/'
    # image_dir = '/home/yaoyi/shared/critical-maas/9month/validation_corner_crops/'

    output_dir = '../output/validation1000_debug'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        remove_all_files_in_directory(output_dir)

    for image_name in val_process_list:
        for i in range(0, 4):
            image = image = cv2.imread(os.path.join(image_dir, image_name))
            process_image(image_dir, image_name + '_' + str(i) +'.jpg', output_dir, point_of_interest = None, threshold_distance = 50)

