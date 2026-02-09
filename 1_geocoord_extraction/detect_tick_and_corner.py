import os 
import sys
import glob 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import json 
import pdb 


sys.path.insert(0, "/Users/li00266/Documents/georef-rectify/code_system")
from utils_corner_closepoi import crop_patches_around_points, process_image, read_map_content_area_from_json

output_dir = 'output_tick_corner_v2'

# input_img_dir = '../data/USGS_PP_1300/pp1300_maps_324/image/'
# map_seg_dir = '../data/map_seg_output/pp1300/'
# output_tick_dir = f'{output_dir}/pp1300'
# extension = '.jpg'


# input_img_dir = '/Users/li00266/Documents/georef-rectify/data/ngmdb/raw_images/nickel'
# map_seg_dir = '../data/map_seg_output/nickel/'
# output_tick_dir = f'{output_dir}/nickel'
# extension = '.cog.tif'


input_img_dir = '/Users/li00266/Documents/georef-rectify/data/htmc505/geotiff/'
map_seg_dir = '../data/map_seg_output/htmc505/'
output_tick_dir = f'{output_dir}/htmc505'
extension = '.tif'


os.makedirs(output_tick_dir, exist_ok = True )


img_list = [
    f for f in os.listdir(input_img_dir)
    if os.path.isfile(os.path.join(input_img_dir, f))
    and f.lower().endswith(extension)
]
img_list = sorted(img_list)
img_list = [os.path.join(input_img_dir, a) for a in img_list]



# -----------------------------
# Parameters
# -----------------------------
BAND_RATIO = 0.02
MIN_TICK_RATIO = 0.004
MAX_TICK_RATIO = 0.03
ANGLE_TOL = np.deg2rad(10)
MIN_SPACING_RATIO = 0.006   # minimum distance between ticks

# -----------------------------
def crop_band(img, bbox, side, band_px):
    x0, y0, x1, y1 = map(int, bbox)
    if side == "top":
        return img[y0:y0+band_px, x0:x1], (x0, y0)
    if side == "bottom":
        return img[y1-band_px:y1, x0:x1], (x0, y1-band_px)
    if side == "left":
        return img[y0:y1, x0:x0+band_px], (x0, y0)
    if side == "right":
        return img[y0:y1, x1-band_px:x1], (x1-band_px, y0)
    raise ValueError


def suppress_close(points, min_dist):
    if len(points) == 0:
        return []

    points = sorted(points)
    kept = [points[0]]

    for p in points[1:]:
        if abs(p - kept[-1]) >= min_dist:
            kept.append(p)

    return kept

# -----------------------------
def detect_ticks_no_clustering(image, map_area_bbox):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    x0, y0, x1, y1 = map_area_bbox
    map_w = x1 - x0
    map_h = y1 - y0
    scale = min(map_w, map_h)

    band_px = int(BAND_RATIO * scale)
    min_len = MIN_TICK_RATIO * scale
    max_len = MAX_TICK_RATIO * scale
    min_spacing = MIN_SPACING_RATIO * scale

    lsd = cv2.createLineSegmentDetector(0)

    results = {k: [] for k in ["top", "bottom", "left", "right"]}

    for side in results:
        band, offset = crop_band(gray, map_area_bbox, side, band_px)
        edges = cv2.Canny(band, 50, 150)
        lines, _, _, _ = lsd.detect(edges)

        if lines is None:
            continue

        coords = []

        for l in lines:
            x1_, y1_, x2_, y2_ = l[0]
            length = np.hypot(x2_-x1_, y2_-y1_)
            angle = abs(np.arctan2(y2_-y1_, x2_-x1_))

            if not (min_len <= length <= max_len):
                continue

            if side in ("top", "bottom"):
                if not (np.pi/2 - ANGLE_TOL < angle < np.pi/2 + ANGLE_TOL):
                    continue
                coords.append((x1_ + x2_) / 2)
            else:
                if not (angle < ANGLE_TOL or abs(angle - np.pi) < ANGLE_TOL):
                    continue
                coords.append((y1_ + y2_) / 2)

        # suppress near-duplicates
        coords = suppress_close(coords, min_spacing)

        # convert to image coords
        for c in coords:
            if side == "top":
                results[side].append((int(x0 + c), int(y0)))
            elif side == "bottom":
                results[side].append((int(x0 + c), int(y1)))
            elif side == "left":
                results[side].append((int(x0), int(y0 + c)))
            elif side == "right":
                results[side].append((int(x1), int(y0 + c)))

    return results

# -----------------------------
def visualize(image, ticks, bbox):
    vis = image.copy()
    x0, y0, x1, y1 = map(int, bbox)
    cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,0), 2)

    colors = {
        "top": (255,0,0),
        "bottom": (0,255,0),
        "left": (0,0,255),
        "right": (255,255,0)
    }

    for side, pts in ticks.items():
        for x,y in pts:
            cv2.circle(vis, (x,y), 20, colors[side], -1)

    plt.figure(figsize=(20,20))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()




IF_VIEW_CORNER = False

for img_path in img_list:
    # map_name = os.path.basename(img_path).rsplit('.', 1)[0]
    map_name = os.path.basename(img_path).split('.')[0]
    print('\n')
    print(map_name)
    # img = Image.open(img_path)
    image = cv2.imread(img_path)
    try: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        pdb.set_trace()
    
    map_area_bbox = read_map_content_area_from_json(os.path.join(map_seg_dir, f"{map_name}_map_segmentation.json" ), use_bbox = True)
    

    ###################
    ## detect tick ##
    ###################

    ticks = detect_ticks_no_clustering(image, map_area_bbox)
    # visualize(image, ticks, map_area_bbox)
    print(f"Found {len(ticks)} ticks")

    

    for side, pts in ticks.items():
        print(f"{side}: {len(pts)} ticks")
    

    ###################
    ## corner refine ##
    ###################

    patch_list, coords_list = crop_patches_around_points(None, image, map_area_bbox, patch_size = (1000, 1000), if_write_file = False, bbox_format='xyxy' )

    xmin, ymin, xmax, ymax = map_area_bbox
    xmin, ymin, xmax, ymax = int(xmin),int(ymin), int(xmax), int(ymax)

    # Four corner points (x, y)
    bounding_box = [
        (xmin, ymin),  # top-left
        (xmax, ymin),  # top-right
        (xmax, ymax),  # bottom-right
        (xmin, ymax)   # bottom-left
    ]

    corner_name_list = ['top-left','top-right','bottom-right','bottom-left']
    
    corner_dict = {}
    for patch, coord, corner_name , seg_corner in zip(patch_list, coords_list,corner_name_list, bounding_box):
        
        # row, col 
        # corner_in_patch = process_image(patch, threshold_distance=80 )
        corner_in_patch = process_image(patch, point_of_interest = (seg_corner[0] - coord['top_left_y'], seg_corner[1] - coord['top_left_x']), threshold_distance=50 )
            
        if corner_in_patch != -1:
            # successful corner detection 

            corner_in_img = (corner_in_patch[0] + int(coord['top_left_y']), corner_in_patch[1] + int(coord['top_left_x'])) 
            corner_dict[corner_name] = {"col":corner_in_img[0], "row": corner_in_img[1]}

            if IF_VIEW_CORNER:
                cv2.circle(image, corner_in_img, 10, (255,0,0,0.5),-1)
                print(corner_in_img)
                plt.imshow(image[corner_in_img[1]-100:corner_in_img[1]+100,corner_in_img[0]-100:corner_in_img[0]+100,:])
                plt.show()

    # # output ticks are cols, rows 
    output_path = os.path.join(output_tick_dir, f"{map_name}_tick_corner.json")

    with open(output_path, 'w') as f:
        json.dump({'ticks':ticks, 'corners':corner_dict}, f, indent=2)


    
