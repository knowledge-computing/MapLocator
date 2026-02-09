import os 
import sys
import geopandas as gpd
from shapely.ops import unary_union
import pandas as pd
from shapely.geometry import Point, Polygon, box
from shapely.validation import explain_validity
import numpy as np 
import multiprocessing as mp
from utils_abbr import map_acronyms
import argparse
import json 
import re
import time 
from collections import defaultdict
from utils_fuzzy import search, build_tfidf_lsh_index_parallel
import pickle
import warnings
import pdb 


sys.path.insert(0, "/Users/li00266/Documents/georef-rectify/code_system")
from utils_corner_closepoi import read_map_content_area_from_json


import warnings
warnings.filterwarnings(
    "error",
    message="invalid value encountered in unary_union"
)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--input_geojson_folder', type=str, default = '../data/ngmdb/linking2/nickel/geojson/' )
    parser.add_argument('--input_seg_folder', type=str, default='../data/map_seg_output/nickel/', help='[Optional] Folder for map plot area segmentation')
    parser.add_argument('--output_folder', type=str, default = 'outputs_paper/ngmdb_nickel_0114_geonames_step1/')


    parser.add_argument('--thresh', type=float, default=0.8, help = 'matching similarity threshold')
    parser.add_argument('--matching', type=str, default = 'fuzzy', choices = ['fuzzy'], help = 'matching method')
    parser.add_argument('--reference', type=str, default = 'geonames', choices = ['mrds', 'roi','kg', 'geonames','gnis'], help = 'reference dataset')
    parser.add_argument('--country', type=str, default = 'US', choices = ['US', 'GB','AU']) # australia
    
    parser.add_argument('--lsh_pickle_path', type=str, default= None, help='reference database processed with lsh')
    
    parser.add_argument('--remove_common_suffix', action='store_true', dest='if_remove_common_suffix', help='Remove the phrase if the phrase is consist of only common suffix like lake, stream, etc.')
    

    return parser.parse_args()
    

def close_ring(coords):
    if coords[0] != coords[-1]:
        coords = coords + [coords[0]]
    return coords



def read_geojson_safe(path):
    with open(path, "r") as f:
        data = json.load(f)

    records = []

    for feat in data["features"]:
        geom = feat.get("geometry")
        props = feat.get("properties", {})

        if geom is None or geom.get("type") != "Polygon":
            continue

        ring = geom["coordinates"][0]

        # flip y
        ring = [[x, -y] for x, y in ring]

        ring = close_ring(ring)

        ring = np.asarray(ring, dtype=float)

        # ---- NaN / Inf guard (CRITICAL) ----
        if not np.isfinite(ring).all():
            continue

        try:
            poly = Polygon(ring)
        except Exception:
            continue

        # ---- validity check ----
        if not poly.is_valid or poly.is_empty:
            # optional debug:
            # print("Invalid polygon:", explain_validity(poly))
            continue

        records.append({**props, "geometry": poly})

    return gpd.GeoDataFrame(records, geometry="geometry")

# def read_geojson_safe(path):
#     with open(path, "r") as f:
#         data = json.load(f)

#     records = []

#     for feat in data["features"]:
#         geom = feat["geometry"]
#         props = feat["properties"]

#         if geom["type"] != "Polygon":
#             continue

#         ring = geom["coordinates"][0]
#         ring = [[x, -y] for x, y in ring]

#         ring = close_ring(ring)

#         try:
#             poly = Polygon(ring)
#             if not poly.is_valid:
#                 continue
#         except Exception:
#             continue

#         records.append({**props, "geometry": poly})

#     return gpd.GeoDataFrame(records, geometry="geometry")


# Function to safely convert location to Point objects
def create_point(location):
    if isinstance(location, str) and 'POINT' in location:
        try:
            coords = location.split('(')[1].strip(')').split()
            return Point(float(coords[0]), float(coords[1]))
        except (IndexError, ValueError):
            return None
    return None


# Function to safely convert latitude and longitude to Point objects
def create_point_from_lat_lon(lat, lon):
    try:
        return Point(float(lon), float(lat))  # Longitude comes first in Point (x, y)
    except (ValueError, TypeError):
        return None



def get_full_name(name): # get all possible full names
    full_names = set()  # Use set to avoid duplicates
    found = False

    for abbr, full_list in map_acronyms.items():
        # Match abbreviation only if it stands alone (not part of another word)
        pattern = r'(?<!\w)' + re.escape(abbr) + r'(?!\w)'
        if re.search(pattern, name):
            found = True
            for full in full_list:
                full_name = re.sub(pattern, full, name)
                full_names.add(full_name)

    # If no abbreviation was matched, return the original name
    if not found:
        full_names.add(name)

    return list(full_names)

def find_candidates(name, gdf, matching, namekey, thresh, vectorizer, X, nn, lsh, minhashes):

    # matches = fuzzy_match_ratio(name, gdf[namekey], threshold=thresh)
    # matches = tfidf_nn_match(name, vectorizer, X, nn, gdf[namekey], threshold=thresh, n_neighbors=50)

    matches = search(name, vectorizer, X, nn, lsh, minhashes, gdf[namekey], threshold=thresh, n_neighbors=50)

    return matches



def process_name(name, gdf, vectorizer, X, nn, matching, namekey, thresh, lsh, minhashes, dataset_type='geonames'):
    results = []
    
    matches = find_candidates(name, gdf, matching, namekey, thresh, vectorizer, X, nn, lsh, minhashes)

    if '.' in name: #potentially has a full name
        full_names = get_full_name(name)
        for full_name in full_names:
            if full_name != name: # a full name exists
                full_name_matches = find_candidates(name, gdf, matching, namekey, thresh, vectorizer, X, nn, lsh, minhashes)

                matches.extend(full_name_matches)


    for match in matches:
        match_site_name, score, index = match
        geom = gdf.iloc[index]['geometry']

        if dataset_type == 'geonames':
            result_entry = {
                'index': str(index),
                'name': match_site_name,
                'lat': geom.y,
                'lon': geom.x,
                'score': score,
                'asciiname': str(gdf.iloc[index].get('asciiname', '')),
                'alternatenames': str(gdf.iloc[index].get('alternatenames', '')),
                'feature_class': str(gdf.iloc[index].get('feature_class', '')),
                'feature_code': str(gdf.iloc[index].get('feature_code', '')),
                'country_code': str(gdf.iloc[index].get('country_code', '')),
                'cc2': str(gdf.iloc[index].get('cc2', '')),
                'admin1_code': str(gdf.iloc[index].get('admin1_code', '')),
                'admin2_code': str(gdf.iloc[index].get('admin2_code', '')),
                'admin3_code': str(gdf.iloc[index].get('admin3_code', '')),
                'admin4_code': str(gdf.iloc[index].get('admin4_code', '')),
                'population': str(gdf.iloc[index].get('population', ''))
            }
        elif dataset_type == 'gnis':
            result_entry = {
                'index': str(index),
                'name': match_site_name,
                'lat': geom.y,
                'lon': geom.x,
                'score': score,
                'feature_class': str(gdf.iloc[index].get('feature_class', '')),
                'state_name': str(gdf.iloc[index].get('state_name', '')),
                'county_name': str(gdf.iloc[index].get('county_name', ''))
            }
        else:
            result_entry = {
                'index': str(index),
                'name': match_site_name,
                'lat': geom.y,
                'lon': geom.x,
                'score': score
            }
        # else:
        #     raise ValueError(f"Unknown dataset_type: {dataset_type}")

        results.append(result_entry)

    return results



def is_number_like(text):
    # Remove leading/trailing whitespace
    stripped = text.strip()
    # Match: string that consists only of digits, spaces, hyphens, periods, or commas
    return re.fullmatch(r'[\d\s\-\.,]+', stripped) is not None


def load_tfidf_lsh_index(index_path):
    with open(index_path, "rb") as f:
        index_data = pickle.load(f)
    return index_data


def remove_common_suffix_as_single_phrase(args, phrase_list):
    if args.reference == 'geonames':
        common_suffix_file = 'geonames_suffix.csv'
    elif args.reference == 'gnis':
        common_suffix_file = 'gnis_suffix.csv'
    else:
        raise NotImplementedError

    suffix_df = pd.read_csv(common_suffix_file)


    # The CSV produced earlier has: index: suffix, column: count
    # So the suffix words are in the first column (regardless of header)
    suffix_list = suffix_df.iloc[:, 0].astype(str).str.lower().tolist()
    suffix_set = set(suffix_list)


    cleaned_phrases = []

    # ---- Filter phrases ----
    filtered = []
    for ph in phrase_list:
        text = ph.get("text", "").strip()

        # Normalize phrase: collapse multiple spaces, lowercase
        tokens = text.split()
        
        # If phrase has a SINGLE word AND that word is in suffix list -> skip
        if len(tokens) == 1 and tokens[0].lower() in suffix_set:
            print(tokens)
            continue

        # Otherwise keep it
        filtered.append(ph)

    num_removed = len(phrase_list) - len(filtered)
    if num_removed > 0:
        print(f"Removed {num_removed} phrases")

    return filtered



def process_name_list(phrase_list, gdf, vectorizer, X, nn, matching, namekey, thresh, lsh, minhashes, dataset_type='geonames'):
    matching_dict = {} 


    # for name in name_list:
    for phrase_entry in phrase_list:

        name = phrase_entry['text']


        matches = process_name(name, gdf, vectorizer, X, nn, matching, namekey, thresh, lsh, minhashes, dataset_type)

        if len(matches) > 0:
            matching_dict[name] = {'candidate_matches': matches}
            if 'vertices_list' in phrase_entry:
                matching_dict[name]['vertices_list'] = phrase_entry['vertices_list']

    return matching_dict




def prepare_phrase_list(args):
    image_phrase_dict = {}

    geojson_files = sorted([
        f for f in os.listdir(args.input_geojson_folder)
        if f.endswith(".geojson")
    ])

    print(f"Found {len(geojson_files)} geojson files")

    for fname in geojson_files:
        geojson_path = os.path.join(args.input_geojson_folder, fname)
        map_name = os.path.splitext(fname)[0]


        print(f"Processing {map_name}")


        # if map segmentation folder is provided, then use it to filter out phrases outside of plot area
        if args.input_seg_folder:
            map_seg_path = os.path.join(args.input_seg_folder,f"{map_name}_map_segmentation.json")
            if not os.path.exists(map_seg_path):
                print(f"[Error] Missing segmentation file for image: {seg_path}")
                return 
            else:
                # map_area_bbox is in the format [xmin, ymin, xmax, ymax]
                map_area_bbox = read_map_content_area_from_json(map_seg_path, use_bbox = True)
                xmin, ymin, xmax, ymax = map_area_bbox
                xmin, ymin, xmax, ymax = map(int, map(round, (xmin, ymin, xmax, ymax)))
                seg_box = box(xmin, ymin, xmax, ymax)

        gdf = read_geojson_safe(geojson_path)

        phrase_list = []

        # Ensure correct ordering
        gdf["word_id"] = gdf["word_id"].astype(int)

        for group_id, sub in gdf.groupby("group_id"):
            sub = sub.sort_values("word_id")

            phrase = " ".join(sub["text"])
            if phrase.isdigit() or len(phrase.replace(" ", ""))<=3 : # remove pure digit as a phrase or phrase less than 3 characters/number
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                phrase_geom = unary_union(sub.geometry)
            

            
            if args.input_seg_folder and (not phrase_geom.intersects(seg_box)):
                continue

            phrase_list.append({
                "text": phrase,
                "vertices_list": [list(g.exterior.coords) for g in sub.geometry],
                "phrase_geometry": phrase_geom   # optional but useful
            })

            if args.if_remove_common_suffix:
                phrase_list = remove_common_suffix_as_single_phrase(args, phrase_list)


        print(f"  #phrases: {len(phrase_list)}")
        print([p["text"] for p in phrase_list])

        image_phrase_dict[map_name] = phrase_list

    return image_phrase_dict




def main(args, image_phrase_dict):


    first_k = 200

    if args.reference == 'mrds':
        df = pd.read_csv('../data/mrds.csv')  # Replace with your file path
    elif args.reference == 'roi':
        df = pd.read_csv('../data/tungsten_target_region.csv')  # Replace with your file path
    elif args.reference == 'kg':
        with open('../data/minmod_kg_merge.json', 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        # df = df.drop_duplicates()
    elif args.reference == 'geonames':
        
        # Define column names
        columns = [
            "geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude",
            "feature_class", "feature_code", "country_code", "cc2", "admin1_code",
            "admin2_code", "admin3_code", "admin4_code", "population", "elevation",
            "dem", "timezone", "modification_date"
        ]

        # Load the text file into a DataFrame
        file_path = '../data/geonames_US.txt'  # Replace with full allcountries path
        # file_path = '../data/geonames_CI.txt'  # 

        df = pd.read_csv(file_path, sep='\t', header=None, names=columns)
        df['name'] = df['name'].astype(str)

    elif args.reference == 'gnis':
        gnis_txt_path = '../data/GNIS/DomesticNames_National_Text/Text/DomesticNames_National.txt'
        df = pd.read_csv(gnis_txt_path, sep='|')
        df['feature_name'] = df['feature_name'].astype(str)
    else:
        raise NotImplementedError


    # Display the first few rows of the DataFrame
    print('First few rows of the reference database:')
    print(df.head())



    print(f'Number of records {len(df)}' )

    if 'location' in df.columns: 
        # Apply the function to create the geometry column
        df['geometry'] = df['location'].apply(create_point)
    elif 'loc_wkt' in df.columns:
        df['geometry'] = df['loc_wkt'].apply(create_point)
    elif 'latitude' in df.columns: # geonames?
        # Apply the function to create the geometry column
        df['geometry'] = df.apply(lambda row: create_point_from_lat_lon(row['latitude'], row['longitude']), axis=1)
    elif 'prim_lat_dec' in df.columns: # gnis, another option is to use source_lat_dec, source_long_dec, which might be the location in the map source
        # prim_lat_dec / prim_long_dec:
        # This is the primary (official) coordinate for the named feature as determined by GNIS.
        # GNIS defines this as the canonical location of the feature, which might have been derived or validated from multiple sources.

        # source_lat_dec / source_long_dec:
        # This is the coordinate as originally sourced or reported, possibly from historical documents, maps, or data submissions.
        # May be less accurate or not adjusted to a consistent geodetic datum (e.g., NAD83 or WGS84) compared to the primary coordinates.
        df['geometry'] = df.apply(lambda row: create_point_from_lat_lon(row['prim_lat_dec'], row['prim_long_dec']), axis=1)
    else:
        print('No valid geometry column found, stopped.')
        raise NotImplementedError

    if 'linkage' in df.columns: # KG data
        # Convert lists in the column to tuples
        df['linkage'] = df['linkage'].apply(lambda x: tuple(x) if isinstance(x, list) else x)


    # Filter out rows where geometry is None
    df = df[df['geometry'].notnull()]



    # Create a GeoDataFrame, assuming CRS is EPSG:4326
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")


    if args.reference == 'kg':
        namekey = 'ms_name' 
    elif args.reference == 'geonames':
        namekey = 'name'
    elif args.reference == 'gnis':
        namekey = 'feature_name'
    elif args.reference == 'roi' or args.reference == 'mrds':
        namekey = 'site_name'
    else:
        raise NotImplementedError
    

    # build tf-idf nn index for geonames
    # vectorizer, X, nn = build_tfidf_nn_index(gdf[namekey])
    # vectorizer, X, nn, lsh, minhashes = build_indices(gdf[namekey])


    if args.lsh_pickle_path and os.path.exists(args.lsh_pickle_path): # if the preprocessed lsh file exists
        index_data = load_tfidf_lsh_index(args.lsh_pickle_path)
        vectorizer = index_data["vectorizer"]
        X = index_data["tfidf_matrix"]
        nn = index_data["nn"]
        lsh = index_data["lsh"]
        minhashes = index_data["minhashes"]
        choices = index_data["choices"]
    else:
        lsh_pickle_path = os.path.join(args.output_folder, f'{args.reference}_lsh.pkl')

        vectorizer, X, nn, lsh, minhashes = build_tfidf_lsh_index_parallel(
            gdf[namekey],
            ngram_range=(3, 4),
            lsh_threshold=0.8,
            num_perm=128,
            n_jobs=4  # or None to use all cores - 1
        )

        # Save everything
        os.makedirs(os.path.dirname(args.output_folder), exist_ok=True)
        with open(lsh_pickle_path, "wb") as f:
            pickle.dump({
                "vectorizer": vectorizer,
                "tfidf_matrix": X,
                "nn": nn,
                "lsh": lsh,
                "minhashes": minhashes,
                "choices": gdf[namekey]
            }, f)


    output_folder = args.output_folder

   
        
    for map_name, phrase_list in image_phrase_dict.items():
        # matched_sites_dict = find_target_entities_every(query_names, gdf, matching = args.matching, namekey = namekey, threshold = args.thresh )
        matched_sites_dict = process_name_list(phrase_list, gdf, vectorizer, X, nn, matching = args.matching, namekey = namekey, thresh = args.thresh, lsh = lsh, minhashes = minhashes , dataset_type=args.reference)
        
        # Save to JSON file with 4-space indentation
        with open(f"{output_folder}/{map_name}_matched_sites.json", "w") as f:
            json.dump(matched_sites_dict, f, indent=4)




def merge_patches(args, stride=1000):
    folder = os.path.join(args.output_folder, 'temp')

    # Collect matched_sites files grouped by prefix
    matched_files = defaultdict(list)
    for filename in os.listdir(folder):
        if filename.endswith("_matched_sites.json"):
            prefix = filename.replace("_matched_sites.json", "")
            matched_files[prefix].append(os.path.join(folder, filename))

    # Group by base id prefix (before `_h`)
    grouped = defaultdict(list)
    for prefix in matched_files:
        base_id = prefix.split("_h")[0]
        grouped[base_id].append(prefix)

    # Merge JSON contents and write to a single file per base_id
    for base_id, prefixes in grouped.items():
        merged_dict = {}
        for prefix in prefixes:
            file_path = os.path.join(folder, f"{prefix}_matched_sites.json")
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, dict):
                        print(f"Warning: {file_path} does not contain a dictionary, skipping.")
                        continue
                    if not data:  # Skip empty dict
                        print(f"Skipping empty dict in {file_path}")
                        continue
                    for k, v in data.items():
                        if k in merged_dict:
                            print(f"Warning: Key '{k}' already exists in merged dict. Overwriting.")
                        merged_dict[k] = v
                except json.JSONDecodeError:
                    print(f"Warning: Failed to decode {file_path}")

        # Save merged data
        output_path = os.path.join(args.output_folder, f"{base_id}_matched_sites.json")
        with open(output_path, 'w') as f:
            json.dump(merged_dict, f, indent=2)

        print(f"Merged {len(prefixes)} files into {output_path}")



if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    phrase_dict = prepare_phrase_list(args)
    
    os.makedirs(args.output_folder, exist_ok=True)

    main(args, phrase_dict) 



# python 1_find_candidates.py --input_seg_folder='../data/map_seg_output/nickel/' --input_geojson_folder='../data/ngmdb/linking2/nickel/geojson' --output_folder='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_step1/' --reference='geonames' --remove_common_suffix --lsh_pickle_path='lsh_cache/geonames_lsh.pkl'

# python 1_find_candidates.py --input_geojson_folder='../data/ngmdb/linking2/nickel/geojson' --output_folder='outputs_paper/ngmdb_nickel_0114_geonames_step1/' --reference='geonames' --remove_common_suffix --lsh_pickle_path='lsh_cache/geonames_lsh.pkl'


