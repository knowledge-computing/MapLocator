import os 
import numpy as np
import geopandas as gpd
import argparse
import json
import pandas as pd 
import glob 
import pdb 


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--input_folder', type=str, default='outputs/rio_geonames_08010_step1')
    parser.add_argument('--output_folder', type=str, default='outputs/rio_geonames_08010_step1')

    return parser.parse_args()


def filter_by_statename(args, map_name, state):
    with open(f'{args.input_folder}/{map_name}_matched_sites.json', 'r') as f:
        raw_place_names = json.load(f)

    # Filter out empty matches
    place_names = {key: value for key, value in raw_place_names.items() if len(value['candidate_matches']) != 0}

    # # Initial approximate center
    # latitudes, longitudes = [], []
    # for entry in place_names.values():
    #     candidates = entry['candidate_matches']
    #     for candidate in candidates:
    #         lat = candidate['lat']
    #         lon = candidate['lon']
    #         latitudes.append(lat)
    #         longitudes.append(lon)

    #         admin1_code = candidate['admin1_code']

    #         if admin1_code != 'state':
    #             continue 
    for entry in place_names.values():
        entry['candidate_matches'] = [
            c for c in entry['candidate_matches']
            if c.get('admin1_code') == state
        ]

    # Optionally remove entries that have no matches after filtering
    place_names = {
        k: v for k, v in place_names.items() if v['candidate_matches']
    }

    # Save filtered results
    with open(f'{args.output_folder}/{map_name}_inroi_matched_sites.json', 'w') as f:
        json.dump(place_names, f, indent=2)

    print(f"Filtered matches saved to {args.output_folder}/{map_name}_matched_sites_inroi.json")





if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    
    # map_list = glob.glob(args.input_folder + '/*_matched_sites.json')
    # map_list = sorted([os.path.basename(a).split('_')[0] for a in map_list])
    map_list = ['geo_map','geology_map_south', 'Teacup_pluton_alt_map', '101130GES013811_3']
    state_list = ['AK', 'AK', 'AZ', 'AZ']

    print(f"number of maps to process: {len(map_list)}")


    for map_name, state in zip(map_list, state_list):
        filter_by_statename(args, map_name, state)
