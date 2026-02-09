import os 
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import argparse
import json
import pandas as pd 
import glob 
import pdb 


def polygon_center(vertices):
    pts = np.array(vertices)
    cx = pts[:, 0].mean()  # column
    cy = pts[:, 1].mean()  # row
    return cy, cx  # return (row, col)


def produce_gcps(args):

    file_suffix = args.file_suffix
    
    input_json_list = [
        f for f in os.listdir(args.input_folder)
        if f.endswith(f"{file_suffix}.json")
    ]

    num_skip = 0 

    for json_path in input_json_list:

        map_name = json_path.split(file_suffix)[0]

        output_path = os.path.join(args.output_folder, f'{map_name}_gcp.csv')


        with open(os.path.join(args.input_folder, json_path), 'r') as f:
            graph_result_dict = json.load(f)


        if len(graph_result_dict) == 0:
            num_skip += 1
            continue 

        rows = []

        for gcp_id, record in graph_result_dict.items():

            if 'vertices_list' not in record or not record['vertices_list']:
                continue

            # Use first polygon if multiple
            vertices = record['vertices_list'][0]

            row, col = polygon_center(vertices)

            rows.append({
                "gcp_id": int(gcp_id),
                "rows_from_top": int(round(row)),
                "columns_from_left": int(round(col)),
                "latitude": float(record['lat']),
                "longitude": float(record['lon']),
            })

        gcp_df = pd.DataFrame(rows)
        gcp_df = gcp_df.sort_values("gcp_id")

        gcp_df.to_csv(output_path, index=False)


    print(f'{num_skip} files skipped')




def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--input_folder', type=str, default='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_scalecleaned_step2/')
    parser.add_argument('--file_suffix', type=str, default='_graph_output') # _graph_output_scalecleaned
    
    parser.add_argument('--output_folder', type=str, default='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_scalecleaned_gcp_step3/')


    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    produce_gcps(args)