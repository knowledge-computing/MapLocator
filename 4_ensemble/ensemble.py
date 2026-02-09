import os 
import sys
import pandas as pd 
import shutil
import argparse


def get_map_names(folder_name):
    path_list = os.listdir(folder_name)

    map_names = [
        f.split('_gcp')[0]
        for f in path_list
        if f.endswith(".csv") and "_gcp" in f
    ]

    return set(map_names)   # return set for easy math



def ensemble_simple(args):
    geocoord_module_output = args.geocoord_module_output
    graph_module_output = args.graph_module_output
    retrieval_module_output = args.retrieval_module_output
    output_folder = args.output_folder


    geocoord_set = get_map_names(geocoord_module_output)
    graph_set = get_map_names(graph_module_output)
    retrieval_set = get_map_names(retrieval_module_output)

    all_results = geocoord_set | graph_set | retrieval_set
    print("Total maps:", len(all_results))

    for map_name in sorted(all_results):

        filename = f"{map_name}_gcp.csv"

        # --- Priority 1: geocoord ---
        if map_name in geocoord_set:
            src = os.path.join(geocoord_module_output, filename)
            source_module = "geocoord"

        # --- Priority 2: graph ---
        elif map_name in graph_set:
            src = os.path.join(graph_module_output, filename)
            source_module = "graph"

        # --- Priority 3: retrieval ---
        elif map_name in retrieval_set:
            src = os.path.join(retrieval_module_output, filename)
            source_module = "retrieval"

        else:
            # Should not happen, but safe guard
            continue

        dst = os.path.join(output_folder, filename)

        shutil.copy2(src, dst)
        print(f"Copied {filename} from {source_module}")



    # check the number of files to target folder
    print("Files in output folder:", len([f for f in os.listdir(output_folder) if f.endswith(".csv")]))




def parse_arguments():
    parser = argparse.ArgumentParser(description='Script to merge map data.')

    parser.add_argument('--geocoord_module_output', type=str,
default='../data/kdd2026_yijun/nickel/gcp/')
    parser.add_argument('--graph_module_output', type=str,
default='outputs_paper/ngmdb_nickel_0114_geonames_plotfiltered_gcp_step3/')  
    parser.add_argument('--retrieval_module_output',
type=str, default='../code_system/output_toporetrieval/nickel/') 

    
    parser.add_argument('--output_folder', type=str, default='outputs_paper/ngmdb_nickel_ensemble')


    return parser.parse_args()



if __name__ == '__main__':
    args = parse_arguments()
    print(args)

    os.makedirs(args.output_folder, exist_ok=True)

    ensemble_simple(args)

