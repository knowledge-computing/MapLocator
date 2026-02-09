
import os
import sys
import io
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import numpy as np
import logging 
import time 
import json 
import re
from fuzzywuzzy import fuzz
import pdb 


sys.path.insert(0, "/Users/li00266/Documents/georef-rectify/code_system")
from utils_corner_closepoi import read_map_content_area_from_json


def parse_title_extraction_result(json_path):
    with open(json_path, 'r') as f:
        title_info = json.load(f)

    return title_info 

def normalize(text):
    if pd.isnull(text):
        return ""
    return (
        text.lower()
            .replace(".", "")
            .replace(",", " ")
            .strip()
    )


# if query name contains multiple states, use this function to split to meaningful parts 
def split_locations(location_text):
    if not location_text:
        return []

    # If already a list,  normalize each element
    if isinstance(location_text, list):
        return [normalize(s) for s in location_text if normalize(s)]

    # Otherwise assume string, split on hyphen, comma, slash, or "and"
    parts = re.split(r"[-,/]| and ", location_text.lower())
    return [normalize(p) for p in parts if normalize(p)]



def load_topo_data(topo_meta_path):
    df_topo = pd.read_csv(topo_meta_path) 

    # clean up and unify string variations
    df_topo["state_norm"] = df_topo["primary_state"].apply(normalize)
    df_topo["map_name_norm"] = df_topo["map_name"].apply(normalize)

    df_topo["county_norm_list"] = df_topo["county_list"].fillna("").apply(
        lambda x: sorted([normalize(c) for c in x.split(",") if c.strip()])
    )

    df_topo["county_key"] = df_topo["county_norm_list"].apply(lambda x: "|".join(x))

    # drop duplicate records 
    df_topo = df_topo.drop_duplicates(
        subset=["map_name_norm", "state_norm", "map_scale", "county_key"],
        keep="first"   # or "last"
    ).reset_index(drop=True)


    return df_topo


def get_topomaps_metadata(df):
    cand_text_list = []
    map_rowid_list = []
    for index, row in df.iterrows():
        cand_text = row['map_name']  

        if not pd.isnull(row['county_list']): 
            cand_text = cand_text + ' ' + row['county_list'].replace(',', ' ')

        cand_text = cand_text + ' ' + row['primary_state'] 
        
        cand_text_list.append(cand_text)
        map_rowid_list.append(index)

    return cand_text_list, map_rowid_list


def location_score(query_county, counties):
    if not query_county or not counties:
        return 0
    return max(
        fuzz.token_set_ratio(query_county, c)
        for c in counties
    )


def load_seg_file(seg_path):

    map_area_bbox = read_map_content_area_from_json(seg_path, use_bbox = True)

    xmin, ymin, xmax, ymax = map_area_bbox
    xmin, ymin, xmax, ymax = int(xmin),int(ymin), int(xmax), int(ymax)

    # Four corner points (x, y)
    bounding_box = [
        (xmin, ymin),  # top-left
        (xmax, ymin),  # top-right
        (xmax, ymax),  # bottom-right
        (xmin, ymax)   # bottom-left
    ]

    return bounding_box


def main(args):

    input_title_dir = args.input_title_dir

    json_list = [
        f for f in os.listdir(input_title_dir)
        if os.path.isfile(os.path.join(input_title_dir, f))
        and f.lower().endswith('.json')
    ]
    json_list = sorted(json_list)
    json_list = [os.path.join(input_title_dir, a) for a in json_list]


    df_topo = load_topo_data(args.topo_meta_path)

    print(f"{len(df_topo)} records to compare with")


    cnt = 0

    num_skip = 0 
    for json_path in json_list:

        map_name = os.path.basename(json_path).split('_title')[0]
        print('\n')
        print(map_name)

        # # # DEBUG
        # map_name = '7e171b5713c706c52c564ad74b77536750c771c731cf799f68969cda01f23192'
        # json_path = os.path.join(args.input_title_dir, f"{map_name}_title.json")



        #######################################

        location_info = parse_title_extraction_result(json_path)
        print(location_info)


        seg_path = os.path.join(args.input_seg_dir, map_name + '_map_segmentation.json')
        print(seg_path)

        map_area_bbox = read_map_content_area_from_json(seg_path, use_bbox = True)
        xmin, ymin, xmax, ymax = map_area_bbox
        xmin, ymin, xmax, ymax = int(xmin),int(ymin), int(xmax), int(ymax)


        ######################################

        # split and normalize each state 
        query_states = split_locations(location_info["state"])

        if query_states: # merge all info in the matched states if query_states is non-empty
            df_state = df_topo[df_topo["state_norm"].isin(query_states)].copy()
        else:
            df_state = df_topo.copy()


        ########################################

        query_county = split_locations(location_info["county"])
        query_quad = split_locations(location_info["quadrangle"])

        try:
            assert len(query_county) > 0 or len(query_quad) > 0
        except:
            num_skip += 1
            print("Both county and quadrangle are empty")
            continue 


        df_state["county_score"] = df_state["county_norm_list"].apply(
            lambda counties: location_score(query_county, counties)
        )

        df_state["quad_score"] = df_state["map_name_norm"].apply(
            lambda quad: location_score(query_quad, quad)
            )


        df_state["final_score"] = (
            0.65 * df_state["quad_score"]
          + 0.35 * df_state["county_score"]
        )

        topk = (
            df_state
            .sort_values("final_score", ascending=False)
            .head(10)
        )

        print(len(topk))
        print(topk[['product_filename', 'county_list']])
        # print(topk)

        top1 = topk.iloc[0]

        westbc = top1['westbc']
        eastbc = top1['eastbc']
        northbc = top1['northbc']
        southbc = top1['southbc']

        # print(top1[['product_filename', 'county_list']])
        
        # --------------- ymin 
        # |          |
        # |          |
        # |          |
        # --------------- ymax

        # xmin       xmax
        # |          |
        # ------------
        # |          |
        # |          |
        # ------------


        gcp_0 = [ymin, xmin, northbc, westbc]
        gcp_1 = [ymin, xmax, northbc, eastbc]
        gcp_2 = [ymax, xmax, southbc, eastbc]
        gcp_3 = [ymax, xmin, southbc, westbc]

        gcps = [
            ["0", ymin, xmin, northbc, westbc],
            ["1", ymin, xmax, northbc, eastbc],
            ["2", ymax, xmax, southbc, eastbc],
            ["3", ymax, xmin, southbc, westbc],
        ]

        gcp_df = pd.DataFrame(
            gcps,
            columns=[
                "gcp_id",
                "rows_from_top",
                "columns_from_left",
                "latitude",
                "longitude",
            ]
        )

        output_path = os.path.join(args.output_dir, f"{map_name}_gcp.csv")

        gcp_df.to_csv(output_path, index=False)

        # cnt += 1
        # if cnt > 1:
        #     break

    print(f'{num_skip} maps skipped.')
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_seg_dir', type=str, default='../data/map_seg_output/nickel/')
    parser.add_argument('--input_title_dir', type=str, default='output_title/nickel/')
    parser.add_argument('--topo_meta_path', type=str, default='/Users/li00266/Documents/critical-maas/topo_meta/ustopo_historical_current.csv') 
    parser.add_argument('--output_dir', type=str, default='output_toporetrieval/nickel/') 

    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    os.makedirs(args.output_dir, exist_ok = True)

    main(args)

