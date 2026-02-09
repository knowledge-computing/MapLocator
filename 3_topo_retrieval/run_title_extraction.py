import os
import io
# from utils_gpt import get_toponym_tokens, prepare_bm25, fuzzy_find_top_k_matching
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import numpy as np
import base64
from openai import OpenAI
import requests
import logging 
import time 
# import torch
import json 


Image.MAX_IMAGE_PIXELS = None
api_key = os.getenv("OPENAI_API_KEY")



def load_data(topo_histo_meta_path, topo_current_meta_path):

    # topo_histo_meta_path = 'support_data/historicaltopo.csv'
    df_histo = pd.read_csv(topo_histo_meta_path) 
            
    # topo_current_meta_path = 'support_data/ustopo_current.csv'
    df_current = pd.read_csv(topo_current_meta_path) 

    # common_columns = df_current.columns.intersection(df_histo.columns)

    # df_merged = pd.concat([df_histo[common_columns], df_current[common_columns]], axis=0)
    df_merged = df_histo #TODO: tempory fix to get Geotiff URL

    bm25 = prepare_bm25(df_merged)

    return bm25, df_merged


def get_topo_basemap(query_sentence, bm25, df_merged, device ):

    print(query_sentence)
    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence, device)
    print(human_readable_tokens)

    query_sent = ' '.join(human_readable_tokens)

    topk = fuzzy_find_top_k_matching(query_sent, df_merged, k=10)
    fuzzy_top10 = df_merged.iloc[[a[0] for a in topk]]

    tokenized_query = query_sent.split(" ")
    
    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]
    
    # Top 10
    top10 = df_merged.iloc[sorted_bm25_list[0:10]]

    # top1 = df_merged.iloc[sorted_bm25_list[0]]

    # return top10, query_sent 
    return fuzzy_top10, query_sent


# Function to encode the image
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image

def getTitle(base64_image, max_trial = 10):

    if base64_image is None:
        return "No file selected"


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4o",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "What's the title of map? please just return the title no more words else"
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    cnt_trial = 1

    
    while ('choices' not in response and cnt_trial < max_trial):
        time.sleep(5) # sleep for 5 seconds before sending the next request
        print('Title extraction failed, retrying...')
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json() 
        cnt_trial += 1 
        # import pdb 
        # pdb.set_trace()


    if 'choices' in response:
        return response['choices'][0]['message']['content']
    else:
        return -1



# check and downscale:

def downscale(image, max_size=10, max_dimension=9500):

    print("downscaling...")

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG")  

    img_size = buffer.tell() / (1024 * 1024)

    if img_size > max_size or max(image.width, image.height) > max_dimension:

        downscale_factor = max_size / img_size

        downscale_factor = max(downscale_factor, 0.1)

        new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))

        while True:

            # to aviod the case err "Maximum supported image dimension is 65500 pixels"
            while max(image.width, image.height) > max_dimension:
                downscale_factor = max_dimension / max(image.width, image.height)
                downscale_factor = max(downscale_factor, 0.1)
                new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))
                image=image.resize(new_size)

            downscaled_img = image.resize(new_size)
            buffer = io.BytesIO()
            downscaled_img.save(buffer, format="JPEG")
            downscaled_size = buffer.tell() / (1024 * 1024)


            if downscaled_size < max_size or max(downscaled_img.width, downscaled_img.height) < max_dimension:
                print("dimension now:")
                print(max(downscaled_img.width, downscaled_img.height))
                break  
            else:
                downscale_factor *= 0.8 


            new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))

        print("after downscaled, the new size is: ")
        print(new_size)

        downscaled_img = image.resize(new_size)

        return downscaled_img

    else:

        return image


def to_camel(title):
    words = title.split()
    return ' '.join(word.capitalize() for word in words)


def extract_location_info(gpt_title):
    """
    Call GPT-4o API to extract state, county, and quadrangle name from the gpt_title.
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Create the prompt for GPT-4o
    prompt = (
        f"Extract the state, county, and quadrangle name from the following geologic map title. "
        f"Return the results in JSON format with keys 'state', 'county', and 'quadrangle'. "
        f"Title: {gpt_title}"
    )

    # Call GPT-4o API
    response = client.chat.completions.create(
        model="gpt-4o",  # Use the correct model name
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=200,
        response_format={"type": "json_object"}  # Ensure the response is in JSON format
    )


    # Extract and parse the response
    try:
        response_content = response.choices[0].message.content
        result = json.loads(response_content)
        print(result)
        return result
    except json.JSONDecodeError as e:
        print(f"Failed to parse response as JSON: {e}")
        return None


def run_gpt_title(args):

    # input_path = '../input_data/CO_Frisco.png' # supported file format: png, jpg, jpeg, tif
    input_path = args.input_path
    
    
    image = Image.open(input_path).convert('RGB') # read image with PIL library

    print('image_size',image.size)
    

    if not os.path.isdir(args.temp_dir):
        os.makedirs(args.temp_dir)
    
    jpg_file_path = os.path.join(args.temp_dir, "output.jpg")

    image.save(jpg_file_path, format="JPEG")

    image =downscale(image)
    # Getting the base64 string
    base64_image = encode_image(image)
    title = getTitle(base64_image)

    if title == -1: # exception when extracting the title 
        logging.error('Failed to extract title, exit with code -1')
        return -1 

    title = to_camel(title)

    os.remove(jpg_file_path)

    return title 


def run_gpt_title_and_parse(args):
    title = run_gpt_title(args)

    location_info = extract_location_info(title)
    location_info.update({'gpt_title':title})

    # output_json_path = os.path.join(args.output_dir, 'test_output.json')
    # # Save the updated data to the output JSON file
    # with open(output_json_path, 'w') as f:
    #     json.dump(location_info, f, indent=2)

    return location_info 


def run_georeferencing_gpt(args):
    # support_data_dir = args.support_data_dir

    # load data store in bm25 & df_merged
    # bm25, df_merged = load_data(topo_histo_meta_path = f'{support_data_dir}/historicaltopo.csv',
    #     topo_current_meta_path = f'{support_data_dir}/ustopo_current.csv')
    
    title = run_gpt_title(args)
    # query_sentence = title

    # # get the highest score: 
    # top10, toponyms = get_topo_basemap(query_sentence, bm25, df_merged, device )
    return top10, title, toponyms


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_img_dir', type=str, default='output_title/nickel/')
    # parser.add_argument('--support_data_dir', type=str, default='/home/yaoyi/li002666/critical_maas/support_data/') 
    parser.add_argument('--temp_dir', type=str, default='temp/') 
    parser.add_argument('--output_dir', type=str, default='output_gcp/nickel/') 
    parser.add_argument('--extension', type=str, default='.cog.tif', help = 'file extension (everything after map name)') 

    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    os.makedirs(args.output_dir, exist_ok = True)

    input_img_dir = args.input_img_dir

    img_list = [
        f for f in os.listdir(input_img_dir)
        if os.path.isfile(os.path.join(input_img_dir, f))
        and f.lower().endswith(args.extension)
    ]
    img_list = sorted(img_list)
    img_list = [os.path.join(input_img_dir, a) for a in img_list]

    for img_path in img_list:
        map_name = os.path.basename(img_path).split('.')[0]
        print('\n')
        print(map_name)

        args.input_path = img_path 

        location_info = run_gpt_title_and_parse(args)

        output_path = os.path.join(args.output_dir, f"{map_name}_title.json")

        with open(output_path, 'w') as f:
            json.dump(location_info, f, indent=2)

    

if __name__ == "__main__":
    main()

