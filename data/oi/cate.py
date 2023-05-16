# extract category from the json file

import json
import os

def extract_cate():
    cate = set()
    with open('val.json', 'r') as f:
        data = json.load(f)
        for row in data['categories']:
            print(row['supercategory'])
            cate.add(row['supercategory'])
    return cate


if __name__ == '__main__':
    cate = extract_cate()
    print(list(cate))