import csv 
import json
import pdb

ocid_file = json.load(open('/home/mi/data/Datasets/3D_VG/OCID/val_expressions.json'))

with open('val.csv', mode='a+', newline='') as file:
    header = ['image_fpath', 'bbox', 'query']
    writer = csv.writer(file, delimiter=',')
    writer.writerow(header)
    
    for key in ocid_file.keys():
        ref = ocid_file[key]

        img = ref['scene_path']
        sentence = ref['sentence']
        bbox = ref['bbox']
        bbox = ref['bbox'].strip('[').strip(']').split(',')
        
        bbox = [int(x) for x in bbox]
        bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        
        data_frame = img, str(bbox), sentence
        writer.writerow(data_frame)