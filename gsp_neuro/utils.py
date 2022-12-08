import re
import csv


def extract_roi(regions):
    single_string = False
    if isinstance(regions, str):
        regions = [regions]
        single_string = True
    
    rois = []
    for region in regions:
        m = re.search('ctx-(.+?) ', region)
        if m:
            found = m.group(1)
            rois.append(found)
    print("{:5d} ROIs found".format(len(rois)))
    if single_string:
        rois = rois[0]
    return rois

def regions_in_file(file):
    with open(file, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
    regions = headers[9:-1:2]
    regions = extract_roi(regions)
    return regions

def split_lr_rois(ROIs):
    rh_rois = [roi.replace('rh-','') for roi in ROIs if roi.startswith('rh')]
    lh_rois = [roi.replace('lh-','') for roi in ROIs if roi.startswith('lh')]
    return {'rh' : rh_rois, 'lh': lh_rois}