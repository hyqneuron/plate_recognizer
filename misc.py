

def find_plates_with_auto_label(min_area=1000, max_area=99999):
    folder_keys = sorted(db2.db.folder_registry.keys())
    found_plates = {}
    for folder_key in folder_keys:
        folder = db2.db.folder_registry[folder_key]
        for frame in folder.frames:
            if not frame.has_run_rfcn(): continue
            for car_bbox in frame.parts:
                for plate_bbox in car_bbox.parts:
                    if plate_bbox.name != 'plate': continue
                    x,y,w,h = plate_bbox.xywh()
                    if w*h < min_area or w*h > max_area: continue
                    if not plate_bbox.has_auto_sequence(): continue
                    if plate_bbox.auto_sequence_confirmed():
                        sequence_str = ''.join([char for x,y,char in plate_bbox.auto_sequence()])
                        found_plates[sequence_str] = True
                    else:
                        pass
    return found_plates

