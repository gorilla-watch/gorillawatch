from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import matplotlib.pyplot as plt
import os

def get_infos_for_constraints(image_path):
    split = image_path.split("_")
    video_name = split[1] + "_" + split[2] + "_" + split[3]
    camera_id = split[1]
    date = split[2]
    frame_number = split[4]
    tracklet_id = split[5].split(".")[0]
    return {
        "video_name": video_name,
        "camera_id": camera_id,
        "date": date,
        "frame_number": frame_number,
        "tracklet_id": tracklet_id,
    }

def is_same_gorilla(filename1, filename2):
    video_name1 = filename1["video_name"]
    tracklet_id1 = filename1["tracklet_id"]
    video_name2 = filename2["video_name"]
    tracklet_id2 = filename2["tracklet_id"]
    return video_name1 == video_name2 and tracklet_id1 == tracklet_id2

def is_different_gorilla(filename1, filename2):
    
    video_name1 = filename1["video_name"]
    camera_id1 = filename1["camera_id"]
    date1 = filename1["date"]
    frame_number1 = filename1["frame_number"]
    tracklet_id1 = filename1["tracklet_id"]
    
    video_name2 = filename2["video_name"]
    camera_id2 = filename2["camera_id"]
    date2 = filename2["date"]
    frame_number2 = filename2["frame_number"]
    tracklet_id2 = filename2["tracklet_id"]
    
    if camera_id1 != camera_id2 and date1 == date2:
        return True
    elif video_name1 == video_name2 and frame_number1 == frame_number2 and tracklet_id1 != tracklet_id2:
        return True
    return False


def get_distance_matrix(image_paths, samples):
    
    distance_matrix = euclidean_distances(samples, samples)
    largest_float = np.finfo(distance_matrix.dtype).max
  
    file_names_split = [get_infos_for_constraints(path) for path in image_paths]
    n = len(image_paths)
    
    same_gorilla_mask = np.array([
        [is_same_gorilla(file_names_split[i], file_names_split[j]) for j in range(n)]
        for i in range(n)
    ])
    different_gorilla_mask = np.array([
        [is_different_gorilla(file_names_split[i], file_names_split[j]) for j in range(n)]
        for i in range(n)
    ])
    
    distance_matrix[same_gorilla_mask] = 0
    distance_matrix[different_gorilla_mask] = largest_float

    return distance_matrix