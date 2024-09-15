# import cudf.pandas
# cudf.pandas.install()
import os
# from torchvision.io import read_image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image


def dataset_data(img_path, folder_name, img_class):
    fname = os.listdir(img_path + "/"+folder_name)
    fname.sort()
    fpath = [img_path + "/"+folder_name+"/" + f for f in fname]
    # height = [read_image(f).size() for f in fpath]
    height = [Image.open(f).size[0] for f in fpath]
    width = [Image.open(f).size[1] for f in fpath]
    channels = [Image.open(f).mode for f in fpath]
    labels = [img_class]*len(fname)

    return fpath, height, width, channels, labels

# Define a search function


def search_string(s, search):
    return search in str(s).lower()

# ---------------------------------------------------------------


def onehot(label, classes=2):
    onehot_val = str(np.base_repr(2**label, 2)).zfill(classes)
    onehot_array = np.array([int(i)for i in [*onehot_val]])
    return onehot_array

# ---------------------------------------------------------------


def remove_fname_space(path):
    for filename in os.listdir(path):
        my_source = path + "/" + filename
        my_dest = path + "/" + filename.strip().replace(" ", "")
        os.rename(my_source, my_dest)

def ds_to_df(absolute_path,relative_paths,paths_classes,original_split,required_split='all'):
    
    if required_split == 'all':
        split_length=len(paths_classes)
        split_lst= range(split_length)
    else:
        split_lst = [i for i, x in enumerate(original_split) if x == required_split]
        split_length=len(split_lst)
        
    
    fpath = [""]*split_length
    height = [""]*split_length
    width = [""]*split_length
    channels = [""]*split_length
    labels = [""]*split_length

    fpath_total = []
    height_total = []
    width_total = []
    channels_total = []
    labels_total = []

    for i,j in enumerate(split_lst):
        fpath[i],  height[i], width[i], channels[i], labels[i] = dataset_data(
            absolute_path, relative_paths[j], paths_classes[j])
        fpath_total += fpath[i]
        height_total += height[i]
        width_total += width[i]
        channels_total += channels[i]
        labels_total += labels[i]

    feature_col = "Image_path"
    height_col = "Height"
    width_col = "Width"
    channels_col = "Channels"
    cat_class_col = "Class"
    num_class_col = "Class_Codes"
    onehot_class_col = "Class_Onehot"

    ds_dict = {
        feature_col: fpath_total,
        height_col: height_total,
        width_col: width_total,
        channels_col: channels_total,
        cat_class_col: labels_total}
    
    ds_df = pd.DataFrame(ds_dict)
    ds_df[cat_class_col] = ds_df[cat_class_col].astype('category')
    ds_df[num_class_col] = ds_df[cat_class_col].cat.codes
    ds_df[onehot_class_col] = ds_df[num_class_col].apply(lambda row: onehot(row))
    
    return ds_df


def dataset_to_df(absolute_path, relative_paths, paths_classes, original_split, re_split, train_ratio, val_ratio, test_ratio):
    classes = list(set(paths_classes))
    classes.sort(reverse=True)

    cat_class_col = "Class"

    df_all = ds_to_df(absolute_path,relative_paths,paths_classes,original_split)

    if re_split:
        # Split the validation and test
        val_rel_ratio = val_ratio/(val_ratio+test_ratio)

        df_train, df_tmp = train_test_split(
            df_all, train_size=train_ratio, stratify=df_all[[cat_class_col]])
        df_val, df_test = train_test_split(
            df_tmp, train_size=val_rel_ratio, stratify=df_tmp[[cat_class_col]])
    else:
        val_test = [x for i, x in enumerate(list(set(original_split))) if x != 'train']
        
        df_train = ds_to_df(absolute_path,relative_paths,paths_classes,original_split,"train")
        
        if len(val_test)==1:
            df_val = ds_to_df(absolute_path,relative_paths,paths_classes,original_split,val_test[0])
            df_test = ds_to_df(absolute_path,relative_paths,paths_classes,original_split,val_test[0])
        else:
            df_val = ds_to_df(absolute_path,relative_paths,paths_classes,original_split,"valid")
            df_test = ds_to_df(absolute_path,relative_paths,paths_classes,original_split,"test")
        
        # Shuffle the dataset
        df_test = df_test.sample(frac=1)
        # Reset the index of the shuffled DataFrame
        df_test = df_test.reset_index(drop=True)
        
    # Return the class statistics
    classes_stats = np.zeros((3, len(classes)), dtype=int)

    for i in range(len(classes)):
        # Qty of the class
        classes_stats[0, i] = len(df_train[df_train[cat_class_col] == classes[i]])
        classes_stats[1, i] = len(df_val[df_val[cat_class_col] == classes[i]])
        classes_stats[2, i] = len(df_test[df_test[cat_class_col] == classes[i]])

    classes_stats_df = pd.DataFrame(classes_stats.tolist(), columns=classes)
    classes_stats_df["Total"] = classes_stats_df.sum(axis=1)
    classes_stats_df.index = ['Training', 'Validation', 'Testing']
    classes_stats_df.loc["Row_Total"] = classes_stats_df.sum()

    return df_all, df_train, df_val, df_test, classes_stats_df


def search_df(ref_df, str):
    # Search for the string in all columns
    mask = ref_df.apply(lambda x: x.map(lambda s: search_string(s, str)))
    # Filter the DataFrame based on the mask
    filtered_df = ref_df.loc[mask.any(axis=1)]
    return filtered_df.index[0]
