import os
import shutil

SOURCE = "ec_data_og"
DEST = "ec_data"
start_indices = [198, 330, 485, 165, 8]

dirs = os.listdir(SOURCE)

for i, curr_dir in enumerate(os.listdir(SOURCE)):
    img_dir = os.path.join(SOURCE, curr_dir, "images")
    dest_dir = os.path.join(DEST, curr_dir, "images")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for j, f_name in enumerate(os.listdir(img_dir)):
        curr_start = start_indices[i]

        if curr_start <= j <= (curr_start + 80):
            curr_file_path = os.path.join(img_dir, f_name)
            dest_file_path = os.path.join(dest_dir, f_name)
            shutil.copyfile(curr_file_path, dest_file_path)