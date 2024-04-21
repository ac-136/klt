import os
import shutil

SOURCE = "ec_data_og"
DEST = "ec_data"
start_indices = [198, 330, 485, 165, 8]

dirs = os.listdir(SOURCE)

for i, curr_dir in enumerate(os.listdir(SOURCE)):
    img_dir = os.path.join(SOURCE, curr_dir, "images")
    img_dest_dir = os.path.join(DEST, curr_dir, "images")

    time_dir = os.path.join(SOURCE, curr_dir, "images.txt")
    time_dest_dir = os.path.join(DEST, curr_dir, "times.txt")

    # Save only specific images
    if not os.path.exists(img_dest_dir):
        os.makedirs(img_dest_dir)
    
    img_names_list = []
    for j, f_name in enumerate(os.listdir(img_dir)):
        curr_start = start_indices[i]

        if curr_start <= j <= (curr_start + 80):
            curr_file_path = os.path.join(img_dir, f_name)
            dest_file_path = os.path.join(img_dest_dir, f_name)
            shutil.copyfile(curr_file_path, dest_file_path)

            img_names_list.append("images/" + f_name)

    # Save the timestamps for those images
    with open(time_dir, 'r') as infile:
        lines = infile.readlines()

        for line in lines:
            vals = line.split()
            time = vals[0]
            img_name = vals[1]

            if img_name in img_names_list:
                with open(time_dest_dir, 'a') as outfile:
                    outfile.write(str(time) + '\n')
            