"""
    Script that creates a deep learning dataset
"""

################################
##Input and Output Directories##
################################

out_path_base = f''
input_path_base=f''


#####################################################
##Get List of Valid Dates and Path of Summary Files##
#####################################################

emailer = Emailer()
start_time = emailer.get_start_time()

dates=[d for d in os.listdir(base_path) if '.txt' not in d]

paths=[] #Valid paths for worker function

for d in dates:
    if d[4:6] !='05': 
        continue

    times = [t for t in os.listdir(join(base_path, d)) if 'basemap' not in t] #Init time

    for t in times:
        path = join(base_path, d , t)

        files = glob(join(path, f'wofs_ENS_[2-7]*'))

        all_nc_files = len([f for f in files if f.endswith('.nc')])

        if all_nc_files == len(files):
            paths.append(path)

print(f'Number of valid paths: {len(paths)}')

emailer.send_email(f'Starting patch extraction', start_time)