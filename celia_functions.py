import os, re
import numpy as np
import pandas as pd
from PIL import Image
from scipy import interpolate as spinter, optimize as spopt
from delta_method import delta_method

############### FUNCTIONS FOR LISTS ###############

def flatten_list(bookshelf):
    papers= []
    bookshelf = list(bookshelf)
    for book in bookshelf:
        if type(book) == str: #strings are iterable but should not be flattened
            papers.append(book)
            continue
        try:
            book[0]
        except: #This is not a list! Add this item to the papers. #Also why does numpy throw an IndexError whereas python throws a TypeError?!?
            papers.append(book)
        else:
            papers += flatten_list(book) #This is a list! Its items must be seperated first, then it can be added to the papers.
    return papers

def transform_list(matrix):
    return [*zip(*matrix)] #IDK how this works. Ross did it for me.

############# FITTING AND SMOOTHING ################

def polynomial(x, *coeffs):
    x = np.array(x)
    y = np.sum([coeff*x**n for n, coeff in enumerate(coeffs)], axis= 0)
    return y

def inverse_square(x, k):
    x = np.array(x)
    y = k**2*(x+k)**-2
    return y

def gradient(x, m):
    x = np.array(x)
    return m*x +1

def roll_average(xs, ys, number_of_points= 100, window_width=1):
    xs = flatten_list(xs)
    ys = flatten_list(ys)

    points = np.linspace(np.nanmin(xs), np.nanmax(xs), number_of_points)
    average = []
    uncertainty = []
    
    for point in points:
        points_in_window = [ys[n] for n, x in enumerate(xs) if x-window_width < point and point < x+window_width] #create a list of points inside the window
        average = average + [np.nanmean(points_in_window)]
        uncertainty = uncertainty + [np.nanstd(points_in_window)]

    return points, average, uncertainty

def polynomial_fit(xs, ys, number_of_points= 100, x0= [0, 0], constraints= None, confidence= 0.1, model_function= polynomial, silent= False):
    xs, ys = flatten_list(xs), flatten_list(ys)
    is_valid = np.logical_and(~np.isnan(xs), ~np.isnan(ys))
    xs, ys = np.array(xs)[is_valid], np.array(ys)[is_valid] #remove nans
    
    if constraints == None: constraints = [np.full_like(x0, -np.inf, dtype= float), np.full_like(x0, +np.inf, dtype= float)]
    bounds = spopt.Bounds(*constraints)

    points = np.linspace(np.nanmin(xs), np.nanmax(xs), number_of_points)

    popt, pcov = spopt.curve_fit(model_function, xs, ys, p0= x0, bounds= bounds, nan_policy= "omit", method= "trf", loss= "cauchy") # loss = "linear" for least squares. "huber" "soft_l1" "cauchy" "arctan"
    delta = delta_method(pcov, popt, points, model_function, xs, ys, confidence)

    poly_fit = model_function(points, *popt)

    if not silent:
        print("Optimal coeffients:")
        print(*popt)

    return points, poly_fit, delta

################# FUNCTIONS FOR IMAGES #################

def autocrop(array, noise_level= 10, silent= True): #crops to rectangular section with many large values.
    row_s2n = np.std(array, axis= 1) # Try different ways of estimating the S2N
    row_threshold = np.percentile(row_s2n, noise_level)
    column_s2n = np.std(array, axis= 0)
    column_threshold = np.percentile(column_s2n, noise_level)
    first_row, last_row = int(np.argwhere(row_s2n > row_threshold)[0]), int(np.argwhere(row_s2n > row_threshold)[-1])
    first_column, last_column = int(np.argwhere(column_s2n > column_threshold)[0]), int(np.argwhere(column_s2n > column_threshold)[-1])

    return array[first_row:last_row, first_column:last_column]

def dead_pixel_filter(image, dead_pixels =1):
    # Remove the most anomalous 1% of pixels and replace with nearest neighbour.
    upper_percentile = 100 - dead_pixels/2
    lower_percentile = dead_pixels/2
    notdead = np.logical_and(image <= np.percentile(image,upper_percentile), image >= np.percentile(image,lower_percentile) )
    coords = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    coords = np.moveaxis(coords, 0, -1) #refromat the array such that we have pairs of coordinates. ie. [[0,0],[0,1],[0,2]] ect.
    nearest = spinter.NearestNDInterpolator(coords[notdead], image[notdead])
    image = nearest(coords[:,:,0],coords[:,:,1])

    return image

def estimate_xray_signal(image, cropping= [747,793,726,775], silent= False): #cropping = [ymin, ymax, xmin, xmax]

    if len(np.shape(image)) != 2:
        if not silent:
            print("WARNING! Xray image has the wrong number of dimentions. Expected 2. Got {0:d}".format(len(np.shape(image))))
        return image

    cropping_mask = np.full_like(image, False, dtype= bool)
    cropping_mask[cropping[0]:cropping[1],cropping[2]:cropping[3]] = True

    signal = np.median(image[cropping_mask]) #Find the median value inside the region illuminated by the pinhole
    noise = np.median(image[~cropping_mask]) #Find the median value in the dark, unilluminated region

    return signal -noise

############## FILE ORGANISATION FUNCTIONS ####################

def read_metadata(requested_data): #DO NOT OPEN MULTIPLE SHEETS AT ONCE. OPEN THE SHOT SHEET AND TARGET SHEET SEPERATELY!
    shot_list = ["SHOT #","Date","Time","Purpose","Target X","Target Y","Target Z","Wheel XPS","position on wheel","TARGET NUMBER","Laser Energy (before comp) [J]","Laser duration [fs]","Horiz ns/div","Trigger val [mV]","C1 [V/div]","C2 [V/div]","C3 [V/div]","C4 [V/div]","C1 att [dB]","C2 att [dB]","C3 att [dB]","C1 diag","C2 diag","C3 diag","C4 diag","Comments JLD scope","THz comments","Column2","X-ray comments","Column3","Column4","Column5"]
    target_list = ["Date","Number","Position in the box","Wheel","Position on the wheel","Rotation angle(TARGET)","XYZ (TARGET)","Thickness (um)","Length(mm)","Material","Type","Comments"]

    if requested_data[0] in shot_list : sheet, header_row = "Shot-list", 1 # weird python syntax
    if requested_data[0] in target_list : sheet, header_row = "Target list", 0

    dataframe = pd.read_excel(io= "organised_data\\Shots_Targets_Diagnostics.xlsx", # numpy is bad at reading excel files.
                              sheet_name= sheet,
                              header= header_row,
                              usecols= requested_data)
    
    dictionary = dataframe.to_dict(orient= "list")
    return dictionary

def filter_files(file_directories, file_names, regex_code):
    match = [bool(re.findall(regex_code, file_name)) for file_name in file_names]
    filtered_file_directories = file_directories[match]
    filtered_file_names = file_names[match]
    number_of_files = len(filtered_file_names)
    filtered_file_paths = [os.path.join(filtered_file_directories[n],filtered_file_names[n]) for n in range(number_of_files)]

    return filtered_file_paths, filtered_file_directories, filtered_file_names

def open_pyro_oscilloscope(requested_data, shots, file_directories, file_names, function= lambda x:x, silent= False):
    channel = "Ch1" if requested_data == "pyro_time" else requested_data
    regex_code = ".*SHOT("+'|'.join(shots)+")[^0-9]*[0-9]{3}[^0-9].*"+channel+".csv"
    filtered_file_paths, _, filtered_file_names = filter_files(file_directories, file_names, regex_code)
                            
    shot_no = [int(re.search(regex_code, filtered_file_name).group(1)) for filtered_file_name in filtered_file_names]
    pyro_data = [function(np.genfromtxt(filtered_file_path, delimiter=',', skip_header =0, usecols= 3 if requested_data == "pyro_time" else 4))
                 for filtered_file_path in filtered_file_paths]

    if not silent: # remember to do silent == True if you use read_data in a loop
        print("Found {:d} files:".format(len(filtered_file_names)))
        [print("   " + filtered_file_name) for filtered_file_name in filtered_file_names]

    return shot_no, pyro_data

def open_emp_oscilloscope(requested_data, shots, file_directories, file_names, function= lambda x:x, silent= False):
    emp_oscilloscope = ["time","bdot","tof","diode"]
    column = emp_oscilloscope.index(requested_data)
    regex_code = "(?i)s0*("+'|'.join(shots)+")[^0-9]*\.csv"
    filtered_file_paths, _, filtered_file_names = filter_files(file_directories, file_names, regex_code)

    shot_no = [int(re.search(regex_code, filtered_file_name).group(1)) for filtered_file_name in filtered_file_names]
    emp_data = [function(np.genfromtxt(filtered_file_path, delimiter=',', skip_header =18, usecols= column))
                for filtered_file_path in filtered_file_paths]

    if not silent: # remember to do silent == True if you use read_data in a loop
        print("Found {:d} files:".format(len(filtered_file_names)))
        [print("   " + filtered_file_name) for filtered_file_name in filtered_file_names]

    return shot_no, emp_data

def open_xray_photos(requested_data, shots, file_directories, file_names, function= lambda x:x, silent= False):
    raw = "-raw" if requested_data == "xray_raw" else ""
    regex_code = ".*[0-9]{2}" +raw+ " SHOT(" +'|'.join(shots)+ ").tif"
    filtered_file_paths, _, filtered_file_names = filter_files(file_directories, file_names, regex_code)

    shot_no = [int(re.search(regex_code, filtered_file_name).group(1)) for filtered_file_name in filtered_file_names]
    data = [function(np.array(Image.open(filtered_file_path))) for filtered_file_path in filtered_file_paths]

    if not silent: # remember to do silent == True if you use read_data in a loop
        print("Found {:d} files:".format(len(filtered_file_names)))
        [print("   " + filtered_file_name) for filtered_file_name in filtered_file_names]

    return shot_no, data

def open_emp_energy(shots, function= lambda x:x, silent= False):
    jean_lucs = pd.read_table("jean-lucs_emp_calculations.txt",
                            encoding= "utf_8",
                            encoding_errors= "ignore", # I don't know which encoding to use for '¡' so I will ignore it for now.
                            header= 0,
                            names= ["shot", "energy"],
                            dtype= {"shot": int,
                                    "energy": float},
                            decimal= ",") # continental format

    jean_lucs = jean_lucs[jean_lucs["shot"].astype(str).isin(shots)]
    shot_no = jean_lucs["shot"].to_list()
    data = jean_lucs["energy"].to_list()
    data = [function(reading) for reading in data]

    if not silent: # remember to do silent == True if you use read_data in a loop
        print("Found 1 file:")
        print("   jean-lucs_emp_calculations.txt")

    return shot_no, data

def read_diagnostic_data(requested_data, functions= None, silent= False): #Currently supports the xray cam, pyro oscilloscope and the emp oscilloscope. Doesn't support target photos, espec, focal spot cam and pyrocams. 
    data_path = "organised_data\\"
    dictionary = {"shot":[]}
    emp_oscilloscope = ["time","bdot","tof","diode"]
    pyro_oscilloscope = ["Ch1","Ch2","Ch3","Ch4","pyro_time"]
    xray_cam = ["xray", "xray_raw"]

    if functions == None:
        functions = {key: lambda x:x for key in requested_data.keys()}

    all_file_directories = [root for root, dirs, files in os.walk(data_path) for file in files]
    all_file_directories = np.array(all_file_directories)
    all_file_names = [file for root, dirs, files in os.walk(data_path) for file in files]
    all_file_names = np.array(all_file_names)
    requested_data = {ch: [str(int(shot)) for shot in shots] for ch, shots in requested_data.items()} # ensure that all shot numbers are strings of ints. floats will screw up regex.

    for request, shots in requested_data.items():
        if request in pyro_oscilloscope:
            new_shots, new_data = open_pyro_oscilloscope(request, shots, all_file_directories, all_file_names, function= functions[request], silent= silent)
        elif request in emp_oscilloscope:
            new_shots, new_data = open_emp_oscilloscope(request, shots, all_file_directories, all_file_names, function= functions[request], silent= silent)
        elif request in xray_cam:
            new_shots, new_data = open_xray_photos(request, shots, all_file_directories, all_file_names, function= functions[request], silent= silent) 
        elif request == "energy":
            new_shots, new_data = open_emp_energy(shots, function= functions[request], silent= silent)
        else:
            new_shots, new_data = [], []
            print("WARNING!! {0} is not a recognised data type.".format(request))

        old_shots = dictionary["shot"]
        union_shots = np.unique(old_shots + new_shots).tolist()

        dictionary = {diagnostic: [readings[old_shots.index(shot)] if shot in old_shots else np.nan for shot in union_shots] for diagnostic, readings in dictionary.items()} #rearrange the existing data to corrispond with its respective shot
        dictionary[request] = [new_data[new_shots.index(shot)] if shot in new_shots else np.nan for shot in union_shots] # Add the new data to it's respective shot.
        dictionary["shot"] = union_shots #add the new shots

    return dictionary