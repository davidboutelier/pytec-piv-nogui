"""
This is a collection of functions for PIV steps
"""

def create_proj_fn(source_path, all_projects_path, exp_name, time_interval, time_unit):
    """
    This function creates a metadata file with the parameters of the project.
    It is the first function to be called.
    """
    import os
    import json
    from util_fn import dprint
    from datetime import datetime

    dprint('INITIATION')
    # create project time stamp
    project_create_time = str(datetime.now())
    dprint(project_create_time)

    # make a copy of the name
    root_exp_name = exp_name

    # create a path name
    this_project_path = all_projects_path + '/' + exp_name

    # check if it exists already
    test = os.path.exists(this_project_path)

    # if it exists add a number to it
    if test:
        counter = 1
        while test:
            exp_name = root_exp_name + '_' + str(counter)
            new_project_path = this_project_path + '_' + str(counter)
            test = os.path.exists(new_project_path)
            counter += 1

    else:
        new_project_path = this_project_path

    # create the directory for this project in the PIV project directory
    os.mkdir(new_project_path)
    dprint('creating new project folder ' + new_project_path)

    # assemble a list of parameters to be saved in json
    project_data = {}

    project_data['source'] = []
    project_data['source'].append({
        'source_path': source_path,
        'exp_name': root_exp_name
    })

    project_data['project'] = []
    project_data['project'].append({
        'project_path': all_projects_path,
        'project_name': exp_name,
        'project_create_time': project_create_time,
        'project_time_interval': time_interval,
        'project_time_interval_unit': time_unit
    })

    # save data in json file in the project
    with open(new_project_path + '/project_metadata.json', 'w') as outfile:
        json.dump(project_data, outfile)

    # save data in json file in sources
    with open('project_metadata.json', 'w') as outfile:
        json.dump(project_data, outfile)


def convert_dng_image(frame_num, file, dir_out):
    """
    This function converts a dng to 16-bit tiff file using rawpy and saves it in designated directory.
    """
    import os
    import rawpy
    import numpy as np
    import warnings
    from skimage import io, img_as_uint
    from util_fn import dprint, rgb2gray

    with rawpy.imread(file) as raw:
        rgb = raw.postprocess()
        grayscale_image = rgb2gray(rgb)
        grayscale_image_max = np.max(grayscale_image.flatten())
        grayscale_image_min = np.min(grayscale_image.flatten())
        grayscale_image = (grayscale_image - grayscale_image_min) / (grayscale_image_max - grayscale_image_min)
        warnings.filterwarnings("ignore", category=UserWarning)
        bit_16_grayscale_image = img_as_uint(grayscale_image)
        io.imsave(os.path.join(dir_out, 'IMG_' + str(frame_num + 1) + '.tif'), bit_16_grayscale_image)
        dprint('- image ' + file + ' imported')


def import_images_fn(fraction_cores):
    """
    This function import the images using the paths stored in the metadata file.
    The files are imported in parallel. The variable fraction_cores defines how many of
    the available cores to use for import.
    """
    import time
    import multiprocessing
    import os
    import json
    from util_fn import dprint
    from joblib import Parallel, delayed

    dprint(' ')
    dprint('IMPORT')

    # for measuring CPU time
    start_time = time.time()

    # defines the number of cores for parallel processing
    num_cores = multiprocessing.cpu_count()
    dprint(str(num_cores) + ' cores available on this computer')
    use_cores = int(fraction_cores * num_cores)
    dprint('using ' + str(use_cores) + ' cores')

    # read the data json
    with open('project_metadata.json') as f:
        project_data = json.load(f)

    project_source = project_data['source']
    project_source = project_source[0]
    source_path = project_source['source_path']
    exp_name = project_source['exp_name']

    project = project_data['project']
    project = project[0]
    project_path = project['project_path']
    project_name = project['project_name']
    project_create_time = project['project_create_time']

    # import the calibration images
    dprint('importing calibration images')
    source_path_calib_img = os.path.join(source_path, exp_name, 'CALIB')
    project_path_calib_img = os.path.join(project_path, project_name, 'CALIB')

    t = os.path.exists(project_path_calib_img)

    if t:
        dprint('saving in directory: ' + project_path_calib_img)
    else:
        os.mkdir(project_path_calib_img)

    list_img = os.listdir(source_path_calib_img)
    n_calib_img = len(list_img)

    Parallel(n_jobs=use_cores)(delayed(convert_dng_image)(frame_num,
                                                          os.path.join(source_path_calib_img, list_img[frame_num]),
                                                          project_path_calib_img) for frame_num in range(0, n_calib_img))

    # now import the experiment images
    dprint('importing experiment images')
    source_path_exp_img = os.path.join(source_path, exp_name, 'EXP')
    project_path_exp_img = os.path.join(project_path, project_name, 'EXP')

    t = os.path.exists(project_path_exp_img)

    if t:
        dprint('saving files in existing directory: ' + project_path_exp_img)
    else:
        os.mkdir(project_path_exp_img)

    list_img = os.listdir(source_path_exp_img)  # dir is your directory path
    n_exp_img = len(list_img)

    Parallel(n_jobs=use_cores)(delayed(convert_dng_image)(frame_num,
                                                          os.path.join(source_path_exp_img, list_img[frame_num]),
                                                          project_path_exp_img) for frame_num in range(0, n_exp_img))

    # save the metadata
    project_data['source'].append({
        'number_calibration_images': n_calib_img,
        'number_experiment_images': n_exp_img
    })

    # save meta data in json file in the project
    with open(project_path + '/project_metadata.json', 'w') as outfile:
        json.dump(project_data, outfile)

    # and save metadata in json file in sources
    with open('project_metadata.json', 'w') as outfile:
        json.dump(project_data, outfile)

    end_time = time.time()  # for measuring CPU time
    elapsed_time = end_time - start_time
    dprint('import completed in ' + str(elapsed_time) + ' s')


def find_calibration_fn(fn_type, calib_poly_order, calib_board, calib_unit):
    """
    This is a collection of functions for PIV steps
    """
    import os
    import json
    from skimage import io, img_as_float, img_as_uint
    from skimage.feature import corner_harris, corner_subpix, corner_peaks
    from skimage import transform as tf
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle
    from skimage.transform import warp
    import warnings
    from util_fn import dprint

    dprint(' ')
    dprint('CALIBRATION')

    # define calibration board
    nx = calib_board[0]
    ny = calib_board[1]
    dxc = calib_board[2]
    dyc = calib_board[2]
    calib_physical_unit = calib_unit

    # read the data json
    with open('project_metadata.json') as f:
        project_data = json.load(f)

    project_source = project_data['source']
    project_source = project_source[0]
    source_path = project_source['source_path']
    exp_name = project_source['exp_name']

    project = project_data['project']
    project = project[0]
    project_path = project['project_path']
    project_name = project['project_name']

    # load the first calibration image
    img = io.imread(os.path.join(project_path, project_name, 'CALIB', 'IMG_1.tif'))

    # show the figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.show()
    ax.imshow(img, cmap='gray')
    fig.canvas.draw()

    # user provides the 4 corners points
    points = plt.ginput(4)
    points = np.asarray(points)

    LX = 0.7 * (0.5 * ((points[1, 0] - points[0, 0]) + (points[2, 0] - points[3, 0]))) / nx
    LY = 0.7 * (0.5 * ((points[0, 1] - points[3, 1]) + (points[1, 1] - points[2, 1]))) / ny

    AR = (0.5 * ((points[1, 0] - points[0, 0]) + (points[2, 0] - points[3, 0]))) / nx \
         + (0.5 * ((points[0, 1] - points[3, 1]) + (points[1, 1] - points[2, 1]))) / ny

    AR = 0.5 * AR / dxc

    dprint('resolution is: '+"{:2.2f}".format(AR)+' pix/'+calib_physical_unit)

    corrected_corners = np.zeros(points.shape)

    for i in range(0, 4):

        x = points[i, 0]
        y = points[i, 1]
        dx = LX / 2
        dy = LY / 2
        [xp, yp] = subimage_CP(img, x, y, dx, dy)

        corrected_corners[i, :] = [xp, yp]
        ax.plot(corrected_corners[:, 0], corrected_corners[:, 1], '+r', markeredgewidth=2, markersize=15)

    # find the arithmetic mean of the four corners
    xm = int(0.25 * (corrected_corners[0, 0]
                     + corrected_corners[1, 0]
                     + corrected_corners[2, 0]
                     + corrected_corners[3, 0]))
    ym = int(0.25 * (corrected_corners[0, 1]
                     + corrected_corners[1, 1]
                     + corrected_corners[2, 1]
                     + corrected_corners[3, 1]))

    ax.plot(xm, ym, '+', color='yellow', fillstyle='none', markeredgewidth=2, markersize=15)
    ax.plot(xm, ym, 'x', color='yellow', fillstyle='none', markeredgewidth=2, markersize=15)
    fig.canvas.draw()

    # define the target position of the 4 corner points relative to the mean
    W = (nx - 1) * dxc
    H = (ny - 1) * dyc
    p0 = [xm - AR * W / 2, ym + AR * H / 2]
    p1 = [xm + AR * W / 2, ym + AR * H / 2]
    p2 = [xm + AR * W / 2, ym - AR * H / 2]
    p3 = [xm - AR * W / 2, ym - AR * H / 2]
    dst = np.asarray([p0, p1, p2, p3])

    fig.savefig("CP.pdf", bbox_inches='tight')
    plt.pause(2)
    plt.close(fig)

    if fn_type in ('PROJ', 'PROJ+POLY'):
        dprint('using projective calibration function')

        tform_proj = tf.estimate_transform('projective', dst, corrected_corners)
        f = open(os.path.join(project_path, project_name, 'CALIB', 'calibration_proj.pckl'), 'wb')
        pickle.dump(tform_proj, f)
        f.close()
        dprint('calibration function saved as :' + os.path.join(project_path, project_name, 'CALIB',
                                                                  'calibration_proj.pckl'))

        # correct the calibration image
        img_warped_proj = warp(img, tform_proj)
        warnings.filterwarnings("ignore", category=UserWarning)
        img_warped_proj2 = img_as_uint(img_warped_proj)

        # Create figure and axes
        fig2, ax2 = plt.subplots(1, figsize=(12, 8))
        ax2.imshow(img_warped_proj2, cmap='gray')
        fig2.canvas.draw()

        LXB_Proj = p1[0] - p0[0]
        LYB_Proj = p1[1] - p2[1]

        DLX_Proj = LXB_Proj / nx
        DLY_Proj = LYB_Proj / ny

        X_PTS_Proj = p3[0] + np.linspace(0, nx, nx + 1) * DLX_Proj
        Y_PTS_Proj = p3[1] + np.linspace(0, ny, ny + 1) * DLY_Proj

        X_PTS_Proj, Y_PTS_Proj = np.meshgrid(X_PTS_Proj, Y_PTS_Proj)  # where points should be
        X_PTS_Proj_Me = np.zeros(X_PTS_Proj.shape)  # where points are after proj
        Y_PTS_Proj_Me = np.zeros(Y_PTS_Proj.shape)

        for j in range(0, len(X_PTS_Proj[0, :])):
            for i in range(0, len(Y_PTS_Proj[:, 0])):

                x = X_PTS_Proj[i, j]
                y = Y_PTS_Proj[i, j]
                dx = DLX_Proj / 2
                dy = DLY_Proj / 2

                [xp, yp] = subimage_CP(img_warped_proj2, x, y, dx, dy)

                X_PTS_Proj_Me[i, j] = xp
                Y_PTS_Proj_Me[i, j] = yp

        ax2.plot(X_PTS_Proj_Me, Y_PTS_Proj_Me, '+', color='blue', fillstyle='none', markeredgewidth=2, markersize=10)
        fig2.canvas.draw()
        fig2.savefig("CP-1.pdf", bbox_inches='tight')
        plt.pause(2)
        plt.close()

        PTS = [X_PTS_Proj_Me, Y_PTS_Proj_Me]
        f = open(os.path.join(project_path, project_name, 'CALIB', 'PTS_proj.pckl'), 'wb')
        pickle.dump(PTS, f)
        f.close()

        margin_percent = 4
        img_warped_proj3 = crop_to_calib(PTS, img_warped_proj2, margin_percent)
        size_warped = img_warped_proj3.shape
        io.imsave(os.path.join(project_path, project_name, 'CALIB', 'IMG_PROJ_1.tif'), img_warped_proj3)

        [mean_ex, mean_ey, s_x, s_y] = correction_error(X_PTS_Proj, Y_PTS_Proj, X_PTS_Proj_Me, Y_PTS_Proj_Me)

        dprint('results of projective calibration')
        dprint('error on x position: ' + "{:2.2f}".format(mean_ex) + ' +- ' + "{:2.2f}".format(s_x) + ' pixels')
        dprint('error on y position: ' + "{:2.2f}".format(mean_ey) + ' +- ' + "{:2.2f}".format(s_y) + ' pixels')
        dprint('calibration function saved')

        if fn_type == 'PROJ+POLY':
            dprint(' ')
            dprint('searching for polynomial transformation function')

            Meas_pts = np.zeros((len(X_PTS_Proj.flatten()), 2))
            Target_pts = np.zeros((len(X_PTS_Proj.flatten()), 2))

            for k in range(0, len(X_PTS_Proj.flatten())):
                Meas_pts[k, 0] = X_PTS_Proj_Me.flatten()[k]
                Meas_pts[k, 1] = Y_PTS_Proj_Me.flatten()[k]

                Target_pts[k, 0] = X_PTS_Proj.flatten()[k]
                Target_pts[k, 1] = Y_PTS_Proj.flatten()[k]

            tform_poly = tf.estimate_transform('polynomial', Target_pts, Meas_pts, order=calib_poly_order)
            f = open(os.path.join(project_path, project_name, 'CALIB', 'calibration_proj_poly.pckl'), 'wb')
            pickle.dump(tform_poly, f)
            f.close()
            dprint('calibration function saved as :' + os.path.join(project_path, project_name, 'CALIB',
                                                                       'calibration_proj_poly.pckl'))
            # correct the calibration image
            img_warped_proj4 = warp(img_warped_proj2, tform_poly)
            warnings.filterwarnings("ignore", category=UserWarning)
            img_warped_proj5 = img_as_uint(img_warped_proj4)

            # Create figure and axes
            fig2, ax2 = plt.subplots(1, figsize=(12, 8))
            ax2.imshow(img_warped_proj5, cmap='gray')
            fig2.canvas.draw()

            X_PTS_Proj_Me2 = np.zeros(X_PTS_Proj.shape)  # where points are after proj
            Y_PTS_Proj_Me2 = np.zeros(Y_PTS_Proj.shape)

            for j in range(0, len(X_PTS_Proj[0, :])):
                for i in range(0, len(Y_PTS_Proj[:, 0])):
                    xp = X_PTS_Proj[i, j]
                    yp = Y_PTS_Proj[i, j]
                    dx = DLX_Proj / 2
                    dy = DLY_Proj / 2
                    [xp, yp] = subimage_CP(img_warped_proj5, xp, yp, dx, dy)

                    X_PTS_Proj_Me2[i, j] = xp
                    Y_PTS_Proj_Me2[i, j] = yp


            ax2.plot(X_PTS_Proj_Me2, Y_PTS_Proj_Me2, '+', color='blue', fillstyle='none', markeredgewidth=2,
                     markersize=10)
            fig2.canvas.draw()
            fig2.savefig("CP-2.pdf", bbox_inches='tight')
            plt.pause(2)
            plt.close()

            PTS = [X_PTS_Proj_Me2, Y_PTS_Proj_Me2]
            f = open(os.path.join(project_path, project_name, 'CALIB', 'PTS_proj_poly.pckl'), 'wb')
            pickle.dump(PTS, f)
            f.close()

            margin_percent = 4
            img_warped_proj6 = crop_to_calib(PTS, img_warped_proj5, margin_percent)

            size_warped = img_warped_proj6.shape

            io.imsave(os.path.join(project_path, project_name, 'CALIB', 'IMG_PROJ_POLY_1.tif'), img_warped_proj6)

            [mean_ex2, mean_ey2, s_x2, s_y2] = correction_error(X_PTS_Proj, Y_PTS_Proj, X_PTS_Proj_Me2, Y_PTS_Proj_Me2)
            dprint('results of polynomial after projective calibration')
            dprint('error on x position: ' + "{:2.2f}".format(mean_ex2) + ' +- ' + "{:2.2f}".format(s_x2) + ' pixels')
            dprint('error on y position: ' + "{:2.2f}".format(mean_ey2) + ' +- ' + "{:2.2f}".format(s_y2) + ' pixels')
            dprint('calibration function saved')
            project_data['calibration'] = []
            project_data['calibration'].append({
                'method': fn_type,
                'poly_order': calib_poly_order,
                'resolution': AR,
                'resolution_unit': 'pix/' + calib_physical_unit,
                'physical_unit': calib_physical_unit,
                'mean_error_x': mean_ex2,
                'mean_error_y': mean_ey2,
                '2_std_error_x': s_x2,
                '2_std_error_y': s_y2,
                'im_width': size_warped[0],
                'im_height': size_warped[1]
            })

        else:
            project_data['calibration'] = []
            project_data['calibration'].append({
                'method': fn_type,
                'poly_order': 1,
                'resolution': AR,
                'resolution_unit': 'pix/' + calib_physical_unit,
                'physical_unit': calib_physical_unit,
                'mean_error_x': mean_ex,
                'mean_error_y': mean_ey,
                '2_std_error_x': s_x,
                '2_std_error_y': s_y,
                'im_width': size_warped[0],
                'im_height': size_warped[1]
            })

    else:
        dprint(' direct polynomial transformation')

    # save data in json file in the project
    with open(project_path + '/project_metadata.json', 'w') as outfile:
        json.dump(project_data, outfile)

    # save data in json file in sources
    with open('project_metadata.json', 'w') as outfile:
        json.dump(project_data, outfile)


def subimage_CP(img, x, y , dx, dy):
    import numpy as np
    from skimage import io, img_as_float, img_as_uint
    from skimage.feature import corner_harris, corner_subpix, corner_peaks

    S = img.shape

    y1 = int(y - dy)
    y2 = int(y + dy)

    x1 = int(x - dx)
    x2 = int(x + dx)

    if y1 < 0:
        y1 = 0

    if y2 > S[0]:
        y2 = S[0]

    if x1 < 0:
        x1 = 0

    if x2 > S[1]:
        x2 = S[1]

    cropped = img[y1:y2, x1:x2]
    cropped2 = img_as_float(cropped)
    cropped2 = (cropped2 - min(cropped2.flatten())) / (max(cropped2.flatten()) - min(cropped2.flatten()))
    cropped2[cropped2 >= 0.5] = 1
    cropped2[cropped2 < 0.5] = 0

    coords = corner_peaks(corner_harris(cropped2, method='k', k=0.2, eps=1e-06, sigma=1), min_distance=5,
                          threshold_rel=0, num_peaks=1)

    coords_subpix = corner_subpix(cropped2, coords, window_size=13)

    if len(coords_subpix) == 0:
        xp = x - dx
        yp = y - dy
    else:
        xp = x - dx + coords_subpix[0, 1]
        yp = y - dy + coords_subpix[0, 0]

    return xp, yp


def crop_to_calib(PTS, img, overlap_percent):

    X_PTS = PTS[0]
    Y_PTS = PTS[1]

    min_x = min(X_PTS.flatten())
    max_x = max(X_PTS.flatten())

    min_y = min(Y_PTS.flatten())
    max_y = max(Y_PTS.flatten())

    calib_w = max_x - min_x
    calib_h = max_y - min_y

    offset_x = int(overlap_percent * calib_w / 100)
    offset_y = int(overlap_percent * calib_h / 100)

    y_crop_min = int(min_y - offset_y)
    y_crop_max = int(max_y + offset_y)

    x_crop_min = int(min_x - offset_x)
    x_crop_max = int(max_x + offset_x)

    img_cropped = img[y_crop_min:y_crop_max, x_crop_min:x_crop_max]

    return img_cropped


def correction_error(X_PTS_target, Y_PTS_target, X_PTS_Mes, Y_PTS_Mes):
    import numpy as np

    DX = np.abs(X_PTS_target - X_PTS_Mes)
    DY = np.abs(Y_PTS_target - Y_PTS_Mes)

    mean_ex = np.mean(DX)
    mean_ey = np.mean(DY)
    std_ex = np.std(DX)
    std_ey = np.std(DY)

    s_x = 2 * std_ex
    s_y = 2 * std_ey
    return mean_ex, mean_ey, s_x, s_y


def correct_image(project_path, project_name, frame_num, cor_method, cor_order, rotation_angle):

    from skimage import io, img_as_uint, img_as_float
    import pickle
    from skimage.transform import warp, rotate
    import warnings
    import os
    from util_fn import dprint

    if cor_method == 'PROJ':

        # load function
        f = open(os.path.join(project_path, project_name, 'CALIB', 'calibration_proj.pckl'), 'rb')
        tform_proj = pickle.load(f)
        f.close()

        # load image
        img = io.imread(os.path.join(project_path, project_name, 'EXP', 'IMG_'+str(frame_num+1)+'.tif'))

        # correct image
        img_warped_proj = warp(img, tform_proj)

        # crop image
        margin_percent = 4
        f = open(os.path.join(project_path, project_name, 'CALIB', 'PTS_proj.pckl'), 'rb')
        PTS = pickle.load(f)
        f.close()

        img_warped_proj2 = crop_to_calib(PTS, img_warped_proj, margin_percent)

        if rotation_angle != 0:
            img_warped_proj3 = rotate(img_warped_proj2, rotation_angle,
                                      resize=True, center=None, order=1,
                                      mode='constant', cval=0, clip=True, preserve_range=False)
        else:
            img_warped_proj3 = img_warped_proj2

        # convert to 16-bit
        warnings.filterwarnings("ignore", category=UserWarning)
        img_warped_proj4 = img_as_uint(img_warped_proj3)

        # save as tif
        io.imsave(os.path.join(project_path, project_name,
                               'EXP', 'CORRECTED', 'IMG_'+str(frame_num+1)+'.tif'), img_warped_proj4)
        dprint('- image '+os.path.join(project_path, project_name,
                               'EXP', 'IMG_'+str(frame_num+1)+'.tif')+' corrected')

    elif cor_method == 'PROJ+POLY':
        # load projective function
        f = open(os.path.join(project_path, project_name, 'CALIB', 'calibration_proj.pckl'), 'rb')
        tform_proj = pickle.load(f)
        f.close()

        # load polynomial function
        f = open(os.path.join(project_path, project_name, 'CALIB', 'calibration_proj_poly.pckl'), 'rb')
        tform_poly = pickle.load(f)
        f.close()

        # load image
        img = io.imread(os.path.join(project_path, project_name, 'EXP', 'IMG_' + str(frame_num + 1) + '.tif'))

        # correct image
        img_warped_proj = warp(img, tform_proj)
        img_warped_proj_poly = warp(img_warped_proj, tform_poly)

        # crop image
        margin_percent = 4
        f = open(os.path.join(project_path, project_name, 'CALIB', 'PTS_proj_poly.pckl'), 'rb')
        PTS = pickle.load(f)
        f.close()

        img_warped_proj_poly2 = crop_to_calib(PTS, img_warped_proj_poly, margin_percent)

        if rotation_angle != 0:
            img_warped_proj_poly3 = rotate(img_warped_proj_poly2, rotation_angle,
                                           resize=True, center=None, order=1,
                                           mode='constant', cval=0, clip=True, preserve_range=False)
        else:
            img_warped_proj_poly3 = img_warped_proj_poly2

        warnings.filterwarnings("ignore", category=UserWarning)
        img_warped_proj_poly4 = img_as_uint(img_warped_proj_poly3)

        io.imsave(os.path.join(project_path, project_name,
                               'EXP', 'CORRECTED', 'IMG_' + str(frame_num + 1) + '.tif'), img_warped_proj_poly4)
        dprint('- image ' + os.path.join(project_path, project_name,
                                         'EXP', 'IMG_' + str(frame_num + 1) + '.tif') + ' corrected')

    else:
        dprint(' not implemented yet')


def apply_correction(fraction_cores, rotation_angle):
    from util_fn import dprint
    import time
    import multiprocessing
    import os
    import json
    from joblib import Parallel, delayed

    dprint(' ')
    dprint('IMAGE CORRECTION')

    # for measuring CPU time
    start_time = time.time()

    # defines the number of cores for parallel processing
    num_cores = multiprocessing.cpu_count()
    use_cores = int(fraction_cores * num_cores)
    dprint('using ' + str(use_cores) + ' cores')

    # read the data json
    with open('project_metadata.json') as f:
        project_data = json.load(f)

    project_source = project_data['source']
    project_source = project_source[0]
    source_path = project_source['source_path']
    exp_name = project_source['exp_name']

    project = project_data['project']
    project = project[0]
    project_path = project['project_path']
    project_name = project['project_name']

    project_calib = project_data['calibration']
    project_calib = project_calib[0]
    cor_method = project_calib['method']
    cor_order = project_calib['poly_order']

    # create new directory
    project_path_exp_img = os.path.join(project_path, project_name, 'EXP')
    project_path_exp_img_cor = os.path.join(project_path, project_name, 'EXP', 'CORRECTED')

    t = os.path.exists(project_path_exp_img_cor)

    if t:
        dprint('saving files in existing directory: ' + project_path_exp_img_cor)
    else:
        os.mkdir(project_path_exp_img_cor)

    list_img = os.listdir(project_path_exp_img)
    n_exp_img = len(list_img)

    Parallel(n_jobs=use_cores)(delayed(correct_image)(project_path,
                                                      project_name,
                                                      frame_num,
                                                      cor_method, cor_order, rotation_angle) for frame_num in range(0, n_exp_img-1))

    end_time = time.time()  # for measuring CPU time
    elapsed_time = end_time - start_time
    dprint('correction completed in ' + str(elapsed_time) + ' s')


def define_roi():
    import json
    import os
    import numpy as np
    from skimage import io
    import matplotlib.pyplot as plt

    # read the data json
    with open('project_metadata.json') as f:
        project_data = json.load(f)

    project = project_data['project']
    project = project[0]
    project_path = project['project_path']
    project_name = project['project_name']

    # load the first calibration image
    img = io.imread(os.path.join(project_path, project_name, 'EXP', 'CORRECTED', 'IMG_1.tif'))

    # show the figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fig.show()
    ax.imshow(img, cmap='gray')
    fig.canvas.draw()

    # user provides the 2 opposite corners points of ROI
    points = plt.ginput(2)
    points = np.asarray(points)

    le_roi = points[1, 0] - points[0, 0]
    he_roi = points[1, 1] - points[0, 1]

    rectangle = plt.Rectangle((points[0, 0], points[0, 1]), le_roi, he_roi, linewidth=1,edgecolor='r',facecolor='none')

    roi = [points[0, 0], points[0, 1], le_roi, he_roi]
    ax.add_patch(rectangle)
    fig.canvas.draw()
    plt.pause(1)
    plt.close()
    return roi


def split_2frames(iw, roi):
    """docstring"""
    import numpy as np

    roi_xmin = roi[0]  # ROI X position
    roi_ymin = roi[1]  # ROI Y position
    roi_w = roi[2]  # ROI width
    roi_h = roi[3]  # ROI height

    step = 0.5 * iw
    n_iw_x = np.floor((roi_w - 2) / step - 1)  # number of iw along X
    n_iw_y = np.floor((roi_h - 2) / step - 1)  # number of iw along Y
    n_iw = np.int(n_iw_x * n_iw_y)  # number of IW

    GX = roi_xmin + np.int(0.5 * iw + np.floor(0.5 * (roi_w - (n_iw_x + 1) * step))) + np.linspace(0, int(n_iw_x - 1), int(n_iw_x)) * step
    GY = roi_ymin + np.int(0.5 * iw + np.floor(0.5 * (roi_h - (n_iw_y + 1) * step))) + np.linspace(0, int(n_iw_y - 1), int(n_iw_y)) * step
    GX2, GY2 = np.meshgrid(GX, GY)

    grid_x_flat = GX2.flatten()
    grid_y_flat = GY2.flatten()

    return grid_x_flat, grid_y_flat, n_iw_x, n_iw_y, n_iw


def cor_2sub(i , f1, f2):
    """docstring"""
    import numpy as np
    from scipy.signal import fftconvolve
    C_sub = fftconvolve(f1, f2, mode='same')

    return C_sub


def cor_2frames(f1, f2, iw, grid_x_flat, grid_y_flat, n_iw):
    """docstring"""
    import numpy as np
    import sys
    from scipy.signal import fftconvolve

    c_map = np.zeros((iw, iw, n_iw))
    stack1 = np.zeros((iw, iw, n_iw))
    stack2 = np.zeros((iw, iw, n_iw))

    for i in range(0, n_iw):

        x1_sub = np.int(grid_x_flat[i] - iw / 2)
        y1_sub = np.int(grid_y_flat[i] - iw / 2)

        x2_sub = np.int(grid_x_flat[i] + iw / 2)
        y2_sub = np.int(grid_y_flat[i] + iw / 2)

        stack1[:, :, i] = f1[y1_sub:y2_sub, x1_sub:x2_sub]
        stack2[:, :, i] = f2[y1_sub:y2_sub, x1_sub:x2_sub]

        c_map[:, :, i] = fftconvolve(stack1[:, :, i], stack2[::-1, ::-1, i], mode='same')

        bar_length = 20
        end_val = n_iw

        '''
        percent = float(i) / end_val
        hashes = '#' * int(round(percent * bar_length))
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
        sys.stdout.flush()
        '''

    #sys.stdout.write("\n")

    CF = c_map.reshape(-1, c_map.shape[-1])
    P = np.max(CF, axis=0)  # peak value
    PP = np.argmax(CF, axis=0)  # peak position in flattened IW
    PP2 = np.asarray(np.unravel_index(PP, (iw, iw)))  # peak position in un_flattened
    M = np.mean(CF, axis=0)
    S = np.std(CF, axis=0)
    SNR = (P - M) / S

    return [c_map, P, PP, PP2, M, S, SNR]


def cor_sub_pix(c_map, peak_g, n_iw):
    """docstring"""
    import numpy as np

    c10 = np.zeros((n_iw, 1))
    c01 = np.zeros((n_iw, 1))
    c11 = np.zeros((n_iw, 1))
    c20 = np.zeros((n_iw, 1))
    c02 = np.zeros((n_iw, 1))

    for k in range(0, n_iw):
        for i in range(-1, 2):
            for j in range(-1, 2):
                c10[k] = c10[k] + i * np.log(c_map[peak_g[0, k] + j, peak_g[1, k] + i, k])
                c01[k] = c01[k] + j * np.log(c_map[peak_g[0, k] + j, peak_g[1, k] + i, k])
                c11[k] = c11[k] + i * j * np.log(c_map[peak_g[0, k] + j, peak_g[1, k] + j, k])
                c20[k] = c20[k] + (3 * i * i - 2) * np.log(c_map[peak_g[0, k] + j, peak_g[1, k] + i, k])
                c02[k] = c02[k] + (3 * j * j - 2) * np.log(c_map[peak_g[0, k] + j, peak_g[1, k] + i, k])

    c10 = c10 / 6
    c01 = c01 / 6
    c11 = c11 / 4
    c20 = c20 / 6
    c02 = c02 / 6

    sub_x = (c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11 * c11)
    sub_y = (c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11 * c11)

    return [sub_x, sub_y]


def define_mask(roi, iw, n_iw_x, n_iw_y, n_iw):
    """docstring"""
    import numpy as np
    roi_xmin = roi[0]  # ROI X position
    roi_ymin = roi[1]  # ROI Y position
    roi_w = roi[2]  # ROI width
    roi_h = roi[3]  # ROI height

    q_mask = np.zeros((4, 2))
    q_mask[0, 0] = roi_xmin
    q_mask[0, 1] = roi_ymin

    q_mask[1, 0] = roi_xmin
    q_mask[1, 1] = roi_ymin + roi_h

    q_mask[2, 0] = roi_xmin + roi_w
    q_mask[2, 1] = roi_ymin + roi_h

    q_mask[3, 0] = roi_xmin + roi_w
    q_mask[3, 1] = roi_ymin

    mask = np.zeros((int(2*(n_iw_x+2) + 2*n_iw_y), 3))

    print(n_iw_x, n_iw_y, n_iw_x*n_iw_y)

    for i in range(0, int(n_iw_y+1)):
        mask[i, 0] = i+1
        mask[i, 1] = q_mask[0, 0]
        mask[i, 2] = q_mask[0, 1] + i*iw

    for i in range(0, int(n_iw_x+1)):
        mask[n_iw_y+1 + i, 0] = n_iw_y+1 + i +1
        mask[n_iw_y+1 + i, 1] = q_mask[1,0] + i*iw

    return mask

def pre_proc(img, gaussian_sigma, kernel_normal):
    from skimage import util
    from skimage import filters
    from skimage import exposure
    from skimage import img_as_float
    import numpy as np
    import warnings

    # convert to 16-bit
    warnings.filterwarnings("ignore", category=UserWarning)
    img_f = img_as_float(img)

    # equalize histogram
    img_max = np.max(img_f.flatten())
    img_min = np.min(img_f.flatten())
    img2 = (img - img_min) / (img_max - img_min)

    # invert image
    img3 = util.invert(img2)

    # subtract gaussian blur
    background = filters.gaussian(img3, sigma=gaussian_sigma, preserve_range=True)
    img4 = img3 - background

    # Adaptive Equalization
    img5 = exposure.equalize_adapthist(img4, kernel_size=kernel_normal, clip_limit=0.01, nbins=256)

    return img5












