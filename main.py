import pytec_fn
import os

## CREATE PROJECT
all_projects_path = '/media/dpb509/SSD/PROJECTS'
source_path = '/media/dpb509/SSD/SOURCES'
time_interval = 5
time_unit = 's'
exp_name = 'MH3'

t = os.path.isfile('log.txt')
if t:
    os.remove('log.txt')

pytec_fn.create_proj_fn(source_path, all_projects_path, exp_name, time_interval, time_unit)

## IMPORT
fraction_cores = 1
pytec_fn.import_images_fn(fraction_cores)

## CALIBRATION
calib_fn_type = 'PROJ+POLY'
calib_board = [30, 28, 15, 15]
calib_unit = 'mm'
calib_poly_order = 3

pytec_fn.find_calibration_fn(calib_fn_type, calib_poly_order, calib_board, calib_unit)

## CORRECTION
rotation_angle = -90
pytec_fn.apply_correction(fraction_cores, rotation_angle)

roi = pytec_fn.define_roi()
for f in range(0, 1):
    f1 = f
    f2 = f1 + 1