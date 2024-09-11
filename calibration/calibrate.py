import jsonpickle
import numpy as np
import sys
try:
    with open('calibration/calibration_data.json', 'r') as calibartion_file:
        string = calibartion_file.read()
except FileNotFoundError:
    print("calibration file not found, exiting")
    sys.exit()
calibrations = jsonpickle.decode(string)

cameraMatrix = np.array(calibrations['cameraMatrix'])
distCoeffs = np.array(calibrations['distCoeffs'])

print(cameraMatrix.shape)
print(distCoeffs.shape)