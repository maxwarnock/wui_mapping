'''*********************************************
author: Max Warnock, Quin Browder, Victoria Madden
Date: 5/3/2025
GEOG 4303 Final Project
WUI Mapping Module - Module for better function organization
*********************************************'''
import os
import arcpy
import arcpy.sa as sa
import numpy
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def promptFloat(message, default=None):
    '''Prompts until a valid float is entered. Blank input returns default, if given.'''
    while True:
        raw = input(message).strip()
        if raw == "" and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            hint = f" (or press Enter for {default})" if default is not None else ""
            print(f"Invalid number, please try again{hint}.")

def promptInt(message, default=None):
    '''Prompts until a valid whole number is entered. Blank input returns default, if given.'''
    while True:
        raw = input(message).strip()
        if raw == "" and default is not None:
            return default
        try:
            return int(raw)
        except ValueError:
            hint = f" (or press Enter for {default})" if default is not None else ""
            print(f"Invalid whole number, please try again{hint}.")

def promptYesNo(message):
    '''Prompts until the user enters yes or no (case-insensitive).'''
    while True:
        raw = input(message).strip().lower()
        if raw in ("yes", "no"):
            return raw
        print("Please enter 'yes' or 'no'.")

def vegWUISelectNLCD(nlcd_numpy,lowLeftPnt,cellSize):
    '''Helps the process of mapping WUI with vegetation classes.
    This function makes the code for Question 1 more streamlined.
    If the user answers 'yes' to Q1, this function will make 2
    arrays for the two main vegetation classes (forest and shrubland).
    Saves them as rasters for reference.'''
    forest = numpy.where((nlcd_numpy == 41) | (nlcd_numpy == 42) | (nlcd_numpy == 43), 1, 0)
    forest_save = arcpy.NumPyArrayToRaster(forest, lowLeftPnt, cellSize, cellSize)
    forest_save.save("steps/forest.tif")

    shrubland = numpy.where((nlcd_numpy == 52) | (nlcd_numpy == 71), 1, 0)
    shrubland_save = arcpy.NumPyArrayToRaster(shrubland, lowLeftPnt, cellSize, cellSize)
    shrubland_save.save("steps/shrubland.tif")
    return forest, shrubland

def circleWindow(classList, classListOut, classListNames, radius):
    '''Moving window using a circular mask "kernel". This function
    takes an input radius from the user, and applies it to the
    kernel/moving window mask. This means the user can change how
    big of a moving window they are using. We also use a fixed
    radius of 1 if the user says yes to vegetation class mapping.'''
    import time

    for i in range(len(classList)):
        start = time.time()
        if i < 2:
            in_radius = radius  # user-defined radius
        else:
            in_radius = 1  # fixed radius = 1
            # 'i' will only be bigger than 2 if 'yes' to veg mapping
        # Create a new kernel for this input radius (Lab 5 for code source)
        dim0 = int(in_radius)
        kernel = np.zeros((dim0 * 2 + 1, dim0 * 2 + 1))
        for row in range(0, kernel.shape[0]):
            for col in range(0, kernel.shape[1]):
                if (((dim0 - row) ** 2 + (dim0 - col) ** 2) ** 0.5) <= in_radius:
                    kernel[row, col] = 1
        #Focal window code using the kernel above (Lab 5 for code source)
        for row in range(in_radius, classList[i].shape[0] - in_radius):
            for col in range(in_radius, classList[i].shape[1] - in_radius):
                window = classList[i][row - in_radius:row + in_radius + 1, col - in_radius:col + in_radius + 1]
                valid_window = np.where(window > -1, window, 0)
                window_sum = (valid_window * kernel).sum()
                classListOut[i][row, col] = window_sum / kernel.sum()
        end = time.time()
        print("Moving window completed for", classListNames[i], "in", round(end - start, 2), "seconds")
    print("Circular moving windows complete")
    return classListOut

def compareWUIMaps(cellSize,lowLeftPnt,wui_intermix_select,wui_interface_select,bupl,bupl_clipped):
    '''This function compares the output WUI map to an input
    WUI map from the SILVIS lab. You should compare a SILVIS map
    with an approximately corresponding date to the WUI map you
    created. This function prepares and clips the SILVIS data for you. Therefore,
    you just have to download correct SILVIS data at the link below, and
    input it in the 'silvis_data' folder. We convert the rasters to numpy arrays,
    combine intermix and interface, add them, and calculate
    statistics of agreement between the two maps. '''

    # COMPARISON TO SILVIS WUI
    # STEP 1: DATA PREPARATION
    # Download SILVIS data for area of interest: https://silvis.forest.wisc.edu/data/wui-change/
    # Put SILVIS data in the silvis_data folder
    # Define the shapefile, relative to the current arcpy workspace
    silvis = os.path.join(arcpy.env.workspace, 'silvis_data', 'CA_wui_block_1990_2020_LA.shp')

    # Select areas of WUI from the shapefile for 2020
    #WUICLASS_2 is for year 2020. You should change this field if you want a different year.
    SILVIS_select = arcpy.analysis.Select(silvis, "steps/SILVIS_WUI_Select.shp","WUICLASS_2 = 'High_Dens_Interface' Or WUICLASS_2 = 'High_Dens_Intermix' Or WUICLASS_2 = 'Low_Dens_Interface' Or WUICLASS_2 = 'Low_Dens_Intermix' Or WUICLASS_2 = 'Med_Dens_Interface' Or WUICLASS_2 = 'Med_Dens_Intermix'")

    # Convert the polygon to raster
    arcpy.conversion.PolygonToRaster(SILVIS_select, "WUICLASS_2", "steps/SILVIS_WUI_Raster.tif","CELL_CENTER", "NONE", cellSize)
    # Make sure projections match
    arcpy.management.ProjectRaster("steps/SILVIS_WUI_Raster.tif", 'steps/silvis_reproject.tif', bupl.spatialReference, "NEAREST",cellSize)
    silvis_raster_reproj = sa.Raster("steps/silvis_reproject.tif")

    # Use extract by mask to make sure they are the same size
    silvis_ext_mask = sa.ExtractByMask(silvis_raster_reproj, bupl_clipped)
    silvis_ext_mask.save("steps/silvis_final.tif")

    # convert to numpy array
    silvis_numpy = arcpy.RasterToNumPyArray(silvis_ext_mask)

    # select intermix
    silvis_intermix = numpy.where((silvis_numpy == 1) | (silvis_numpy == 2) | (silvis_numpy == 3), 1, 0)

    # convert intermix back to raster
    silvis_intermix_raster = arcpy.NumPyArrayToRaster(silvis_intermix, lowLeftPnt, cellSize, cellSize)
    silvis_intermix_raster.save('steps/silvis_intermix.tif')

    # select interface
    silvis_interface = numpy.where((silvis_numpy == 4) | (silvis_numpy == 5) | (silvis_numpy == 6), 1, 0)

    # convert interface back to raster
    silvis_interface_raster = arcpy.NumPyArrayToRaster(silvis_interface, lowLeftPnt, cellSize, cellSize)
    silvis_interface_raster.save('steps/silvis_interface.tif')

    # COMPARISON OF THE INTERMIXES AND INTERFACES
    # combine intermix and interface for both our WUI and SILVIS maps
    combine_WUI = np.where((wui_intermix_select == 1) | (wui_interface_select == 1), 1, 0)

    combine_silvis = np.where((silvis_intermix == 1) | (silvis_interface == 1), 2, 0)

    combine_both = combine_WUI + combine_silvis

    mydict = {'True Positive': np.count_nonzero(combine_both == 3),
              'False Negative': np.count_nonzero(combine_both == 2),
              'False Positive': np.count_nonzero(combine_both == 1),
              'True Negative': np.count_nonzero(combine_both == 0)}

    accuracy = (mydict['True Positive'] + mydict['True Negative']) / (mydict['True Positive'] + mydict['False Positive'] + mydict['True Negative'] + mydict['False Negative'])
    precision = mydict['True Positive'] / (mydict['True Positive'] + mydict['False Positive'])
    recall = (mydict['True Positive']) / (mydict['True Positive'] + mydict['False Negative'])
    F1 = (2 * precision * recall) / (precision + recall)
    print(f"Accuracy: {round(accuracy,2)}")
    print(f"Precision: {round(precision,2)}")
    print(f"Recall: {round(recall,2)}")
    print(f"F1: {round(F1,2)}")
    return silvis_intermix, silvis_interface

def computeConfusionMatrixWUI(wui_intermix_select,wui_interface_select,silvis_intermix,silvis_interface):
    '''This function uses the sklearn library to generate a confusion
    matrix for comparison of the output WUI map to the SILVIS map.
    It also prints an SKlearn classification report to compare
    to the manually calculated statistics of the previous function.'''

    combine_WUI = np.where((wui_intermix_select == 1) | (wui_interface_select == 1), 1, 0)

    combine_silvis = np.where((silvis_intermix == 1) | (silvis_interface == 1), 1, 0)

    actual = combine_silvis.tolist()
    predicted = combine_WUI.tolist()

    # This line removes all extra brackets from actual
    actual = [item for sublist in actual for item in sublist]
    # This line removes all extra brackets from predicted
    predicted = [item for sublist in predicted for item in sublist]

    # Double check statistics of accuracy, precision, and F1 with SKlearn
    report = classification_report(actual, predicted)
    print("***SKLearn Report***")
    print(report)

    # Generate the confusion matrix
    cm = confusion_matrix(actual, predicted, labels=[1, 0])
    print("Confusion Matrix:\n", cm)
    return cm

def mapVegWUI(result,lowLeftPnt,cellSize,wui_interface_select,wui_intermix_select):
    '''This function applies vegetation classes to an output WUI map.
    It uses moving window outputs for 'forest' and 'shrubland' classes, and
    selects areas greater than a threshold of this type of vegetation cover.
    We found that >20% worked well. Then, we apply a buffer of 2.4km to these
    areas because the vegetation doesn't usually overlap with all the WUI areas.
    Then we basically clip these vegetation maps to the output WUI map, while
    differentiating between all the possible classes. '''
    # Vegetation mapping using moving window outputs
    # FOREST
    forest_win = result[2]
    forest_select = numpy.where((forest_win >= 0.2), 1, 0)
    forest_select_r = arcpy.NumPyArrayToRaster(forest_select, lowLeftPnt, cellSize, cellSize)
    forest_select_r.save('steps/forest_moving_window.tif')

    # Set 0 cells to NoData, keep 1 cells https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/set-null.htm
    forest_recode = sa.SetNull(forest_select_r == 0, forest_select_r)
    forest_recode.save("steps/forest_recode.tif")

    # convert to poly
    arcpy.conversion.RasterToPolygon(forest_recode, "steps/forest_poly.shp", "NO_SIMPLIFY", "Value")

    # buffer to 2.4km
    arcpy.analysis.Buffer("steps/forest_poly.shp", "steps/forest_buffer", 2400, "FULL", "ROUND", "ALL", None)

    # convert back to a raster
    arcpy.conversion.PolygonToRaster("steps/forest_buffer.shp", "FID", "steps/forest_buff_raster.tif", "CELL_CENTER", "NONE",cellSize)

    # Converting back to Numpy to select values easier
    forest_buff_numpy = arcpy.RasterToNumPyArray("steps/forest_buff_raster.tif")
    forest_buff_recode = numpy.where((forest_buff_numpy == 0), 10, 0)
    # Class forest = 10

    # Converting back to raster
    forest_buff_recode_raster = arcpy.NumPyArrayToRaster(forest_buff_recode, lowLeftPnt, cellSize, cellSize)
    forest_buff_recode_raster.save('steps/forest_buff_recode_raster.tif')

    # SHRUBLAND
    shrubland_win = result[3]
    shrubland_select = numpy.where((shrubland_win >= 0.2), 1, 0)
    shrubland_select_r = arcpy.NumPyArrayToRaster(shrubland_select, lowLeftPnt, cellSize, cellSize)
    shrubland_select_r.save('steps/shrubland_moving_window.tif')

    # Set 0 cells to NoData, keep 1 cells
    shrubland_recode = sa.SetNull(shrubland_select_r == 0, shrubland_select_r)
    shrubland_recode.save("steps/shrubland_recode.tif")

    # convert to poly
    arcpy.conversion.RasterToPolygon(shrubland_recode, "steps/shrubland_poly.shp", "NO_SIMPLIFY", "Value")

    # buffer to 2.4km
    arcpy.analysis.Buffer("steps/shrubland_poly.shp", "steps/shrubland_buffer", 2400, "FULL", "ROUND", "ALL", None)

    # convert back to a raster
    arcpy.conversion.PolygonToRaster("steps/shrubland_buffer.shp", "FID", "steps/shrubland_buff_raster.tif", "CELL_CENTER", "NONE",cellSize)

    # Converting back to Numpy to select values easier
    shrubland_buff_numpy = arcpy.RasterToNumPyArray("steps/shrubland_buff_raster.tif")
    shrubland_buff_recode = numpy.where((shrubland_buff_numpy == 0), 5, 0)
    # Class shrubland = 5

    # Converting back to raster
    shrubland_buff_recode_raster = arcpy.NumPyArrayToRaster(shrubland_buff_recode, lowLeftPnt, cellSize, cellSize)
    shrubland_buff_recode_raster.save('steps/shrubland_buff_recode_raster.tif')

    veg_sum = forest_buff_recode + shrubland_buff_recode
    veg_sum_raster = arcpy.NumPyArrayToRaster(veg_sum, lowLeftPnt, cellSize, cellSize)
    veg_sum_raster.save('steps/veg_sum_raster.tif')
    # veg_sum is forest and shrub added into one raster
    # Classes for veg_sum:
    # 0 = nothing
    # 5 = shrubland
    # 10 = forest
    # 15 = forest and shrubland

    # FOREST AND SHRUB INTERFACE
    wui_interface_forest_shrub = veg_sum + wui_interface_select
    wui_interface_forest_shrub_overlap = numpy.where((wui_interface_forest_shrub == 16), 1, 0)
    wui_interface_forest_shrub_overlap_r = arcpy.NumPyArrayToRaster(wui_interface_forest_shrub_overlap, lowLeftPnt,cellSize, cellSize)
    wui_interface_forest_shrub_overlap_r.save('steps/wui_interface_forest_shrub.tif')

    # FOREST AND SHRUB INTERMIX
    wui_intermix_forest_shrub = veg_sum + wui_intermix_select
    wui_intermix_forest_shrub_overlap = numpy.where((wui_intermix_forest_shrub == 16), 1, 0)
    wui_intermix_forest_shrub_overlap_r = arcpy.NumPyArrayToRaster(wui_intermix_forest_shrub_overlap, lowLeftPnt,cellSize, cellSize)
    wui_intermix_forest_shrub_overlap_r.save('steps/wui_intermix_forest_shrub.tif')

    # FOREST INTERFACE
    wui_interface_forest = veg_sum + wui_interface_select
    wui_interface_forest_overlap = numpy.where((wui_interface_forest == 11), 1, 0)
    wui_interface_forest_overlap_r = arcpy.NumPyArrayToRaster(wui_interface_forest_overlap, lowLeftPnt, cellSize,cellSize)
    wui_interface_forest_overlap_r.save('steps/wui_interface_forest.tif')

    # FOREST INTERMIX
    wui_intermix_forest = veg_sum + wui_intermix_select
    wui_intermix_forest_overlap = numpy.where((wui_intermix_forest == 11), 1, 0)
    wui_intermix_forest_overlap_r = arcpy.NumPyArrayToRaster(wui_intermix_forest_overlap, lowLeftPnt, cellSize,cellSize)
    wui_intermix_forest_overlap_r.save('steps/wui_intermix_forest.tif')

    # SHRUBLAND INTERFACE
    wui_interface_shrubland = veg_sum + wui_interface_select
    wui_interface_shrubland_overlap = numpy.where((wui_interface_shrubland == 6), 1, 0)
    wui_interface_shrubland_overlap_r = arcpy.NumPyArrayToRaster(wui_interface_shrubland_overlap, lowLeftPnt,cellSize, cellSize)
    wui_interface_shrubland_overlap_r.save('steps/wui_interface_shrubland.tif')

    # SHRUBLAND INTERMIX
    wui_intermix_shrubland = veg_sum + wui_intermix_select
    wui_intermix_shrubland_overlap = numpy.where((wui_intermix_shrubland == 6), 1, 0)
    wui_intermix_shrubland_overlap_r = arcpy.NumPyArrayToRaster(wui_intermix_shrubland_overlap, lowLeftPnt,cellSize, cellSize)
    wui_intermix_shrubland_overlap_r.save('steps/wui_intermix_shrubland.tif')

    # Preparing to add into one raster with the following classes:
    # wui_interface_forest_shrub_overlap = class 1
    # wui_intermix_forest_shrub_overlap = class 2
    # wui_interface_forest_overlap = class 3
    # wui_intermix_forest_overlap = class 4
    # wui_interface_shrubland_overlap = class 5
    # wui_intermix_shrubland_overlap = class 6

    wui_intermix_forest_shrub_overlap_reclass = numpy.where((wui_intermix_forest_shrub_overlap == 1), 2, 0)

    wui_interface_forest_overlap_reclass = numpy.where((wui_interface_forest_overlap == 1), 3, 0)

    wui_intermix_forest_overlap_reclass = numpy.where((wui_intermix_forest_overlap == 1), 4, 0)

    wui_interface_shrubland_overlap_reclass = numpy.where((wui_interface_shrubland_overlap == 1), 5, 0)

    wui_intermix_shrubland_overlap_reclass = numpy.where((wui_intermix_shrubland_overlap == 1), 6, 0)

    wui_veg_final_numpy = wui_interface_forest_shrub_overlap + wui_intermix_forest_shrub_overlap_reclass + wui_interface_forest_overlap_reclass + wui_intermix_forest_overlap_reclass + wui_interface_shrubland_overlap_reclass + wui_intermix_shrubland_overlap_reclass

    wui_veg_final_numpy_r = arcpy.NumPyArrayToRaster(wui_veg_final_numpy, lowLeftPnt, cellSize, cellSize)
    wui_veg_final_numpy_r.save('results/wui_veg.tif')
    print("*****ADDITIONAL WUI MAP WITH VEGETATION CLASSES COMPLETE*****")