'''*********************************************
author: Max Warnock, Quin Browder, Victoria Madden
Date: 5/3/2025
GEOG 4303 Final Project
WUI Mapping Tool - Main .py file
*********************************************'''

import numpy
import numpy as np
import arcpy
from arcpy import env
import arcpy.sa as sa
import wui_mapping_module
from sklearn.metrics import ConfusionMatrixDisplay

# Edit the path below to define your workspace.
env.workspace = r"C:\Users\warno\Desktop\wui_mapping_workspace"
env.overwriteOutput = 1
arcpy.CheckOutExtension("Spatial")

# DATA PREPARATION
# Define the NLCD raster from the data folder.
# This is the input raster that you can clip to your own study area of interest.
# You can use our sample data area for LA, or you can clip an NLCD
# raster to your own area of interest and input the name and path the line below.
nlcd = sa.Raster(r'/data/nlcd_2020.tif')
print("NLCD height:",nlcd.height)
print("NLCD width:",nlcd.width)
print("NLCD cell height:",nlcd.meanCellHeight)
print("NLCD cell width:",nlcd.meanCellWidth)

# Define the BUPL raster
# We will clip the BUPL raster in this script, so you can use a larger BUPL raster if you want.
bupl = sa.Raster(r'/data/BUPL_2020.tif')
arcpy.env.snapRaster = nlcd #https://pro.arcgis.com/en/pro-app/latest/tool-reference/environment-settings/snap-raster.htm
# Snap raster allows us to make sure extents of both NLCD and BUPL rasters are the same
arcpy.env.extent = nlcd.extent
arcpy.env.cellSize = bupl.meanCellHeight
print("BUPL height:",bupl.height)
print("BUPL width:",bupl.width)
print("BUPL cell height:",bupl.meanCellHeight)
print("BUPL cell width:",bupl.meanCellWidth)

# reproject NLCD raster to match BUPL
# Also, we're saving all intermediate steps to a 'steps' folder
nlcd_reproject = arcpy.management.ProjectRaster(nlcd,'steps/nlcd_reproject.tif',bupl.spatialReference,"NEAREST", env.cellSize)

# resample NLCD raster to match BUPL
nlcd_resample = arcpy.management.Resample(nlcd_reproject, 'steps/nlcd_resample.tif', env.cellSize, "NEAREST")
nlcd_resample = sa.Raster('steps/nlcd_resample.tif')

# Clip the BUPL data to the NLCD. This allows the user to just input
# a clip of the NLCD, and select a BUPL layer of their choice (for whole US).
# The BUPL is low enough resolution that this is still fairly efficient.
# To clip, we use extract by rectangle, which prevents NoData values on the edges.
bupl_clipped = sa.ExtractByRectangle(bupl, nlcd_resample)
bupl_clipped.save("steps/BUPL_Clipped.tif")
arcpy.env.mask = bupl_clipped
#Use extract by mask to make sure they are the same size
nlcd_final = sa.ExtractByMask(nlcd_resample, bupl_clipped)
nlcd_final.save("steps/nlcd_final.tif")


#check NLCD and BUPL final size. They should be the same
print("NLCD final height:",nlcd_final.height)
print("NLCD final width:",nlcd_final.width)
print("BUPL final height:",bupl_clipped.height)
print("BUPL final width:",bupl_clipped.width)

# convert NLCD to numpy array
nlcd_numpy = arcpy.RasterToNumPyArray(nlcd_final)

# convert BUPL to numpy array
bupl_numpy = arcpy.RasterToNumPyArray(bupl_clipped)

#check shapes
print("NLCD array shape:", nlcd_numpy.shape)
print("BUPL array shape:", bupl_numpy.shape)

# assign cell size, since height and width are the same
cellSize = bupl_clipped.meanCellHeight

# find the lower left point of the BUPL raster
lowLeftPnt = bupl_clipped.extent.lowerLeft

# select vegetation from NLCD array
vegetation = np.where((nlcd_numpy==41)|(nlcd_numpy==42)|(nlcd_numpy==43)|(nlcd_numpy==51)|(nlcd_numpy==52)|(nlcd_numpy==71), 1, 0)
vegetation_save = arcpy.NumPyArrayToRaster(vegetation, lowLeftPnt,cellSize,cellSize)
vegetation_save.save('steps/vegetation.tif')

while True:
    print("---Question 1---")
    print('Would you like to map the WUI according to different \n'
          'types of vegetation cover? Type yes or no.')
    vegCheckQ1 = input("yes/no:")
    if vegCheckQ1 == 'yes':
        forest, shrubland = wui_mapping_module.vegWUISelectNLCD(nlcd_numpy, lowLeftPnt, cellSize)
        classList = [bupl_numpy, vegetation, forest, shrubland]
        classListNames = ["bupl_numpy", "vegetation","forest","shrubland"]
    else:
        classList = [bupl_numpy, vegetation]
        classListNames = ["bupl_numpy", "vegetation"]
        break
    break

#MOVING WINDOW OPERATIONS (Steps 1 and 2)
# get the spatial reference of the raster
spref = bupl.spatialReference

# Create an output list of arrays of same size with zeros
classListOut = [numpy.zeros(array.shape).astype(float) for array in classList]

# We ask a question for the user to define the size of the mask radius
print("---Question 2---")
print("Choose radius for circular moving window...\nRadius of 1 = 250m")
print("Whole numbers only.")
print("Standard radius: 2")
question2 = input("Enter radius size:")
try:
    radius = int(question2)
except ValueError:
    print("Invalid number")

#run the moving window function (circleWindow)
result = wui_mapping_module.circleWindow(classList,classListOut,classListNames,radius)

#Select developed areas with at least x buildings per moving window radius size
print("---Question 3---")
print("Choose threshold for minimum number of buildings per cell:")
print("Standard threshold: 6.17")
print("Alternative threshold: 1")
question3 = input("Enter threshold:")
try:
    building_threshold = float(question3)
except ValueError:
    print("Invalid number")

developed_win = result[0] #select bupl moving window result
developed_select = np.where((developed_win >= building_threshold), 1, 0) #select areas >= threshold
developed_select_to_raster = arcpy.NumPyArrayToRaster(developed_select, lowLeftPnt,cellSize,cellSize)
developed_select_to_raster.save('steps/developed_select.tif')

#view vegetation result as raster for double checking
vegetation_win = result[1]
vegetation_win_raster = arcpy.NumPyArrayToRaster(vegetation_win, lowLeftPnt,cellSize,cellSize)
vegetation_win_raster.save('steps/vegetation_win.tif')

#Select vegetation areas with greater than x% vegetation cover
print("---Question 4---")
print("Choose vegetation areas greater than x percent for intermix criteria")
print("Standard percentage: 50")
question4 = input("Enter percent value:")
try:
    veggie_percent_g = float(question4)
except ValueError:
    print("Invalid number")
veggie_percent_convert_g = veggie_percent_g *0.01
# Select areas greater than input
vegetation_select_greater = np.where((vegetation_win >= veggie_percent_convert_g), 1, 0)
vegetation_select_to_raster_greater = arcpy.NumPyArrayToRaster(vegetation_select_greater, lowLeftPnt,cellSize,cellSize)
vegetation_select_to_raster_greater.save('steps/vegetation_greater.tif')
# Select areas less than input
vegetation_select_less = numpy.where((vegetation_win <= veggie_percent_convert_g), 1, 0)
vegetation_select_to_raster_less = arcpy.NumPyArrayToRaster(vegetation_select_less, lowLeftPnt,cellSize,cellSize)
vegetation_select_to_raster_less.save('steps/vegetation_less.tif')
print("Selections completed \nEnd of steps 1 and 2")

#STEP 3
print("---Question 5---")
print("Choose large vegetation area size...")
print("Input units in square km:")
print("Standard size: 5")
question5 = input("Enter size:")
try:
    sq_km = float(question5)
except ValueError:
    print("Invalid number")
# Calculate number of 250x250m cells needed for x km^2
sq_meters = sq_km * 1000000
square_cell_area = 250*250
veggie_area = sq_meters / square_cell_area
print("Number of cells for",sq_km,"square km:", veggie_area)

# Select vegetation areas greater than 75% (or other threshold)
print("---Question 6---")
print("Select vegetation % coverage for 'large vegetation areas'")
print("Standard value: 75")
question6 = input("Enter percent:")
try:
    large_veg_percent = float(question6)
except ValueError:
    print("Invalid number")
large_veg_percent_convert = large_veg_percent * 0.01
#select >= this value from the vegetation moving window
vegetation_select_for_interface = np.where((vegetation_win >= large_veg_percent_convert), 1, 0)
#convert to raster so we can apply region group tool
vegetation_for_interface_r = arcpy.NumPyArrayToRaster(vegetation_select_for_interface, lowLeftPnt,cellSize,cellSize)
vegetation_for_interface_r.save('steps/vegetation_for_interface.tif')

#region group tool
region_group_veg = sa.RegionGroup(vegetation_for_interface_r, "EIGHT", "WITHIN", "", 0)
region_group_raster_veg = "steps/vegetation_group.tif"
region_group_veg.save(region_group_raster_veg)

#use search cursor to get the count of cells per group
region_dict = {} #dictionary containing sizes of all groups
with arcpy.da.SearchCursor(region_group_raster_veg, ['Value', 'Count']) as cursor:
    for val, count in cursor:
        if val != 0 and count > veggie_area:
            region_dict[val] = 1 # codes it as 1, or NoData

# RemapValue function allows us to code all areas greater than user input size to 1.
# Other pixels are NoData.
remap = sa.RemapValue([[val, bin_val] for val, bin_val in region_dict.items()])

# Reclassify and apply out remapping definition.
veggie_five_km = sa.Reclassify(region_group_raster_veg, "Value", remap, "NODATA")
veggie_five_km.save("steps/veggie_five_km.tif")
veggie_five_km = sa.Raster('steps/veggie_five_km.tif')
print("Step 3 complete")

#STEP 4
# convert the x km^2 veggie data to polygons to make for easier buffering
veggie_to_poly = arcpy.conversion.RasterToPolygon(veggie_five_km,"steps/veggie_five_km_poly.shp","NO_SIMPLIFY", "Value")

#ask question for user to define the buffer distance
print("---Question 7---")
print("Choose the buffer distance for vegetation areas \nof at least your input of",sq_km,"square km")
print("Input units in kilometers")
print("Standard buffer distance: 2.4")
question7 = input("Enter size:")
try:
    buff_dist = float(question7)
except ValueError:
    print("Invalid number")

buff_m = buff_dist * 1000 # convert km to m for buffer tool

veggie_buffer = arcpy.analysis.Buffer(veggie_to_poly,"steps/veggie_buffer.shp",buff_m,"FULL","ROUND","ALL",None)

#convert back to a raster
veggie_buff_raster = arcpy.conversion.PolygonToRaster(veggie_buffer,"FID", "steps/veggie_buff_raster.tif","CELL_CENTER","NONE",cellSize)

#Using extract by mask to clip buffered areas outside the study area
veggie_buff_extract = sa.ExtractByMask(veggie_buff_raster, bupl_clipped)
veggie_buff_extract.save("steps/veggie_buff_extract.tif")

#Converting back to Numpy array to select values easier
veggie_buff_numpy = arcpy.RasterToNumPyArray(veggie_buff_extract)
veggie_buff_recode = np.where((veggie_buff_numpy == 0), 1, 0)

#Converting back to raster
veggie_buff_recode_raster = arcpy.NumPyArrayToRaster(veggie_buff_recode, lowLeftPnt,cellSize,cellSize)
veggie_buff_recode_raster.save('steps/veggie_buff_recode_raster.tif')
print("Step 4 complete")

#STEP 5 (selecting WUI intermix)
wui_intermix = developed_select + vegetation_select_greater
wui_intermix_select = numpy.where((wui_intermix == 2), 1, 0)
wui_intermix_raster = arcpy.NumPyArrayToRaster(wui_intermix_select,lowLeftPnt,cellSize,cellSize)
wui_intermix_raster.save('steps/wui_intermix.tif')
print("WUI intermix map complete")

#STEP 6 (select WUI interface)
wui_interface = developed_select + veggie_buff_recode + vegetation_select_less
wui_interface_select = numpy.where((wui_interface == 3), 1, 0)
wui_interface_raster = arcpy.NumPyArrayToRaster(wui_interface_select, lowLeftPnt,cellSize,cellSize)
wui_interface_raster.save('steps/wui_interface.tif')
print("WUI interface map complete")

# Combine WUI intermix and interface into one raster:
# Class value 1 = interface
# Class value 2 = intermix
# Save to results folder
wui_intermix_reclass = numpy.where((wui_intermix_select == 1), 2, 0)
wui_class_combine = wui_intermix_reclass + wui_interface_select
wui_class_combine_r = arcpy.NumPyArrayToRaster(wui_class_combine, lowLeftPnt,cellSize,cellSize)
wui_class_combine_r.save('results/wui.tif')

print("***************************************")
print("***Summarizing User Input parameters***")
print("Moving window radius:",radius)
print("Minimum building threshold:",building_threshold)
print("Vegetation cover selection greater than",veggie_percent_g,"percent")
print("Vegetation % coverage for 'large vegetation areas",large_veg_percent,"percent")
print("Vegetation region group selection of:",sq_km,"square km")
print("Buffer distance for region grouped vegetation:",buff_dist)
print("*******WUI MAP COMPLETE******* (Go check your results folder)")

#Vegetation mapping using moving window outputs
if vegCheckQ1 == 'yes':
    print("working on applying vegetation classes to WUI map...")
    wui_mapping_module.mapVegWUI(result,lowLeftPnt,cellSize,wui_interface_select,wui_intermix_select)

#Comparison statistics to SILVIS WUI map and confusion matrix
while True:
    print("---Question 8---")
    print("Would you like to compare your output WUI map to \n"
          "a SILVIS Lab WUI map?")
    print("Prototype: statistics of Accuracy, Precision, Recall, and F1, will be calculated.")
    question8 = input("Enter yes/no:")
    if question8 == 'yes':
        print("Working on comparison statistics...")
        silvis_intermix, silvis_interface = wui_mapping_module.compareWUIMaps(cellSize, lowLeftPnt, wui_intermix_select, wui_interface_select,bupl,bupl_clipped)
        print("---Question 9---")
        print("Do you want to use SKLearn to recompute statistics and a \n"
              "confusion matrix to compare your WUI map to the SILVIS WUI map?")
        question9 = input("Enter yes/no:")
        if question9 =='yes':
            print("Working on confusion matrix...")
            cMatrix = wui_mapping_module.computeConfusionMatrixWUI(wui_intermix_select, wui_interface_select, silvis_intermix, silvis_interface)
            disp = ConfusionMatrixDisplay(confusion_matrix=cMatrix, display_labels=["Positive", "Negative"])
            disp_plot = disp.plot()
            print("confusion matrix completed")
            print("ending script")
        break
    else:
        print("ending script")
        break