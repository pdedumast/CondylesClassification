import os
import sys
import shutil

def cleanDir(currentDir):
	for element in os.listdir(currentDir) :
		if os.path.isdir(currentDir + "/" + element):
			shutil.rmtree(currentDir + "/" + element)
		else:
			os.remove(currentDir + "/" + element)

# ------------------------------------------------------------------------------------- #


def sortData(inputDir, outputDir, mappedSphere):
	# Verify directory integrity
	if not os.path.isdir(spharmDir) or not os.path.isdir(groupsDir):
		sys.exit("Error: " + spharmDir + " or " + groupsDir + " is not a directory.")

	# Prepare Groups directory (both .DS_Store & previous data)
	cleanDir(groupsDir)

	meshDir = groupsDir + "/Mesh"
	os.makedirs(meshDir)
	propDir = groupsDir + "/attributes"
	os.makedirs(propDir)
	sphereDir = groupsDir +"/sphere"
	os.makedirs(sphereDir)

	suffixSphere1 = "surf_para.vtk" 
	suffixSphere2 = "para.vtk"
	
	suffixMesh = "surfSPHARM_procalign.vtk"
	suffixProp1 = "surf_paraPhi.txt"
	suffixProp2 = "surf_paraTheta.txt"
	suffixProp3 = "surf_medialMeshArea.txt"
	suffixProp4 = "surf_medialMeshPartialArea.txt"
	suffixProp5 = "surf_medialMeshPartialRadius.txt"
	suffixProp6 = "surf_medialMeshRadius.txt"

	spharmList = os.listdir(spharmDir)
	for file in spharmList:
		# Check if it's a sphere
		if file[len(file)-len(suffixSphere1):] == suffixSphere1 and mappedSphere:
			shutil.copyfile(spharmDir + "/" + file, sphereDir + "/" + file)

		elif file[len(file)-len(suffixSphere1):] != suffixSphere1 and file[len(file)-len(suffixSphere2):] == suffixSphere2 and not mappedSphere:
			shutil.copyfile(spharmDir + "/" + file, sphereDir + "/" + file)
		# Check if it's a mesh
		elif file[len(file)-len(suffixMesh):] == suffixMesh:
			shutil.copyfile(spharmDir + "/" + file, meshDir + "/" + file)

		# Check if it's a property
		elif file[len(file)-len(suffixProp1):] == suffixProp1 or file[len(file)-len(suffixProp2):] == suffixProp2 or file[len(file)-len(suffixProp3):] == suffixProp3 or file[len(file)-len(suffixProp4):] == suffixProp4 or file[len(file)-len(suffixProp5):] == suffixProp5 or file[len(file)-len(suffixProp6):] == suffixProp6:
			shutil.copyfile(spharmDir + "/" + file, propDir + "/" + file)

	return True

# ------------------------------------------------------------------------------------- #
# 																						#
# ------------------------------------------------------------------------------------- #

spharmDir = "/Users/prisgdd/Desktop/TestPipeline/outputSPHARM/Mesh/SPHARM"
groupsDir = "/Users/prisgdd/Desktop/TestPipeline/inputGroups"
mappedSphere = True


sortData(spharmDir, groupsDir, mappedSphere)





