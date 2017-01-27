import os
import sys
import subprocess
# import qt


SRemesh = "/Users/prisgdd/Documents/Projects/Groups/SurfRemesh-build/-build/bin/SRemesh"

meshDir = "/Users/prisgdd/Desktop/TestPipeline/inputGroups/Mesh"
sphereDir = "/Users/prisgdd/Desktop/TestPipeline/inputGroups/sphere"
coeffDir = "/Users/prisgdd/Desktop/TestPipeline/outputGroups"
outputDir = "/Users/prisgdd/Desktop/TestPipeline/outputSurfRemesh"

# Verify directory integrity
if not os.path.isdir(meshDir) or not os.path.isdir(sphereDir) or not os.path.isdir(coeffDir) or not os.path.isdir(outputDir):
	sys.exit("Error: At least one input is not a directory.")


listMesh = os.listdir(meshDir)
listSphere = os.listdir(sphereDir)
listCoeff = os.listdir(coeffDir)

refSphere = "/Users/prisgdd/Desktop/TestPipeline/inputSurfRemesh/ref-sphere.vtk"
# refSphere = sphereDir + "/" + listSphere[0]

if listMesh.count(".DS_Store"):
	listMesh.remove(".DS_Store")
if listSphere.count(".DS_Store"):
	listSphere.remove(".DS_Store")
if listCoeff.count(".DS_Store"):
	listCoeff.remove(".DS_Store")


for i in range(0,len(listMesh)):
	command = list()

	command.append(SRemesh)
	command.append("-t")
	command.append(sphereDir + "/" + listSphere[i])

	command.append("-i")
	command.append(meshDir + "/" + listMesh[i])

	command.append("-r")
	command.append(refSphere)

	command.append("-c")
	command.append(coeffDir + "/" + listCoeff[i])

	outputFile = outputDir + "/" + listCoeff[i].split(".")[:-1][0] + "-Remesh.vtk"
	file = open(outputFile, 'w')
	file.close()
	command.append("-o")
	command.append(outputFile)

	# Run SurfRemesh
	print subprocess.call(command)



	# # Process to run SurfRemesh
	# process = qt.QProcess()
	# process.setProcessChannelMode(qt.QProcess.MergedChannels)

	# process.start(SRemesh, arguments)
	# process.waitForStarted()
	# # print "state: " + str(self.process.state())
	# process.waitForFinished(-1)
	# # print "error: " + str(self.process.error())

	# processOutput = str(self.process.readAll())








