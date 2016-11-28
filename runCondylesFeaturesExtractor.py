import os
import sys
import subprocess
# import qt


CondylesFeaturesExtractor = "/Users/prisgdd/Documents/Projects/CNN/CondylesFeaturesExtractor-build/src/bin/condylesfeaturesextractor"

meshDir = "/Users/prisgdd/Documents/Projects/CNN/DataVTK/G07"
outputDir = "/Users/prisgdd/Documents/Projects/CNN/outputVTK-CondFeatExt/G07"
meanGroup = "/Users/prisgdd/Documents/Projects/CNN/drive-download-20161123T180828Z"


# Verify directory integrity
if not os.path.isdir(meshDir) or not os.path.isdir(outputDir):
	sys.exit("Error: At least one input is not a directory.")

listMesh = os.listdir(meshDir)

if listMesh.count(".DS_Store"):
	listMesh.remove(".DS_Store")

for i in range(0,len(listMesh)):
	command = list()

	command.append(CondylesFeaturesExtractor)
	command.append("--input")
	command.append(meshDir + "/" + listMesh[i])

	outputFile = outputDir + "/" + listMesh[i].split(".")[:-1][0] + "-Features.vtk"
	print outputFile
	file = open(outputFile, 'w')
	file.close()
	command.append("--output")
	command.append(outputFile)

	command.append("--meanGroup")
	command.append(meanGroup)	

	subprocess.call(command)

## Process to run SurfRemesh
# process = qt.QProcess()
# process.setProcessChannelMode(qt.QProcess.MergedChannels)

# process.start(SRemesh, arguments)
# process.waitForStarted()
# # print "state: " + str(self.process.state())
# process.waitForFinished(-1)
# # print "error: " + str(self.process.error())

# processOutput = str(self.process.readAll())








