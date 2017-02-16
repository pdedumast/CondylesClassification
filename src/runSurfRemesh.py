import os
import sys
import subprocess


SRemesh = "/Users/prisgdd/Documents/Projects/Groups/SurfRemesh-build/-build/bin/SRemesh"


parser = argparse.ArgumentParser()
parser.add_argument('-meshDir', action='store', dest='meshDir', help='Input file to classify', 
                    default = "/Users/prisgdd/Desktop/TestPipeline/inputGroups/Mesh")
parser.add_argument('-sphereDir', action='store', dest='sphereDir', help='Directory with sphere', 
					default="/Users/prisgdd/Desktop/TestPipeline/outputSurfRemesh")
parser.add_argument('-coeffDir', action='store', dest='coeffDir', help='Directory containing coefficients from Groups', 
					default="/Users/prisgdd/Desktop/TestPipeline/outputSurfRemesh")
parser.add_argument('-outputDir', action='store', dest='outputDir', help='Directory for output files', 
					default="/Users/prisgdd/Desktop/TestPipeline/outputSurfRemesh")

args = parser.parse_args()
meshDir= args.meshDir
sphereDir = args.sphereDir
coeffDir = args.coeffDir
outputDir = args.outputDir

# Verify directory integrity
if not os.path.isdir(meshDir) or not os.path.isdir(sphereDir) or not os.path.isdir(coeffDir) or not os.path.isdir(outputDir):
	sys.exit("Error: At least one input is not a directory.")

listMesh = os.listdir(meshDir)
listSphere = os.listdir(sphereDir)
listCoeff = os.listdir(coeffDir)

# refSphere = "/Users/prisgdd/Desktop/TestPipeline/inputSurfRemesh/ref-sphere.vtk"
refSphere = sphereDir + "/" + listSphere[0]

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








