#####################################################################################################################
##### Python modules
import os, json
import numpy as np

##### OpenSRA modules and functions
from src import fcn_gen

##### SimCenter modules
import lib.simcenter.OpenSHAInterface as OpenSHAInterface

#####
save_dir = r'C:\Users\barry\OneDrive - SlateGeotech\Fragility\OpenSRA\lib\simcenter\ucerf3_1'

#####
reg_proc = OpenSHAInterface.initEQHazardInterface(1)

##
nSources = reg_proc.getNumSources()

## Initializing the interface to EQHazardCalc
rupSet, rupSourceSections = OpenSHAInterface.set_up_get_rup()
seg_unique = {}
ps_unique = {}
src_connect = {}

for src_i in range(nSources):
    if src_i%10000==0:
        print(src_i)
    try:
        ## get list of segments
        listSeg = list(rupSourceSections[src_i].toArray())
        src_connect.update({str(src_i):listSeg})

        ## get list of nodes for all segments in current source
        for j in range(len(listSeg)):
            if not str(listSeg[j]) in seg_unique:
                seg_unique[str(listSeg[j])] = {}
                section = rupSet.getFaultSectionData(listSeg[j])
                trace = section.getFaultTrace()
                nodes = []
                for point in trace:
                    nodes.append([float(point.getLongitude()),float(point.getLatitude()),float(point.getDepth())])
                ##
                seg_unique[str(listSeg[j])].update({'name':str(section.getSectionName())})
                seg_unique[str(listSeg[j])].update({'trace':nodes})

    except:
        ## point sources
        reg_proc.setCurrentRupture(src_i,0)
        rupture = reg_proc.getRupture()
        surface = rupture.getRuptureSurface()
        nodes = [[float(surface.getLocation().getLongitude()), 
                    float(surface.getLocation().getLatitude()), 
                    float(surface.getLocation().getDepth())]]
        ps_unique.update({str(src_i):nodes[0]})


##
src_connect_arr = []
for i in src_connect:
    src_connect_arr.append([i,src_connect[i]])
    
f = open(os.path.join(save_dir,'src_connect.txt'), "w")
for line in src_connect_arr:
    # write line to output file
    f.write(str(line))
    f.write("\n")
f.close()

##
with open(os.path.join(save_dir,'rup_seg.json'), "w") as f:
    json.dump(seg_unique, f, indent=4)
    
##
ps_unique_arr = []
for i in ps_unique:
    ps_unique_arr.append([i,ps_unique[i][0],ps_unique[i][1],ps_unique[i][2]])

np.savetxt(os.path.join(save_dir,'point_source.txt'),ps_unique_arr,fmt='%s')