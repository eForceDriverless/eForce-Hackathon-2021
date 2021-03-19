import sys
import json

def read_pointcloud(filename):
    with open(filename) as f:
        return json.load(f)

Kuzele = read_pointcloud(sys.argv[1])
import matplotlib.pyplot as plt

f = open(sys.argv[2], "w")
Out = []
for idx, i in enumerate(Kuzele):
    maxZ = -999
    minZ = 999
    maxI = -999
    minI = 999
    for j in i:
        if j["z"] > maxZ:
            maxZ = j["z"]
        if j["z"] < minZ:
            minZ = j["z"]
        if j["intensity"] > maxI:
            maxI = j["intensity"]
        if j["intensity"] < minI:
            minI = j["intensity"] 

    StredD = (maxZ-minZ)/3+minZ
    StredN = (maxZ-minZ)*2/3+minZ
    StredI = (maxI+minI)/2
    HS = 0
    HC = 0
    SS = 0
    SC = 0
    LS = 0
    LC = 0
    Vys = "unknown"
    for j in i:
        if j["intensity"] < StredI:
            continue
        
        if j["z"] > StredN:
            HS += j["intensity"]
            HC += 1
        elif j["z"] > StredD:
            SS += j["intensity"]
            SC += 1
        else:
            LS += j["intensity"]
            LC += 1

    if HC == 0:
        HR = 0
    else:
        HR = (HS/HC)

    if SC == 0:
        SR = 0
    else:
        SR = (SS/SC)

    if LC == 0:
        LR = 0
    else:
        LR = (LS/LC)

    if HR > SR or LR > SR:
        Vys = "yellow"
    else:
        Vys = "blue"
    Out.append(Vys)

f.write(str(Out).replace("'", "\""))
f.close()