import cameraMatrix as cm
import pickle

with open("data.pickle","rb") as fr:
    cali = pickle.load(fr)

print("Connecting Camera...")
cali.removeCamDistort(0)
print("Finish--")