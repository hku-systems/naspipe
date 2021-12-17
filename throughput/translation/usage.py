import time
import nvidia_smi
import sys
nvidia_smi.nvmlInit()

handles = []
for i in range(4):
    handles.append(nvidia_smi.nvmlDeviceGetHandleByIndex(i))

with open(sys.argv[1], 'w') as f:
    while True:
        usage = []
        for handle in handles:
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            usage.append(res.gpu)
        f.write("%.2f %.2f %.2f %.2f\n" % (usage[0], usage[1], usage[2], usage[3]))
        time.sleep(0.1)