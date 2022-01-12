import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

params = {'axes.labelsize': '7',
          'xtick.labelsize': '5',
          'ytick.labelsize': '5',
          # 'lines.linewidth': '0.2',
          'figure.figsize': '4, 2',
          'legend.fontsize': '5'}
from matplotlib import rcParams
rcParams.update(params)

labels = ['NLP.c0', 'NLP.c1', 'NLP.c2', 'NLP.c3', 'CV.c1', 'CV.c2', 'CV.c3']

f = open(sys.argv[1], "r")
content = f.read()
content_list = content.splitlines()
f.close()



gpipe = [0, 0, 0, 0, 0.1, 0.1, 0.1]
# pipedream
naspipe_rep = [0, 0, 0, 0, 0, 0, 0]
# vpipe
naspipe_switch = [0, 0, 0, 0, 0, 0, 0]
naspipe = [0, 0, 0, 0, 0, 0, 0]

gpipe_b = [0, 32, 64, 128, 0.1, 0.1, 0.1]
# pipedream
naspipe_rep_b = [0, 16, 24, 48, 0, 0, 0]
# vpipe
naspipe_switch_b = [192,192,192,192,192,192,192]
naspipe_b = [192,192,192,192,192,192,192]

curr = "default"
cnt = -1
for line in content_list:
    if line == "NasPipe.":
        curr = "naspipe"
        cnt = 0 
    elif line == "GPipe.":
        curr = "gpipe"
        cnt = 0
    elif line == "Pipedream.":
        curr = "pipedream"
        cnt = 0 
    elif line == "VPipe.":
        curr = "vpipe"
        cnt = 0 

    if curr == "naspipe":
        tokens = line.split(' ')
        if tokens[0] == "Steps:":
            steps = float(tokens[1])
            time = float(tokens[3])
            naspipe[cnt] = (1 / time)*steps*192*64
            cnt += 1
    if curr == "vpipe":
        tokens = line.split(' ')
        if tokens[0] == "Steps:":
            steps = float(tokens[1])
            time = float(tokens[3])
            naspipe_switch[cnt] = (1 / time)*steps*192*64
            cnt += 1

    if curr == "gpipe":
        tokens = line.split(' ')
        if tokens[0] == "Steps:":
            steps = float(tokens[1])
            time = float(tokens[3])
            gpipe[cnt+1] = (1 / time)*steps*gpipe_b[cnt+1]*8
            cnt += 1

    if curr == "pipedream":
        tokens = line.split(' ')
        if tokens[0] == "Steps:":
            steps = float(tokens[1])
            time = float(tokens[3])
            naspipe_rep[cnt+1] = (1 / time)*steps*naspipe_rep_b[cnt+1]*8
            cnt += 1




# normalized
gpipe_nm = [0, 1, 1, 1, 0, 0, 0]
naspipe_rep_nm = [0]
naspipe_switch_nm = [0]
naspipe_nm = [0]
for i in range(1, len(gpipe)):
    naspipe_rep_nm.append(naspipe_rep[i]/gpipe[i])
    naspipe_switch_nm.append(naspipe_switch[i]/gpipe[i])
    naspipe_nm.append(naspipe[i]/gpipe[i])
naspipe_switch_nm[0] = naspipe_switch[0]/gpipe[1]
naspipe_nm[0] =  naspipe[0] / gpipe[1]
x = np.arange(len(labels))  # the label locations
width = 0.20  # the width of the bars

fig, ax = plt.subplots()

b1 = ax.bar(x-1.5*width, gpipe_nm, width, color='#17becf')
b2 = ax.bar(x-0.5*width, naspipe_rep_nm, width, color='lightgreen')
b3 = ax.bar(x+0.5*width, naspipe_switch_nm, width, color='burlywood')
b4 = ax.bar(x+1.5*width, naspipe_nm, width, color='r')

# for a,b,c in zip(x,gpipe_nm, gpipe):  
#     plt.text(a-1.5*width, b+0.015, '%.1f' % c, ha='center', va= 'bottom',fontsize=5) 

# for a,b,c in zip(x,naspipe_rep_nm, naspipe_rep):  
#     plt.text(a-0.5*width, b+0.015, '%.1f' % c, ha='center', va= 'bottom',fontsize=5) 
    
# for a,b,c in zip(x,naspipe_switch_nm, naspipe_switch):  
#     plt.text(a+0.5*width, b+0.015, '%.1f' % c, ha='center', va= 'bottom',fontsize=5) 
naspipee =  ["%.2f" % (naspipe[0]/(192*64*1000))+'k', "%.2f" % (naspipe[1]/(192*64*1000))+'k', "%.2f" % (naspipe[2]/(192*64*1000))+'k', "%.2f" % (naspipe[3]/(192*64*1000))+'k', ' ', ' ', ' ']
for a,b,c in zip(x, naspipe_nm, naspipee):
    plt.text(a+1.5*width, b+0.015, '%s' % c, ha='center', va= 'bottom',fontsize=5) 
plt.text(-1*width, 0.015, 'N/A', ha='center', va= 'bottom',fontsize=5)

# b3 = ax.bar(x + 0.5*width, oneshot, width-0.04, color='y', edgecolor='y')
# b4 = ax.bar(x + 1.5*width, xxxoneshot, width-0.04, color='b', edgecolor='b') 

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Search Space')
ax.set_ylabel('Normalized Throughput')
ax.set_xticks(x)
ax.set_xticklabels(labels)
# ax.set_ylim(bottom = 0.5)
ax.set_ylim(top = 12)
ax.legend(labels=['GPipe', 'PipeDream','VPipe', 'NASPipe'])
# fig.legend(handles=[b1,b2], labels=['Retiarii-NasPipe', 'Retiarii-Gpipe'])

fig.tight_layout()
fig.savefig(sys.argv[2] + ".pdf")
# plt.show()