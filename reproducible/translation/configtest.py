import json

maxt = 200
n = 4
subnets = json.load(open('config_4.json'))
p = [0, 17, 33, 50, 51]
tasks = [[] for _ in range(n)]

target = 18

sb18 = subnets[target]

sbs = [json.loads(s)['op'] for s in subnets[:target]]

ops = json.loads(sb18)['op']
#print(sbs)
#print(ops)

nup = []
up = []
noshow = []
for i,op in enumerate(ops):
    updated = False
    cnt = 0
    appear = False
    for j, sb in enumerate(sbs):
        if op == sb[i]:
            updated = True 
            cnt += 1
            #print("updated by ", j, " on ", op)
        for sop in sb:
            if op == sop:
                appear = True

    if not appear:
        print("not appear ", op)
    if not updated:
        nup.append(op)
        #print("not updated ", op)
    elif cnt ==1:
        up.append(op)
        #print("update ", op, " by times", cnt)

print(set(nup))
print(set(up))

def print_task(tasks):
    for task in tasks:
        for t in task:
            print(t)
        print("--------------------")

def has_same(a, b):
    for a_, b_ in zip(a ,b):
        if a_ == b_:
            return True
    return False

def check_dependency(checklist):
    for i, t in enumerate(checklist):
        dependency = False
        for ct in checklist[:i]:
            if has_same(t[0], ct[0]):
                dependency = True
                break
        t[-1] = dependency

for i, sub in enumerate(subnets[:5000]):
    sub = json.loads(sub)
    op = sub['op']
    # op = [i, i] + sub['op']
    f = [[0]]
    for i in range(n):
        f.append([float('inf')])
    b = [f[-1]]
    for i in range(n):
        b.append([float('inf')])
    for i in range(n):
        # receive / send forward, receive / send backward, forward, backward, dependency
        tasks[i].append([op[p[i]:p[i + 1]], f[i], f[i + 1], b[n - i - 1], b[n - i], False])



# print_task(tasks)

