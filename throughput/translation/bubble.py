import json

maxt = 200
n = 4
subnets = json.load(open('c_48.json'))
# p = [0, 17, 33, 50, 51]
p = [0, 8, 16, 24, 32]
tasks = [[] for _ in range(n)]

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

for task in tasks:
    check_dependency(task[:maxt])

# print_task(tasks)

data = [0, 0]
def run(task, s):
    for i, t in enumerate(task[:maxt]):
        if not t[-1]:
            if t[1][0] <= s and t[2][0] == float('inf'):
                t[2][0] = s + 1
                data[1] += 1
                break
            elif t[3][0] <= s and t[4][0] == float('inf'):
                t[4][0] = s + 1
                data[1] += 1
                task.pop(i)
                check_dependency(task[:maxt])
                break

# for s in range(20):
#     for task in tasks:
#         run(task, s)

#     print_task(tasks)

s = 0
while True:
    data[0] += n

    for task in tasks:
        run(task, s)
    s += 1

    print(sum([len(task) for task in tasks]))
    if sum([len(task) for task in tasks]) == 0:
        break

print(data[1] / data[0])