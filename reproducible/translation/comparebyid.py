

f = open("reproducible1", "r")
content = f.read()
loglist1 = content.splitlines()
f.close()

f = open("reproducible2", "r")
content = f.read()
loglist2 = content.splitlines()
f.close()


for i, line1 in enumerate(loglist1):
   
    if i >= len(loglist2):
        break


    seqid1 = line1.split(' ')[-4].split('\t')[0]

    for j, line2 in enumerate(loglist2):

        #print(line2)
        seqid2 = line2.split(' ')[-4].split('\t')[0]

        if seqid1 == seqid2:
            loss1 = line1.split(' ')[-2]
            loss2 = line2.split(' ')[-2]

    

            if loss1 != loss2:
                print("diff at", seqid1, " ", seqid2)
                print(loss1)
                print(loglist1[i-1])
                print(loglist1[i-2])
        # print(loglist1[i-3])
        # print(loglist1[i-4])
                print(loss2)
                print(loglist2[j-1])
                print(loglist2[j-2])        
        # print(loglist2[i-3])
        # print(loglist2[i-4])