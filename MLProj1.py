import numpy, random
import matplotlib.pyplot as plt

eta = 0.02
numdig = 4
minval,maxval = 0, 2**numdig-1
niter = 100
alpha = 2

def randomgraph(minX, maxX, minY, maxY, pointNum):
    arr = []
    for x in range(0, pointNum):
        arr.append([random.randint(minX,maxX),random.randint(minY,maxY)])
    return arr

def genrandline(minX, maxX, minY, maxY):
    newminX = (((minX + maxX)/2.0) + minX)/2.0
    newmaxX = (((minX + maxX)/2.0) + maxX)/2.0
    newminY = (((minY + maxY)/2.0) + minY)/2.0
    newmaxY = (((minY + maxY)/2.0) + maxY)/2.0
    p1 = [random.uniform(newminX,newmaxX),random.uniform(newminY,newmaxY)]
    p2 = [random.uniform(newminX,newmaxX),random.uniform(newminY,newmaxY)]
    while (p1 == p2):
        p2 = [random.uniform(newminX, newmaxX), random.uniform(newminY, newmaxY)]
    return p1 + p2

def splitdata(points,line,posside):
    pointsA = []
    pointsB = []
    if(line[0]!=line[2]):
        m = float(line[3]-line[1])/(line[2]-line[0])
        if(m!=0):
            b = line[1] - m * line[0]
            for i in points:
                y = i[1]
                x = (y - b)/m
                if(x <= i[0]):
                    pointsA.append(i)
                else:
                    pointsB.append(i)
        else:
            for i in points:
                yconst = line[1]
                if(yconst <= i[1]):
                    pointsA.append(i)
                else:
                    pointsB.append(i)
    else:
        for i in points:
            xconst = line[0]
            if (xconst <= i[0]):
                pointsA.append(i)
            else:
                pointsB.append(i)
    if(posside=='r'):
        return pointsB,pointsA
    return pointsA,pointsB

def newweightrecalc(oldweight, trueclass, predclass, x):
    return (oldweight + eta*(trueclass - predclass)*x)

def WNweightcalc(oldweight, trueclass, predclass, x):
    if (x > 0):
        return oldweight* alpha ** (trueclass - predclass)
    return oldweight

def dotproduct(x,w):
    s = 0
    for i in range(len(x)):
        s = s + x[i] * w[i]
    return s

def genweights(x):
    weights = []
    for i in range(len(x)-1):
        weights.append(random.random())
    return weights

def WNgenweights(x):
    weights = []
    for i in range(len(x)-1):
        weights.append(1)
    return weights

def sign(x,w):
    sum = dotproduct(x,w)
    s = 0
    if(sum>=0):
        s = 1
    else:
        s = 0
    return s

def WNsign(x,w):
    sum = dotproduct(x,w)
    theta = len(w) - 0.01
    s = 0
    if(sum>=theta):
        s = 1
    else:
        s = 0
    return s

def train(input, weight):
    guess = sign(input[:-1],weight)
    if(guess == input[-1]):
        return weight,0
    for i in range(len(weight)):
        weight[i] = newweightrecalc(weight[i],input[-1],guess, input[i])
    return weight, 1

def WNtrain(input, weight):
    guess = WNsign(input[:-1],weight)
    if(guess == input[-1]):
        return weight,0
    for i in range(len(weight)):
        weight[i] = WNweightcalc(weight[i],input[-1],guess, input[i])
    return weight, 1

def makebin(intarr,numdig):
    binarr = []
    for i in intarr:
        ba=[int(x) for x in list('{0:0b}'.format(i))]
        for ix in range(len(ba),numdig):
            ba = [0] + ba
        binarr = binarr + ba
    return binarr

def inversebin(arr):
    binarr = []
    for b in arr:
        if (b>0):
            binarr.append(0)
        else:
            binarr.append(1)
    return binarr

def addvardata(pos,neg,numofbad):
    totaldata = []
    rows = len(pos) + len(neg)
    for i in range(rows):
        col = [1]
        if (i < len(pos)):
            col = col + makebin(pos[i],numdig)
            #col = col + pos[i]
            for ib in range(numofbad):
                col.append(random.randint(0,1))
            col.append(1)
        else:
            col = col + makebin(neg[len(pos)-i],numdig)
            #col = col + neg[len(pos)-i]
            for ib in range(numofbad):
                col.append(random.randint(0,1))
            col.append(0)
        totaldata.append(col)
    random.shuffle(totaldata)
    return totaldata

def WNaddvardata(pos,neg,numofbad):
    totaldata = []
    rows = len(pos) + len(neg)
    for i in range(rows):
        col = []
        if (i < len(pos)):
            t = []
            for ib in range(numofbad):
                t.append(random.randint(0,1))
            col = makebin(pos[i],numdig) + t + inversebin(makebin(pos[i],numdig)) + inversebin(t)
            col.append(1)
        else:
            t = []
            for ib in range(numofbad):
                t.append(random.randint(0,1))
            col = makebin(neg[i-len(pos)],numdig) + t + inversebin(makebin(neg[i-len(pos)],numdig)) + inversebin(t)
            col.append(0)
        totaldata.append(col)
    random.shuffle(totaldata)
    return totaldata

def plotdata(data):
    for d in data:
        if (d[-1]):
            plt.plot(d[1],d[2], marker='+', color='b', ls='')
        else:
            plt.plot(d[1],d[2], marker='.', color='r', ls='')

def conbintoint(barr):
    sum = 0
    for bi in range(len(barr)):
        sum += 2**(bi)*barr[-1*(bi+1)]
    return sum

def intarr(arr, num):
    intar = []
    for i in range(0,len(arr),num):
        intar.append(conbintoint(arr[i:i+num]))
    return intar

def plotbin(tarr,num):
    tdata = []
    print(tarr[0][1:-1])
    for e in tarr:
        t = e[0] + intarr(e[1:-1],num) + e[-1]
    print(tdata)
    plotdata(tdata)


points = randomgraph(minval,maxval,minval,maxval,12)
line = genrandline(minval,maxval,minval,maxval)
pos,neg = splitdata(points,line,'r')
t = numpy.arange(-10, 10, 0.5)
#print(line)
#print(pos)
data = addvardata(pos,neg,3)
#print(data)
#plotdata(data)
wt = genweights(data[0])
error = []
for i in range(niter):
    es = 0
    for d in data:
        wt,e = train(d,wt)
        es = es + e
    error.append(es)
    print(wt)
print(error)
print(data)
WNdata = WNaddvardata(pos,neg,3)
WNwt = genweights(WNdata[0])
WNerror = []
for i in range(niter):
    es = 0
    for d in WNdata:
        WNwt,e = train(d,WNwt)
        es = es + e
    WNerror.append(es)
    print(WNwt)
print(WNerror)
# plotbin(data,numdig)
#print(intarr([0,1,0,0,1,1,0,1],numdig))
#plt.plot(*zip(*pos), marker='+', color='b', ls='')
#plt.plot(*zip(*neg), marker='.', color='r', ls='')
#plt.plot([line[0],line[2]],[line[1],line[3]],  color='g')
#plt.show()
