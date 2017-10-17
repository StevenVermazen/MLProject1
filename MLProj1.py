import numpy, random
import matplotlib.pyplot as plt

eta = 0.02
numdig = 8
minval,maxval = 0, 2**numdig-1
niter = 100
alpha = 2

def randomgraph(relattr, pointNum):
    arr = []
    for r in range(0, pointNum):
        col = []
        for c in range(0, pointNum):

    return arr


def genweights(x):
    # creates random weights for perceptron
    weights = []
    for i in range(len(x)-1):
        weights.append(0.5)
    return weights

def WNgenweights(x):
    # makes all the weights and sets them as 1
    # which is apparently better than random val
    weights = []
    for i in range(len(x)-1):
        weights.append(1)
    return weights

def splitdata(points,line,posside):
    #this breaks the data
    #points is all the randomly generated points
    #line is a list with 4 digits [x1,y1,x2,y2]
    pointsA = []
    pointsB = []
    #2 special cases needed to check first being if x1 = x2
    #if so slope is infinite and you cant calc m
    if(line[0]!=line[2]):
        m = float(line[3]-line[1])/(line[2]-line[0])
        # second special case being if y1 = y2
        # meaning slope is 0
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
            # so if slope is 0 its pretty easy to just use 1 of the y
            # and check if the random points' y is less or greater to yconst
            for i in points:
                yconst = line[1]
                if(yconst <= i[1]):
                    pointsA.append(i)
                else:
                    pointsB.append(i)
    else:
        # if x is const then we can just check whether
        # the points' x is greater or less than xconst
        for i in points:
            xconst = line[0]
            if (xconst <= i[0]):
                pointsA.append(i)
            else:
                pointsB.append(i)
    # to give the option of randomly picking which side was positive and which was negative
    # r was random probably should've used a boolean
    if(posside=='r'):
        return pointsB,pointsA
    return pointsA,pointsB

def newweightrecalc(oldweight, trueclass, predclass, x):
    # just the proceptron weight calculation
    return (oldweight + eta*(trueclass - predclass)*x)

def WNweightcalc(oldweight, trueclass, predclass, x):
    # just the WINNOW weight calculation
    if (x > 0):
        return oldweight* alpha ** (trueclass - predclass)
    return oldweight

def dotproduct(x,w):
    # getting the sum of form multipling two vectors
    s = 0
    for i in range(len(x)):
        s = s + x[i] * w[i]
    return s

def sign(x,w):
    # if dotproduct is greater than or equal to 0 to check whether it was positive
    sum = dotproduct(x,w)
    s = 0
    if(sum>=0):
        s = 1
    else:
        s = 0
    return s

def WNsign(x,w):
    # if dotproduct is greater than or equal to theta to check whether it was positive
    # theta being threshold which is slightly less than the number of weights
    sum = dotproduct(x,w)
    theta = len(w) - 0.01
    s = 0
    if(sum>=theta):
        s = 1
    else:
        s = 0
    return s

def train(input, weight):
    # guess check whether or not weights give us a positive class or negative
    guess = sign(input[:-1],weight)
    # if guess matches no need for further weight recalc
    if(guess == input[-1]):
        return weight,0
    # if wrong recalc all the weights
    for i in range(len(weight)):
        weight[i] = newweightrecalc(weight[i],input[-1],guess, input[i])
    return weight, 1

def WNtrain(input, weight):
    # guess check whether or not weights give us a positive class or negative
    guess = WNsign(input[:-1],weight)
    # if guess matches no need for further weight recalc
    if(guess == input[-1]):
        return weight,0
    # if wrong recalc all the weights
    for i in range(len(weight)):
        weight[i] = WNweightcalc(weight[i],input[-1],guess, input[i])
    return weight, 1

def makebin(intarr,numdig):
    # converts int to binary of numdig digits long
    binarr = []
    for i in intarr:
        ba=[int(x) for x in list('{0:0b}'.format(i))]
        for ix in range(len(ba),numdig):
            ba = [0] + ba
        binarr = binarr + ba
    return binarr

def inversebin(arr):
    # given binary array ie [0,1,0,0] it generates the inverse [1,0,1,1]
    binarr = []
    for b in arr:
        if (b>0):
            binarr.append(0)
        else:
            binarr.append(1)
    return binarr

def addvardata(pos,neg,numofbad):
    # combines the positive  and negative data points also add irrelevant attributes
    totaldata = []
    rows = len(pos) + len(neg)
    for i in range(rows):
        col = [1]
        if (i < len(pos)):
            col = col + makebin(pos[i],numdig)
            for ib in range(numofbad):
                col.append(random.randint(0,1))
            col.append(1)
        else:
            col = col + makebin(neg[len(pos)-i],numdig)
            for ib in range(numofbad):
                col.append(random.randint(0,1))
            col.append(0)
        totaldata.append(col)
    # shuffles the data
    random.shuffle(totaldata)
    return totaldata

def WNaddvardata(pos,neg,numofbad):
    # combines the positive  and negative data points also add irrelevant attributes
    # also add the inverse attributes
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

def getdata(numbad, datasize, minv, maxv, MLA, maxruns):
    epochs = []
    errorrate = []
    error = []
    for x in range(numbad):
        points = randomgraph(minv,maxv,minv,maxv, datasize)
        line = 2
        if (random.randint(0,1)):
            side1, side2 = splitdata(points,line,'r')
        else:
            side1, side2 = splitdata(points,line,'l')
        if (MLA=='p'):
            data = addvardata(side1, side2, numbad)
            wt = genweights(data[0])

            for i in range(maxruns):
                es = 0
                for d in data:
                    wt, e = train(d, wt)
                    es = es + e
                error.append(es)




		
	

points = randomgraph(minval,maxval,minval,maxval,100)
line = genrandline(minval,maxval,minval,maxval)
pos,neg = splitdata(points,line,'r')
t = numpy.arange(-10, 10, 0.5)
#print(line)
#print(pos)
data = addvardata(pos,neg,100)
#data.append(data[0][:-1]+[int(not data[0][-1])])
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
    #print(wt)
print(error)
#print(data)
WNdata = WNaddvardata(pos,neg,100)
#WNdata.append(WNdata[0][:-1]+[int(not WNdata[0][-1])])
WNwt = genweights(WNdata[0])
WNerror = []
for i in range(niter):
    es = 0
    for d in WNdata:
        WNwt,e = train(d,WNwt)
        es = es + e
    WNerror.append(es)
    #print(WNwt)
print(WNerror)
print(data[0])
print(data[-1])
print(WNdata[0])
print(WNdata[-1])
# plotbin(data,numdig)
#print(intarr([0,1,0,0,1,1,0,1],numdig))
#plt.plot(*zip(*pos), marker='+', color='b', ls='')
#plt.plot(*zip(*neg), marker='.', color='r', ls='')
#plt.plot([line[0],line[2]],[line[1],line[3]],  color='g')
#plt.show()
