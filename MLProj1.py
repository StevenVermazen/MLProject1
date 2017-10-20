import numpy, random
import matplotlib.pyplot as plt

eta = 0.08
numdig = 8
minval,maxval = 0, 2**numdig-1
niter = 100
alpha = 2

def randomgraph(relattr, pointNum):
	arr = []
	for r in range(0, pointNum):
		col = []
		for c in range(0, relattr):
			col.append(random.randint(0,1))
		arr.append(col)
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

def dotproduct(x,w):
    # getting the sum of form multipling two vectors
    s = 0
    for i in range(len(x)):
        s = s + x[i] * w[i]
    return s

def sign(x,w):
    # if dotproduct is greater than or equal to 0 to check whether it was positive
    sm = dotproduct(x,w)
    s = 0
    if(sm>=0):
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

def train(inp, weight):
    # guess check whether or not weights give us a positive class or negative
    guess = sign(inp[:-1],weight)
    # if guess matches no need for further weight recalc
    if(guess == inp[-1]):
        return weight,0
    # if wrong recalc all the weights
    for i in range(len(weight)):
        weight[i] = newweightrecalc(weight[i],inp[-1],guess, inp[i])
    return weight, 1

def WNtrain(inp, weight):
    # guess check whether or not weights give us a positive class or negative
    guess = WNsign(inp[:-1],weight)
    # if guess matches no need for further weight recalc
    if(guess == inp[-1]):
        return weight,0
    # if wrong recalc all the weights
    for i in range(len(weight)):
        weight[i] = WNweightcalc(weight[i],inp[-1],guess, inp[i])
    return weight, 1

def splitdata(points, posside):
    #this breaks the data
    #points is all the randomly generated points
    #line is a list with 4 digits [x1,y1,x2,y2]
    w = []
    for i in range(len(points[0])):
        w.append(random.uniform(-1,1))
    pointsA = []
    pointsB = []
    #2 special cases needed to check first being if x1 = x2
    #if so slope is infinite and you cant calc m
    for p in points:
        if(sign(p,w)):
            pointsA.append(p)
        else:
            pointsB.append(p)
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

def getdata(numbad, datasize, relattr, MLA, maxruns, testa):
    epochs = []
    errorrate = []
    error = []
    terr = []
    points = randomgraph(relattr, datasize)
    if (random.randint(0,1)):
        side1, side2 = splitdata(points,'r')
    else:
        side1, side2 = splitdata(points,'l')
    for x in range(numbad+1):
        tesnum = 0.0
        err = []
        eps = 0
        if (MLA=='p'):
            dw = addvardata(side1, side2, x)
            data = dw[:-testa]
            wt = genweights(data[0])
            for i in range(maxruns):
                es = 0
                for d in data:
                    wt, e = train(d, wt)
                    es = es + e
                eps+=1
                err.append(es)
                if(es==0): break
            for it in range(testa-1):
                if(sign(dw[-testa+it][:-1],wt) != dw[-testa+it][-1]):
                    tesnum = tesnum + 1
            terr.append(tesnum/testa)
        else:
            dw = WNaddvardata(side1, side2, x)
            data = dw[:-testa]
            wt = WNgenweights(data[0])
            for i in range(maxruns):
                es = 0
                for d in data:
                    wt, e = train(d, wt)
                    es = es + e
                eps+=1
                err.append(es)
                if(es==0): break
            for it in range(testa-1):
                if(WNsign(dw[-testa+it][:-1],wt) != dw[-testa+it][-1]):
                    tesnum = tesnum + 1
            terr.append(tesnum/testa)
        epochs.append(eps)
        error.append(err)

    for ep in error:
        erate=[]
        for e in ep:
            erate.append(e/datasize)
        errorrate.append(erate)
    return epochs, errorrate, terr

def mean(a):
	ss=[0]*len(a[0])
	for s in a:
		ss = [x + y for x, y in zip(ss, s)]
	for i in range(len(ss)):
		ss[i] = ss[i]/len(a[0])
	return ss

irrel = 6
dsize = 300
rell = 10
maxruns = 1000
testam = 30
erave =[]
epave =[]
tave = []
for i in range(200):
    epochnum, erate, te = getdata(irrel, dsize, rell,'w', maxruns, testam)
    erave.append(erate)
    epave.append(epochnum)
    tave.append(te)
    print(i)
#print(te)
ta = mean(tave)
#Wepochnum, Werate, wte = getdata(irrel, dsize, rell,'W', maxruns, testam)
#print(ta)
'''
aveerr=[]
for e in erate:
    aveerr.append(sum(e[:-1])/(len(e)-1))
'''
t=numpy.arange(0,len(ta),1)
#t2=numpy.arange(0,len(wte),1)
plt.plot(t,ta,'bo',t,ta,'b-')
plt.title('Error rate on test data WINNOW')
plt.ylabel('Error rate')
plt.xlabel('Irrelevant attributes')
plt.show()





