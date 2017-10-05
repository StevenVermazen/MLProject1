import numpy, random
import matplotlib.pyplot as plt

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
    p1 = [random.randint(newminX,newmaxX),random.randint(newminY,newmaxY)]
    p2 = [random.randint(newminX,newmaxX),random.randint(newminY,newmaxY)]
    while (p1 == p2):
        p2 = [random.randint(newminX, newmaxX), random.randint(newminY, newmaxY)]
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

points = randomgraph(-10,10,-10,10,40)
line = genrandline(-10,10,-10,10)
pos,neg = splitdata(points,line,'r')
t = numpy.arange(-10, 10, 0.5)
#print(line)
#print(points)
print(pos,neg)
plt.plot(*zip(*pos), marker='+', color='b', ls='')
plt.plot(*zip(*neg), marker='.', color='r', ls='')
#plt.plot([line[0],line[2]],[line[1],line[3]],  color='g')
plt.show()
