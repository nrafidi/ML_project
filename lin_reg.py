import numpy
import glob
#import danny's codes

def lin_reg_train(directory)

    
    X = []
    Y = []
    for filename in glob.glob(directoryr + '/*.txt'):
        with open(filename) as f
            s = f.readline()
            
            #We need to discuss how these files are formatted
            ss = s.split('" (')

            #Here I call Danny's code. What is it returning?
            X.append(danny(ss[0] + '"'))

            y = [];

            for i in ss[1]
                if (i != ',' and i != ' ' and i != ')')
                    y.append(int(i))
                    
            Y.append(y)
            
    numex = len(X)
    errL = [] #make numpy array
    for lam in range(0:1000:1)
        err = 0;
        for j in range(0:(numex - 1))
            testX = X[j]
            testY = Y[j]

            trainX = X.pop(j)
            trainY = Y.pop(j)

            X.insert(j, testX)
            Y.insert(j, testY)


            #figure out how to use numpy
            #M = trainXT*trainX + lam*I (I is size of feature space)
            #Invert M try: except LinAlgError:
            #W = Minv*trainXT*trainY

            guessY = dot(testX, W)

            if !(guessY == testY).all():
                err += 1
            
        err  = err/num_x
        errL.append(err)

    L = errL.argmin()

    #get W and return it

def lin_reg_test(W, fname)
    X = []
    Y = []
    f = open(fname, 'r')
    s = f.readline()
    
    #We need to discuss how these files are formatted
    ss = s.split('" (')

    #Here I call Danny's code. What is it returning?
    X.append(danny(ss[0] + '"'))

    y = [];

    for i in ss[1]
        if (i != ',' and i != ' ' and i != ')')
            y.append(int(i))
            
    Y.append(y)

    guessY = dot(X, W)

    return [Y, guessY]
    
