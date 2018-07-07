import numpy as np
import matplotlib.pyplot as plt
import dataImporter



nsample = 50
epochs = 10

X = np.random.rand(nsample, 2)
Y = np.zeros((nsample, 1))


for i in range(nsample):
    minVal = X[i, 0] + 0.15
    maxVal = X[i, 0] + 0.3

    if (X[i, 1] > maxVal):
        Y[i] = -1
    elif (X[i, 1] > minVal and X[i, 1] < maxVal):
        Y[i] = 0
    else:
        # X[i,1] < minVal
        Y[i] = 1


# np.where returns a tuple, first part is all the indices, second part is condition.nonzero() which is a bit weird
# we just need first part of tuple
X = X[np.where(Y != 0)[0], :]
Y = Y[np.where(Y != 0)[0]]

#print("X:\n", X, "\nY:\n", Y)


plt.ion()  # turn on interactive mode
fig = plt.figure(dpi=180)
ax = fig.add_subplot(111)
ax.set_xlabel('x1: Feature 1')
ax.set_ylabel('x2: Feature 2')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
#plt.show()

# update total no of sample
nsample = Y.shape[0]

# transform the matrices
X = np.transpose(X)
# add last row of x2 values which will be multiplied by the threshold value i.e. ...
# sum(x0*w0 + x1*w1 + x2*w2) = 0 will give w2 = "- threshold" only when x2=1 for all samples
X = np.array([X[0], X[1], np.ones(X.shape[1])])
Y = np.transpose(Y)

print("X:\n", X, "\nY:\n", Y)
print("Sizes, X: ", np.shape(X), "Y: ", np.shape(Y), "nsample: ", nsample)

# weight vector, by defining it this way, we are defining a col vector or 3x1 matrix
w = np.array([[0.01], [0.01], [0]])

# get all x points corresponding to positive Y
Xpos = X[:, np.where(Y[0] > 0)[0]]
# get all x points corresponding to negative Y
Xneg = X[:, np.where(Y[0] < 0)[0]]

ax.scatter(Xpos[0, :], Xpos[1, :], marker='o',
           facecolors='none', edgecolors='g', s=80)
ax.scatter(Xneg[0, :], Xneg[1, :], color='r', marker='x', s=80)
#fig.canvas.draw()
line, = ax.plot([0, 1], [-(w[2]/w[1]), -((w[0]+w[2])/w[1])], 'b')

sq = None

plt.draw()

plt.show()

w_old = np.copy(w)
# main loop
for i in range(epochs):
    for ii in range(nsample):
        # cycle through each sample

        # boundary is where w0*x0+w1*x1=-w2 where w2 is -thresh and x2 =1 for all x
        # eq is x1 = (-w0/w1)*x0 - (w2/w1)
        # note this is the BOUNDARY LINE, ideally no point should lie there i.e. linearly separable
        # for each point ii: y_i = sign(w_Transposed <DOT> x_iith_col) => (1x3) * (3x1) => (1x1)
        # when we do dot prod we get ||w_vec||*||x_vec||*cosTheta
        # only cosTheta gives sign:
        #   0 < Theta < 90deg => neg
        #   90deg < Theta < 180deg => pos
        # So if LHS i.e. Y[i] != RHS i.e. the sign of the result of dot product
        # Then we add/subt X_ii_vect from w_vect depending if Y is neg or pos
        # i.e. w_new = w +- x
        # which makes Theta from obtuse -> acute or vice versa
        # this classifies THAT point as correctly, but this is bad as it affects all weights so...
        # if data is too close to itself, there will always be some point(s) which will be wrong and
        # w will keep toggling between them

        # line given by 2 points:
        # lowest-point: [x0=0 ; x1= -(w2/w1)]
        # highest-point [x0=1 ; x1= -(w0+w2)/w1]

        # remove previous lines and squares
        def remPlot(plotHandle):
            if plotHandle is not None:
                plotHandle.remove()
            return None

        line = remPlot(line)
        sq = remPlot(sq)
        # plot type returns a tuple we only need first handle!
        # give all x points as array and all y points as array
        line, = ax.plot([0, 1], [-(w[2]/w[1]), -((w[0]+w[2])/w[1])], 'b')

        # IMPORTANT: Its tricky to correctly extract columns of X as a column vector!!
        # python by default using range slicing returns a ROW VECTOR
        # A col vector in python is a NESTED row vector with each element being 1x1 sub-array
        # i.e. top level array contain multiple elements as ROWS and inner arrays contain COLS
        # so to extract a col from X do X[:, [ii] => [[X0], [X1], [X2]]
        # This above^ is a 3 row cause 3 outer elements and 1 col matrix cause 1 inner element each
        # so this is a col vector..
        # Simply X[:, ii] => [X0, X1, X2] which is a ROW VECTOR!!
        if np.sign(np.matmul(np.transpose(w), X[:, [ii]])) != Y[0, ii]:
            w = w + Y[0, ii]*(X[:, [ii]])
            sq, = ax.plot(X[0, ii], X[1, ii], 'bs')
            plt.pause(.1)
        #av

        print("\r                                     \r", "Epoch: ", i+1, " of ", epochs, "\tSample: ", ii+1, " of ", nsample, end='',flush=True)
        plt.draw()
        plt.pause(.001)
        #pass
    if np.array_equal(w, w_old):
        break
    else:
        w_old = np.copy(w)

print("\nFinal w:\n", w_old, "\nPlease close plot window to exit.")

plt.ioff()
line, = ax.plot([0, 1], [-(w[2]/w[1]), -((w[0]+w[2])/w[1])], 'b')
plt.draw()
plt.show()