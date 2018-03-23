import numpy as np
from math import exp, sin, log, tanh
from sympy import symbols, diff, lambdify
import matplotlib.pyplot as plt
from functools import reduce
import time
import pandas as pd



##PROBLEMS
## layer size, activation fn, bias issues, complex model

## define target fn
# the nn cant learn any decreasing fn
# only monotonically insreasing fn can be learned!!
y_real_fn = lambda x: 1 if (x> 0.5) else 0#0.4*(x**2)+0.2+0.3*x*sin(4*x)#0.2+0.4*(x**2)+0.3*x*sin(15*x) #3*sin(x)#5*(x**0.2)*sin((x**1.8)*exp(-0.16*x)) #add white guassian noise

y_real_fn_and_noise = lambda x: y_real_fn(x) #+ 0.01*np.random.normal()

## helper functions
np_array_mappper = lambda X, FN: np.fromiter((FN(xi) for xi in X), X.dtype, len(X))
to_col_vect = lambda X: np.reshape(X, (-1, 1))

genDataset = lambda st, end, step, fn: (to_col_vect(np.arange(st,end,step)) , to_col_vect(np_array_mappper(to_col_vect(np.arange(st,end,step)), fn)))

data = pd.DataFrame(pd.read_csv('spambase.data', header=None, sep=','))
np_data = data.as_matrix()

total_records = np.shape(np_data)[0]
np.random.shuffle(np_data) # this will be done by k-fold eval!
train_rec = int(0.7*total_records)  # approx 70%
test_rec = total_records - train_rec

# X_train = np_data[:train_rec,:-1]
# y_train = np_data[:train_rec,-1].astype(int)

X_test = np_data[train_rec:,:-1]
y_test = np_data[train_rec:,-1].astype(int)
## get training data 
(X_train_vect, Y_train_vect) = (np_data[:train_rec,:-1], np_data[:train_rec,-1].astype(int))
#(X_test_vect, Y_test_vect) = genDataset(0.0,80.0,3.37, y_real_fn_and_noise)

# for plotting and visual comparision only, not used for training or testing
(X_real_vect, Y_real_vect) = genDataset(0.0,1.0,0.001, y_real_fn)

print("X_train:\n", X_train_vect, "\nY_train:\n", Y_train_vect)
#Y_train_dict = dict(zip(X_train_vect[:,0].tolist(), Y_train_vect[:,0].tolist()))

#plt.ion()  # turn on interactive mode
# fig = plt.figure(dpi=180)
# ax, ax2 = fig.subplots(2, 1)
# curve1,curve2 = None, None

# ax.set_xlabel('x ->')
# ax.set_ylabel('y ->')
# ax2.set_xlabel('epoch ->')
# ax2.set_ylabel('MSE ->')
# ax2.set_xlim(0, 10)
# #ax3.set_xlabel('epoch ->')
# #ax3.set_ylabel('Gen Error ->')
# curve_train = ax.scatter(X_train_vect, Y_train_vect, color='r', marker='x')
# #curve_test = ax.scatter(X_test_vect, Y_test_vect, color='g', marker='o',facecolors='none')

# #plot real data
# curve_plot = ax.plot(X_real_vect, Y_real_vect, 'b')
# curve1,curve2 = None, None
# ax.set_xlim(np.min(X_train_vect), np.max(X_train_vect))
# ax.set_ylim(np.min(Y_train_vect) -0.1, np.max(Y_train_vect) + 0.1)
# plt.draw()
# #plt.show()
# plt.pause(0.1) #plt.pause(0.1)


# one thing that you didn't do as online is that for each of neuron we have act(wT*x+b) i.e. each neuron has a bias
# associated with it so then we have many bias components!!

## need helper functions!!
def getWeightMatx(nodeList):
    print("Nodes including input are: ", nodeList)
    print("Total Layers Detected:", len(nodeList) - 1)
    # newLst = list(map(lambda x: (x, x), nodeList))
    # print(newLst)
    
    # recursive call
    def _getWeightMatx(oldLst, newLst):
        if len(oldLst) >= 2:
            # add 1 as 0th weight is for the bias term
            newLst.append((oldLst[0]+1, oldLst[1]))
            return _getWeightMatx(oldLst[1:], newLst)
        else:
            return newLst

    weightDimLst = _getWeightMatx(nodeList,[])
    print("Weight Matx Dimensions:\n", weightDimLst)
    #return list(map(lambda x: np.ones((x[0], x[1])), weightDimLst))    # for debugging only
    #return list(map(lambda x: 1.5*np.random.rand(x[0], x[1])-0.8, weightDimLst))
    return list(map(lambda x: np.random.rand(x[0], x[1]), weightDimLst))



# these values are without the bias component
# remember these are lists of 2d weight arrays but u must transpose them for computation! i.e. w(l)_T*x etc
wMatxLst = getWeightMatx([56,16,1])

print("List of weight matrices:\n\n", wMatxLst)
#time.sleep(5)

lr = 0.01 # have a learning rate

# def activ fn
reluFn = lambda x: x if (x > 0) else (0.01*x)#tanh(x)#x if (x > 0) else (0.01*x)#x#log(exp(x)+1) #x if (x > 0) else (0.01*x)

#def activ fn deriv
reluFnDeriv =lambda x: 1 if (x > 0) else -0.01#(1 - tanh(x)**2)#1 if (x > 0) else 0.01 #1 #1/(1+exp(-x))#x*(1-x)#1 if (x > 0) else -0.01

sigmoid = lambda x: 1/(1+exp(-x))
sigmoidDeriv = lambda x: sigmoid(x)*(1-sigmoid(x))

identityFn = lambda x: x

#square error
# x_l = h when l=L i.e. last layer, this only applies to last layer
errFn = lambda x_L, y: (x_L - y)**2

errFnDeriv = lambda x_L, s_L, y: 2*(x_L - y)*(sigmoidDeriv(s_L)) #final layer derivative of actFn is 1 as actFn is identity

actFnLst = [reluFn, sigmoid]

print("ReluTest x = 5 =>", reluFn(5), "; x = -5 =>", reluFn(-5))

def remPlot(plotHandle):
    if plotHandle is not None:
        plotHandle.remove()
    return None


epochs = 300

def calcXLst(inputVect, lstOfWMatx, lstOfActFn):
            #lstOfWMatxAndActFn = zip(lstOfWMatx, lstOfActFn)
            #input_S_l_and_X_l_lst = ([inputVect], [inputVect])
            
            def _calcXLst(listOfWeightMatx, theActFnLst, SLst, XLst):
                # compute output of current layer, use last element in layer list
                # this is S_l = w_l * X_(l-1)   # the second index is for tuples below
                S_l = np.matmul(listOfWeightMatx[0].T, XLst[-1])

                
                # get output by passing through act fn if need be 
                X_l = to_col_vect(np_array_mappper(S_l, theActFnLst[0]))

                if len(listOfWeightMatx) > 1 :
                        # add bias placeholder => [1] and recursively calc output of next layer
                        SLst.append(to_col_vect(np.insert(S_l, 0, [1])))
                        XLst.append(to_col_vect(np.insert(X_l, 0, [1])))
                        return _calcXLst(listOfWeightMatx[1:], theActFnLst[1:], SLst, XLst)
                else: 
                        SLst.append(S_l)    # append final y_scalar, without bias
                        XLst.append(X_l)    # activate final y_scalar
                        return (SLst, XLst) 
            return _calcXLst(lstOfWMatx, lstOfActFn, [inputVect], [inputVect])

for iter in range(epochs):
    np.random.shuffle(X_train_vect) #shuffle training vector every epoch
    #print(X_train_vect)

    # toplot training results
    fullHypEval = []
    actualY = []
    
    #loop through each sample in shuffled array because we doing SGD
    for n in range(0, len(X_train_vect)):
        # n => indices 1 .. N for all N samples
        # calc X_l for all layers: 1 <= l <= L
        #fwd pass
        
        
        x_n = X_train_vect[n]    #convert to scalar

        ##print("***n_value: ", n, "\t\t\t******x_val: ", x_n)

        #a list of all nodes evaluated
        all_S_lst, all_X_lst = calcXLst(to_col_vect(x_n), wMatxLst, actFnLst) #result is 1x1 need to convert to scalar
        h_n = all_X_lst[-1] #convert to scalar # get last value i.e. output
        y_n = Y_train_vect[n]
        fullHypEval.append(h_n) # to be used later for plotting
        actualY.append(y_n)
        #print("Train => ( x:", x_n,"; h:", h_n, "; y_n: ", y_n, ")")

        ##print("Without Act: \n", all_S_lst, "\n\nWith Act: \n", all_X_lst)
        

        # perform backprop recursion
        # calc delta_l for all layers: L >= l > 1
        # this will be done in REVERSE, so last element in output list will be delta for first layer
        def calcDeltaLst(wMatxLst, SLst, XLst, currInput):
            # first calc top-level error deriv i.e. error of last layer or output

            delta_L = to_col_vect(errFnDeriv(XLst[-1], SLst[-1], y_n))
            ##print("error delta/deriv at last layer", delta_L)
            #def _calcDeltaLst

            ## also do regul at a later stage

            def _calcDeltaLst(currWMatxLst, SLst, deltaLst):
                # l =>  layer: "l"
                # deleta_current_lth layer vector

                # each new delta_vector will be appened at end of list
                # so as you go towards lower layers the correct way to get the last computer layer
                # aka layer l will be using the '-1' index
                # after recursion this list will be reversed
                delta_curr_l_vector = deltaLst[-1]
                ##print("delta_curr_l vector: ", delta_curr_l_vector)

                # need to do W *MATMUL* delta_vector
                # then result *MATMUL* deriv_vector_transposed (called theta_prime_vector)
                # then get Trace of result

                if len(currWMatxLst) > 1 :
                    # get the last weight matx
                    W = currWMatxLst[-1][1:,:]
                    s_vector = SLst[-2][1:,:] # we need the unactivated outputs of the previous layer

                    # we need to remove the bias at this stage cause there is no point of
                    # computing delta for bias, this is for delta_curr_l where
                    # where 1 <= j < d so BIAS NOT INCLUDED in DELTA_J_vector!!

                    # need to remove bias's from current layer i.e. current output layer
                    # as they are no weight connection to them on the lower end or their left side
                        # this is a risky move but if thr is only +1 difference then its most likely
                        # due to bias, we can also check bias values
                        # removing this now will help when computing gradients
                    #s_vector = s_vector[1:,:]   #remove top row
                    
                    # because deltas are always used as j's removing delta_0 for ith version is still ok
                    # it will only be needed as the jth index so its ok
                    
                    ########
                    result = np.matmul(W, delta_curr_l_vector)

                    # derivative of activation fn w.r.t. each of 's' for last layer nodes
                    if np.shape(s_vector)[0] == 1:
                        #print("1 shape")
                        theta_prime_vector = s_vector
                    else: theta_prime_vector = to_col_vect(np_array_mappper(s_vector, reluFnDeriv))
                    

                    # calc the hadamard product which is the element-wsie multiplication of 
                    # two vectors so each elem in result is mult theta' to produce
                    # delta_l-1 = theta' * w_vector^T * delta_l
                    delta_prev_l_vector = np.multiply(result, theta_prime_vector) # calculating backward

                    deltaLst.append(delta_prev_l_vector)

                    # as you go to prev layers, remove unneeded vectors from end for purpose of this fn
                    return _calcDeltaLst(currWMatxLst[:-1], SLst[:-1], deltaLst)
                else:
                    return deltaLst
            
            deltaLst = _calcDeltaLst(wMatxLst, SLst, [delta_L])
            deltaLst.reverse()
            ##print("deltaLst:\n\n", deltaLst)
            return deltaLst

        all_delta_lst = calcDeltaLst(wMatxLst, all_S_lst, all_X_lst, x_n)

        for i in range(0, len(wMatxLst)):
            delta_vector = all_delta_lst[i]
            x_vector = all_X_lst[i]
            #print("x vector", x_vector)
            #time.sleep(2)
            mult = np.matmul(x_vector, delta_vector.T)
            redMult = lr*mult
            wMatxLst[i] -= redMult
        
        # def updWeights (wMatxLst, deltaLst, XLst):
        #     def _updWeights (oldWMatxList, newWMatxList, deltaLst, XLst):
        #         if len(oldWMatxList) > 0:
        #             Wnew = oldWMatxList[0]
        #             delta_vector = deltaLst[0]
        #             x_vector = XLst[0]
        #             mult = np.matmul(x_vector, delta_vector.T)
        #             #print(mult)
                    
        #             Wnew = Wnew - lr*mult#np.matmul(x_vector, delta_vector.T)
        #             newWMatxList.append(Wnew)
        #             return _updWeights(oldWMatxList[1:], newWMatxList, deltaLst[1:], XLst[1:])
        #         else:
        #             return newWMatxList

        #     return _updWeights(wMatxLst, [], deltaLst, XLst)
        
        #print("Old weights:\n\n", wMatxLst)
        # wMatxLst = updWeights(wMatxLst, all_delta_lst, all_X_lst)
        #print("\n\nnew weight matx:\n\n", wMatxLst)

    #hAndyLst = zip(fullHypEval, actualY)
    #print("FUllhyp\n\n: ", fullHypEval)
    r = np.count_nonzero(Y_train_vect != fullHypEval)
    print(r)
    #break
    #errLst = list(map((lambda node: errFn(node[0], node[1])), hAndyLst))    
    #mse = np.average(errLst)
    # print("Epoch: ", iter, "\t\t***** MSE ******:", mse)

    # curve1,curve2 = remPlot(curve1), remPlot(curve2)
    # curve1 = ax.scatter(X_train_vect, fullHypEval, color='c', marker='o')
    # #curve2, = ax.plot(X_outsample, H_eval_out, 'b:')
    # plt.draw()
    # plt.pause(0.01)

# plt.ioff()    
# plt.draw()
# plt.show()    