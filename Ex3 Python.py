
"""
Created on Thu Feb 20 12:09:54 2020

@author: angus
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import statistics as st

detector1 = 0.55
detector2 = 0.6
detector3 = 0.85
detector4 = 0.5


def sinAnal():
    
    """
    This function uses the analytical method to generate a random number
    between 0 and pi with a distribution proportional to sin(x).
    """
    
    randNum = np.random.random()
    sinNum = np.arccos(1 - 2*randNum)                                           # Found by integrating sin(x) and finding the inverse (including normalisation).
    
    return sinNum


def sinAR():
    
    """
    This function uses the accept/reject method to generate a random number
    between 0 and pi with a distribution proportional to sin(x).
    """
    
    randNum = np.random.random()
    randPi = np.pi * np.random.random()
    
    if randNum < np.sin(randPi):
        return randPi
    else:
        return sinAR()                                                          # Carry out the function again if the number doesn't fit the distribution.


def sinDistAnal(numVal):
    
    """
    This function iterates the analytical random number function to fill
    a list with valid random numbers that can then be plotted.
    """

    sinListAnal = []
    
    for i in range(int(numVal)):
        sinListAnal.append(sinAnal())
    
    return sinListAnal
    

def sinDistAR(numVal):
    
    """
    This function iterates the accept/reject random number function to fill
    a list with valid random numbers that can then be plotted.
    """
    
    sinListAR = []
    
    for i in range(int(numVal)):
        sinListAR.append(sinAR())
    
    return sinListAR


def histPlot(numVal):
    
    """
    This function plots the sine distributions for both the analytical and 
    accept/reject methods as histograms.
    """
    
    sinListAnal = sinDistAnal(numVal)
    sinListAR = sinDistAR(numVal)
    
    plt.hist(sinListAnal,50,density = True)
    plt.title("Sine distribution by analytical method")
    plt.show()
    
    plt.hist(sinListAR,50,density = True)
    plt.title("Sine distribution by accept/reject method")
    plt.show()
 
      
numVal = 10000 
"""
histPlot(numVal)
"""

###############################################################################

"""
This section of code measures the difference between the sine curve generated
by the two random number methods and numpy sine, with a varying number of
random numbers generated per sine curve, repeated 5 times and averaged. 
This is to see how the accuracy of each method changes with sample size. 
"""

"""
numVals1 = np.linspace(1000,10000,100)                                           # Starting with a sine distribution consisting of 10 random numbers, going up to one with 10,000, with gaps of 10.
avAnal = []
avAR = []

print("This will take about 25 seconds.")


for j in range(100):
    
    averageAnal = []
    averageAR = []
    diffAnal = 0
    diffAR = 0
    histAnal = np.histogram(sinDistAnal(numVals1[j]),200,density = True)        # Calculating histograms for each method to be compared against numpy sine.
    histAR = np.histogram(sinDistAR(numVals1[j]),200,density = True)
    
    for average in range(5):
    
        for i in range(200):
            
            sinHistAnal = np.sin(histAnal[1][i]) / 2
            sinHistAR = np.sin(histAR[1][i]) / 2
            diffAnal += abs(histAnal[0][i] - sinHistAnal)/sinHistAnal               # Keeping track of the difference for each curve so the average can be taken.
            diffAR += abs(histAR[0][i] - sinHistAR)/sinHistAR
        
        averageAnal.append(diffAnal/200)                                             # Calculating the average.
        averageAR.append(diffAR/200)
    
    meanAnal = st.mean(averageAnal)
    meanAR = st.mean(averageAR)    
    
    avAnal.append(meanAnal)
    avAR.append(meanAR)
 
plt.plot(numVals1,avAnal,label = "Analytical method")
plt.plot(numVals1,avAR,label = "Accept/reject method")
plt.xscale("log")
plt.xlabel("Number of random numbers generated")
plt.ylabel("Average fractional difference")
plt.title("Average difference between random and numpy sine versus number of iterations")
plt.legend()
plt.show()
"""

###############################################################################

"""
This section of code measures the amount of time that each process takes to
produce a sine distribution, and compares that with a varying number of
iterations. This is also averaged over 5 iterations. This is to show how 
the computing time changes with the number of iterations.
"""

"""
timeAnal = []
timeAR = []
numVals2 = np.linspace(100000,1000000,10)
errorAnal = []
errorAR = []
tAnal = []
tAR = []

print("This will take a short while.")

for j in range(10):
    for average in range(5):
        startAnal = time.process_time()
        sinDistAnal(numVals2[j])
        timeAnal.append(time.process_time() - startAnal)
        startAR = time.process_time()
        sinDistAR(numVals2[j])
        timeAR.append(time.process_time() - startAR)
    
    errorAnal.append(st.stdev(timeAnal))
    errorAR.append(st.stdev(timeAR))
    tAnal.append(st.mean(timeAnal))
    tAR.append(st.mean(timeAR))

plt.errorbar(numVals2,tAnal,yerr=errorAnal,label = "Analytical method")
plt.errorbar(numVals2,tAR,yerr=errorAR,label = "Accept/reject method")
plt.xlabel("Number of random numbers generated")
plt.ylabel("Time (s)")
plt.title("Time taken to generate sine distribution versus number of iterations")
plt.legend()
plt.show()
"""

###############################################################################


def cosMu():
    
    """
    This function generates the cos^2 distribution for the incident muons 
    using the accept/reject method, with a normalisation term 4/pi.
    """
    
    randNum = np.random.random()
    randPi = np.random.uniform(0,np.pi/2)
    
    if randNum <= (4/np.pi) * (np.cos(randPi)**2):                              # Random value for theta using accept/reject.
        return randPi
    else:
        return cosMu()
    
    
def phiMu():
    
    """
    This function generates a uniform distribution of random numbers from 0 to
    2 pi.
    """
    
    randPhi = np.pi*2*np.random.random()
    
    return randPhi
    

def initialXY():
    
    """
    This function generates a uniform distribution over the area of the
    detector.
    """
    
    randX = 20 * np.random.random()                                             # Random x and y coordinates from 0 to 20.
    randY = 20 * np.random.random()
    
    return randX,randY


def initialPosition():
    
    """
    This function calculates the direction of the incident muon. It calculates
    the position of the muon every 0.25cm into the detector, and checks that
    the muon hasn't left the detector.
    """
    
    theta = cosMu()
    phi = phiMu()
    Xi,Yi = initialXY()
    layerWidths = [4.25,1,4.25,1,4.25,1,4.25]                                   # Thickness of each layer.
    pos = 0
    X = [Xi]                                                                    # Initial X, Y, and Z values.
    Y = [Yi]
    Z = [pos]
    
    for i in range(7):
        pos += layerWidths[i]
        X.append(Xi + pos * np.tan(theta) * np.cos(phi))
        Y.append(Yi + pos * np.tan(theta) * np.sin(phi))
        Z.append(pos)
        
        if X[-1] > 20 or Y[-1] > 20 or X[-1] < 0 or Y[-1] < 0:
            X.pop()
            Y.pop()
            Z.pop()
            break
                
    return X,Y,Z,theta,phi


def detection_m():
    
    """
    This function determines if the muon is detected by each of the four
    scintillators. It also finds if the muon is stopped by a copper layer.
    """
    
    X,Y,Z,theta,phi = initialPosition()
    muon_detections = [0] * 4                                                   # Keeps track of which layer the muon is detected in.
    CuProb = 5e-3/np.cos(theta)                                                 # Probability of the muon stopping in copper.
    probs = [detector1,CuProb,detector2,CuProb,detector3,CuProb,detector4]
    Xe_pos,Ye_pos,Ze_pos = [],[],[]
    detections_e = 0
    
    for layer in range(7):
        
        if layer >= len(X) or layer >= len(Y):
            break
        
        if np.random.random() <= probs[layer]:
            
            if layer == 0:                                                      # Even indices correspond to scintillators.
                muon_detections[0] += 1
                
            if layer == 2:
                muon_detections[1] += 1
                
            if layer == 4:
                muon_detections[2] += 1
                
            if layer == 6:
                muon_detections[3] += 1
                
            if layer % 2 != 0:                                                  # Odd indices correspond to copper layers.
            
                X = X[:layer+1]
                Y = Y[:layer+1]
                Z = Z[:layer+1]
            
                if len(X) == 1 or len(X) == 3 or len(X) == 5:                   
                    break
                else:
                    Xe_pos,Ye_pos,Ze_pos,detections_e = detection_e(X[layer],Y[layer],Z[layer])
                    break
            
    return X,Y,Z,Xe_pos,Ye_pos,Ze_pos,detections_e,muon_detections


def electronAngle():
    
    """
    This function generates a uniform distribution over a sphere. The reason 
    for alpha having a sine distribution is that, if alpha was uniform from 
    0 to pi, there would be an abundance of numbers generated at the poles.
    Hence a sinusoidal distribution is required.
    """
    
    randNum = np.random.random()
    alpha = np.arccos(1 - 2*randNum)                                            # Alpha is the vertical angle (similar to theta for the muon, but goes from 0 to pi).
    beta = 2*np.pi * np.random.random()

    return alpha,beta 
   

def electronDir(Xe0,Ye0,Ze0):
    
    """
    This function calculates the direction of the emitted electron. Like the 
    muon, the position is calculated at every 0.25cm height in the detector.
    There is also a condition for if the electron leaves the detector.
    """
    
    alpha,beta = electronAngle()
    Xe = [Xe0]                                                                  # Initial X, Y, and Z values of the electron (the end point of the muon).
    Ye = [Ye0]
    Ze = [Ze0]
    

    for i in range(1,81):                                                       # There are 80 steps of 0.25cm each in the 20cm detector.
            
        if alpha > np.pi/2:                                                     # Electron points down (positive z).
            pos = Ze[0] + 0.25 * i
        else:
            pos = Ze[0] - 0.25 * i                                              # Electron points up (negative z).
        
        Xe.append(Xe[0] + abs(pos-Ze[0]) * np.tan(alpha) * np.cos(beta))
        Ye.append(Ye[0] + abs(pos-Ze[0]) * np.tan(alpha) * np.sin(beta))
        Ze.append(pos)
        
        if Xe[-1] > 20 or Ye[-1] > 20 or Xe[-1] < 0 or Ye[-1] < 0 or Ze[-1] < 0 or Ze[-1] > 20:
            Xe.pop()
            Ye.pop()
            Ze.pop()
            break
                  
    return Xe,Ye,Ze,alpha


def electron_absorption(dis,alpha):
    
    """
    This function updates the distance the electron has travelled to check if
    it has been absorbed by the copper (after the 1.8cm maximum path length.)
    """
    
    dis = dis + 1/np.cos(alpha)                                                 # Updates the distance travelled by the electron.
    absorption_e = False
    if dis >= 1.8:
        absorption_e = True
    if dis < 1.8:
        absorption_e = False
    
    return absorption_e,dis


def detection_e(Xe0,Ye0,Ze0):
    
    """
    This function determines if the emitted electron is detected and/or 
    absorbed in each layer of the detector.
    """
    
    Xe,Ye,Ze,alpha = electronDir(Xe0,Ye0,Ze0)
    
    detections_e = [0] * 4                                                      # Keeps track of which layer the electron is detected in.
    dis = 0
    test1,test2,test3,test4 = False,False,False,False                           # Keeps track of if the electron has entered the scintillator once. 
    
    for height in Ze:
        
        if height <= 4.25 and test1 is False:
            
            test1 = True
            if np.random.random() <= detector1:
                detections_e[0] += 1
                
        if 4.25 < height <= 5.25:
            
            absorption_e,dis = electron_absorption(dis,alpha)
            
            dis = dis + 1/np.cos(alpha)
            absorption_e = False
            if dis >= 1.8:
                absorption_e = True
            if dis < 1.8:
                absorption_e = False
            
            if absorption_e is True:
                Xe = Xe[:2]                                                     # Truncates the electron path if it is absorbed.
                Ye = Ye[:2]
                Ze = Ze[:2]
                break
            
        if 5.25 < height <= 9.25 and test2 is False:
            
            test2 = True
            if np.random.random() <= detector2:
                detections_e[1] += 1
                
        if 9.5 < height <= 10.5:
            
            absorption_e,dis = electron_absorption(dis,alpha)
            
            if absorption_e is True:
                Xe = Xe[:4]
                Ye = Ye[:4]
                Ze = Ze[:4]
                break
            
        if 10.5 < height <= 14.75 and test3 is False:
            
            test3 = True
            if np.random.random() <= detector3:
                detections_e[2] += 1
                
        if 14.75 < height <= 15.75:
            
            absorption_e,dis = electron_absorption(dis,alpha)
            
            if absorption_e is True:
                Xe = Xe[:6]
                Ye = Ye[:6]
                Ze = Ze[:6]
                break
            
        if 15.75 < height <= 20 and test4 is False:
            
            test4 = True
            if np.random.random() <= detector4:
                detections_e[3] += 1
                
    return Xe,Ye,Ze,detections_e


def plot():
    
    """
    This function plots the muon path and, if applicabale, electron path.
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0,20)
    ax.set_ylim(0,20)
    ax.set_zlim(20,0) 
    X,Y,Z,Xe,Ye,Ze,electron_detections,muon_detections = detection_m()
    ax.plot(X,Y,Z,label = "Muon path")
    ax.plot(Xe,Ye,Ze,label = "Electron path")
    plt.legend()
    plt.show()

"""
plot()
"""

###############################################################################


"""
This section of code checks that the distribution of muons and electrons 
matches what is required. It generates a scatter plot of x, y, and z
coordinates corresponding to randomly generated angles.
"""

"""
fig = plt.figure()
fig_e = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax_e = fig_e.add_subplot(111, projection='3d')

x = []                                                                          # Muon position in cartesian coordinates.
y = []
z = []

xe = []                                                                         # Electron position in cartesian coordinates.
ye = []
ze = []

r = 10                                                                          # Distance from centre to point.


for i in range(5000):
    
    alpha,beta = electronAngle()
    theta = cosMu()
    phi = phiMu()
    
    x.append(r*np.sin(theta)*np.cos(phi))                                       # Converting polar coordinates to cartesian.
    y.append(r*np.sin(theta)*np.sin(phi))
    z.append(-r*np.cos(theta))
    
    xe.append(r*np.sin(alpha)*np.cos(beta))
    ye.append(r*np.sin(alpha)*np.sin(beta))
    ze.append(r*np.cos(alpha))

ax.scatter(x,y,z,s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Muons")

ax_e.scatter(xe,ye,ze,s=1)
ax_e.set_xlabel('X')
ax_e.set_ylabel('Y')
ax_e.set_zlabel('Z')
ax_e.set_title("Electrons")
plt.show()
"""

###############################################################################

"""
This section of code plots bar charts for how many muons each scintillator
detected, which scintillators each muon was detected by, and how many
electrons were detected by each scintillator.
"""


av = []
av_count = []
av_e = []
error = []
error_count = []
error_e = []
absorptions = 0
muons = 1000                                                                    # Number of muons entering the detector.
MDetections = np.zeros((4,5))
detections = np.zeros((5,5))
e_detections = np.zeros((4,5))

def noScintillator(muon_detections):                                            # Sums the total number of detections to calculate how many scintillators each muon is detected by.
    
    count = muon_detections[0]+muon_detections[1]+muon_detections[2]+muon_detections[3]
    
    return count

for j in range(5):                                                              # Repeats 5 times and calculates an average and standard deviation so error bars can be plotted.
    
    for i in range(muons):
    
        X,Y,Z,Xe_pos,Ye_pos,Ze_pos,detections_e,muon_detections = detection_m()
        Xe,Ye,Ze,electron_detections = detection_e(X[-1],Y[-1],Z[-1])
        
        if noScintillator(muon_detections) == 0:                                # How many scintillators each muon is detected by.
            detections[0][j] += 1
        if noScintillator(muon_detections) == 1:
            detections[1][j] += 1
        if noScintillator(muon_detections) == 2:
            detections[2][j] += 1
        if noScintillator(muon_detections) == 3:
            detections[3][j] += 1
        if noScintillator(muon_detections) == 4:
            detections[4][j] += 1
        
        if electron_detections != [0,0,0,0]:                                    # Determines if a muon was absorbed.
            absorptions += 1
        e_detections[0][j] += electron_detections[0]
        e_detections[1][j] += electron_detections[1]
        e_detections[2][j] += electron_detections[2]
        e_detections[3][j] += electron_detections[3]
        
        MDetections[0][j] += muon_detections[0]
        MDetections[1][j] += muon_detections[1]
        MDetections[2][j] += muon_detections[2]
        MDetections[3][j] += muon_detections[3]
        
absorptions /= 5                                                                # Average number of absorptions.
    
for i in range(4):
    av.append(st.mean(MDetections[i])/muons)                                    # Avergage number of detections and standard deviation.
    error.append(st.stdev(MDetections[i])/muons)
    av_e.append(st.mean(e_detections[i])/absorptions)
    error_e.append(st.stdev(e_detections[i])/absorptions)
    
for i in range(5):
    av_count.append(st.mean(detections[i])/muons)
    error_count.append(st.stdev(detections[i])/muons)

plt.figure()
labels = ["Scintillator 1","Scintillator 2","Scintillator 3","Scintillator 4"]
plt.xticks(range(len(av)), labels)
plt.bar(range(len(av)),av,yerr=error)
plt.ylabel("Fraction of muons")
plt.title("How many muons were detected by each scintillator?")
plt.show()

plt.figure()
labels = ["0 scintillators","1 scintillator","2 scintillators","3 scintillators","4 scintillators"]
plt.xticks(range(len(av_count)), labels)
plt.bar(range(len(av_count)),av_count,yerr=error_count)
plt.ylabel("Fraction of muons")
plt.title("How many scintillators was each muon detected by?")
plt.show()

plt.figure()
labels = ["Scintillator 1","Scintillator 2","Scintillator 3","Scintillator 4"]
plt.xticks(range(len(av_e)), labels)
plt.bar(range(len(av_e)),av_e,yerr=error_e)
plt.ylabel("Fraction of electrons")
plt.title("How many electrons were detected by each scintillator?")
plt.show()

print("Total number of muons entering the detector:",muons)
print("\n")
print("Fraction of muons detected by the first scintillator:",av[0])
print("Fraction of muons detected by the second scintillator:",av[1])
print("Fraction of muons detected by the third scintillator:",av[2])
print("Fraction of muons detected by the fourth scintillator:",av[3])
print("\n")
print("Fraction of muons detected by zero scintillators:",av_count[0])
print("Fraction of muons detected by one scintillator:",av_count[1])
print("Fraction of muons detected by two scintillators:",av_count[2])
print("Fraction of muons detected by three scintillators:",av_count[3])
print("Fraction of muons detected by four scintillators:",av_count[4])
print("\n")
print("Fraction of muons that decayed to electrons:",absorptions/muons)
print("\n")
print("Fraction of electrons detected by the first scintillator:",av_e[0])
print("Fraction of electrons detected by the second scintillator:",av_e[1])
print("Fraction of electrons detected by the third scintillator:",av_e[2])
print("Fraction of electrons detected by the fourth scintillator:",av_e[3])


###############################################################################


L = 10e-9                                                                       # Value of the luminosity.


def backgroundPoisson():
    
    """
    This function generates a Gaussian with mean 4.8 and standard deviation
    0.5. It then generates a Poisson distribution with random values from
    the Gaussian.
    """
    
    background_poisson = []
    background_mean = np.random.normal(4.8,0.5)                                 # Gaussian distribution.
    
    for i in range(50):
        
        background_poisson.append(np.random.poisson(background_mean))           # Background Poisson distribution.
    
    return background_poisson


def Sigma():
    
    """
    This function generates the values of the cross section that will be 
    used when finding the upper limit of the cross section.
    """
    
    sigma = []
    signal_mean = []
    
    for i in range(750):
        
        signal_mean.append(L*(0.02e8 + 0.02e8*i))
        sigma.append(signal_mean[i]/L)                                          # List of cross sections.
        
    return signal_mean,sigma


def signalPoisson(signal_mean):
    
    """
    This function generates a Poisson distribuition with values from the mean
    values of the signal (the cross section multiplied by the luminosity).
    """
    
    signal_poisson = np.random.poisson(signal_mean)                             # Signal Poisson distribution.
        
    return signal_poisson


def events():
    
    """
    This function generates a Poisson distribution representing the sum of 
    the background events and the signal events. It also determines how
    many values in the Poisson exceed the number of measured events, 6,
    and finds a list of cross sections that satisfy the confidence level
    (95% in this case). The first value in this list is the upper bound.
    """
    
    validSigma = []
    sixList = []
    num = 500                                                                   # Number of iterations per cross section.
    
    for i in range(750):
        
        signal_mean,sigma = Sigma()
        greaterThanSix = 0
        poisson = []
        
        for j in range(num):
            
            background_poisson = np.random.choice(backgroundPoisson())
            signal_poisson = signalPoisson(signal_mean[i])
            poisson.append(background_poisson + signal_poisson)                 # Total Poisson distribution.
            
            if poisson[j] > 6:
                
                greaterThanSix += 1
               
        if greaterThanSix/num >= 0.95:                                          # Confidence level.
            
            validSigma.append(sigma[i])
            
        sixList.append(greaterThanSix/num)
        
    return validSigma,sigma,sixList
            
"""
validSigma,sigma,sixList = events()

plt.plot(sigma,sixList)
plt.hlines(0.95,0,1.5e9)
plt.xlabel("Sigma")
plt.ylabel("Fraction of pseudo-experiments with more than 6 events")
plt.show()

print("The upper limit on the value of sigma is",validSigma[0]/1e9,"nb.")
"""





