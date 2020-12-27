#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matlab.engine
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import datetime
from scipy.io import loadmat
from collections import defaultdict
from os import system
import math
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')


# In[2]:


eng = matlab.engine.start_matlab()
clear = lambda: system('clear')

quietDays = np.array(eng.quiet())
# Quiet days from 2009/09/5 to 2009/09/19
# quietDays[1580-1] = 2009/09/14

# year = np.int16(quietDays[1572-1][0])
# month = np.int16(quietDays[1572-1][1])
# day = np.int16(quietDays[1572-1][2])
year = 2000
month = 1
day = 1
hour = 0
minute = 0
second = 0
numberOfDaysWithData = 3


numberOfDaysWithDataDate = datetime.timedelta(days=numberOfDaysWithData)

startDate = datetime.datetime(year, month, day, hour, minute, second)
endDate = startDate+numberOfDaysWithDataDate
startDateBase = startDate

startDateMatlab = [startDate.year, startDate.month, startDate.day, startDate.hour, startDate.minute, startDate.second]
endDateMatlab = [endDate.year, endDate.month, endDate.day, endDate.hour, endDate.minute, endDate.second]


stationsWithNoData = []
stationsNonexistentInFolder = []
stationsOut = dict()
stationIndicatorVariation = defaultdict(list)

trainingDatasetPathWindows = "D:/IRAP/TrainingDataset"

# WINDOWS
mat = loadmat("D:/IRAP/dir.indices/station.mat")
# LINUX
# mat = loadmat("../../../../opt/dir.indices/station.mat")

allMatData = mat.get("station", "none")[0]
allStationCodes=np.array([])
allStationLatgeos=np.array([])

for x in allMatData:
    allStationCodes=np.append(allStationCodes,x[1][0])
    allStationLatgeos=np.append(allStationLatgeos,x[3][0])
    
latMin=36
latMax=60
    

stationIndicatorRatioVariation = defaultdict(list)
numberOfMinutesNeededInTheTimeStamp=0
timeBetweenValues=0
numberOfValues=0
dataResolution=0
indicatorVariationArray = np.array([])
localIndicatorVariationArray = np.array([])
indicatorVariationArrayLocalTime = np.array([])
localNormalized01StationIndicatorVariation = np.array([])
stationsPerLat = defaultdict(list)
normalizedStationIndicatorVariation = defaultdict(list)
normalized01StationIndicatorVariation = defaultdict(list)
maxValueinDataset=0
minValueinDataset=0
latsWithoutData = np.array([])
ReconstructedArray = np.array([])
maxValueInMatrix=0
minValueInMatrix=0


# In[36]:


def GenerateTrainingSet():
    global year
    global month
    global day
    global startDate
    global endDate
    global startDateMatlab
    global endDateMatlab
    global stationsWithNoData
    global stationsNonexistentInFolder
    global stationsOut
    global indicatorVariationArrayLocalTime
    
    stationsWithNoData = []
    stationsNonexistentInFolder = []
    stationsOut = dict()
    stationIndicatorVariation = defaultdict(list)
    index=0
    indexBaseSave = 0
    dataSetSize = 3000
    matrixDurationDays = 1
    for i in range(0,dataSetSize,matrixDurationDays):
    	try:
            startDate = startDateBase+datetime.timedelta(days=i)
            endDate = startDate+datetime.timedelta(days=numberOfDaysWithData)

            startDateMatlab = [startDate.year, startDate.month, startDate.day, 0, 0, 0]
            endDateMatlab = [endDate.year, endDate.month, endDate.day, 0, 0, 0]
            sys.stdout.flush()

            RequestMatlab()
            MakeStationIndicatorVariation()
            ManuallyNormalizeData01()      
            makeIndicatorVariationArray(False,True)
            ResizeForPlot()
            np.save("{}/x_train/Y2_{}_{}_{}_base".format(trainingDatasetPathWindows, latMin, latMax, index), indicatorVariationArrayLocalTime)
            RemoveDefectiveStation(indicatorVariationArrayLocalTime)
            ReconstructedArray = PredictIndicatorForAllLatitudes(indicatorVariationArrayLocalTime)
            np.save("{}/x_train/Y2_{}_{}_{}".format(trainingDatasetPathWindows, latMin, latMax, index), ReconstructedArray)
            infosArray = np.array([startDate, latMax, latMin, maxValueinDataset, minValueinDataset, matrixDurationDays])
            np.save("{}/x_train/Y2_{}_{}_{}_infos".format(trainingDatasetPathWindows, latMin, latMax, index), infosArray)
            print("Matrix saved for date: {}".format(startDate))
            print("Sample {} out of {}".format(i, dataSetSize))

            index+=1
    	except:
    	    print ("Error with index {}".format(i))


# In[33]:


def RemoveDefectiveStation(array):
    rmseRef = np.array([])
    rmseRefIndex = np.array([])
    for i in range(array.shape[0]):
        if not math.isnan(np.sum(array[i])):
            rmseRefIndex = np.append(rmseRefIndex, i)
            rmseRef = np.append(rmseRef, mean_squared_error(np.nanmean(array, axis=0), array[i]))
    array[np.int16(rmseRefIndex[np.argmax(rmseRef)])] = np.full(array.shape[1], np.nan)
    return array


# In[6]:


def RequestMatlab():
    global startDateMatlab, endDateMatlab, startDate, endDate, year, month, day
    for i in range(0,allStationCodes.shape[0]): 
        if allStationLatgeos[i]>latMin and allStationLatgeos[i]<latMax:
            try:
                stationsOut[allStationCodes[i]] = eng.indices_alpha(matlab.double(startDateMatlab), matlab.double(endDateMatlab),str(allStationCodes[i]))
            except:
                stationsNonexistentInFolder.append(st)


# In[7]:


def IndicatorCalculation(dataSt, timeshift, currentDate, igrf):
    
    x1=np.float32(dataSt.get("x1")[timeshift])
    y1=np.float32(dataSt.get("y1")[timeshift])
    y2=np.float32(dataSt.get("y2")[timeshift])
    z1=np.float32(dataSt.get("z1")[timeshift])

    ratio = ((math.sqrt(pow(x1, 2)+pow(y1, 2)+pow(z1, 2)))/igrf)

    return np.round(y2,10), ratio


# In[8]:


def CalculateIGRF(dataSt, currentDate):    
    stLongeo=np.float32(dataSt.get("longeo"))
    stLatgeo=np.float32(dataSt.get("latgeo"))
    stAlt=np.float32(dataSt.get("alt"))
    
    currentDateMatlab = [currentDate.year, currentDate.month, currentDate.day, currentDate.hour, currentDate.minute, currentDate.second]
    b=eng.igrf(matlab.double([stLongeo]), matlab.double([stLatgeo]), matlab.double([stAlt]), matlab.double([currentDateMatlab]), nargout=4)
    return b[3]


# In[9]:


def MakeStationIndicatorVariation():
    
    global stationIndicatorVariation
    global stationIndicatorRatioVariation
    global stationIndicatorVariation
    global numberOfMinutesNeededInTheTimeStamp
    global timeBetweenValues
    global numberOfValues
    global dataResolution
    global latMin
    global latMax
    
    stationIndicatorRatioVariation = defaultdict(list)
    stationIndicatorVariation.clear()
    numberOfMinutesNeededInTheTimeStamp = 1440*numberOfDaysWithData
    timeBetweenValues = 10
    numberOfValues = np.int16(numberOfMinutesNeededInTheTimeStamp/timeBetweenValues)
    dataResolution = 0

    if float(360%(numberOfMinutesNeededInTheTimeStamp/timeBetweenValues)).is_integer():
        dataResolution = 360/(numberOfMinutesNeededInTheTimeStamp/timeBetweenValues)

        for st in stationsOut.keys():
            if stationsOut[st]:
                if stationsOut[st].get("latgeo") > latMin and stationsOut[st].get("latgeo") < latMax:
                    igrf=CalculateIGRF(stationsOut[st], startDate)
                    delta = endDate-startDate
                    totalMinutes = (delta.total_seconds()+1)/60
                    for i in range(0, np.int16(totalMinutes), timeBetweenValues):
                        try:
                            update = datetime.timedelta(minutes=i)
                            currentDate = startDate+update
                            magneticValue, ratio = IndicatorCalculation(stationsOut.get(st), i, currentDate, igrf)
                            stationIndicatorVariation[st].append(magneticValue)
                            stationIndicatorRatioVariation[st].append(ratio)
                        except:
                            None

    else:
        print("!ERROR! -- Your timeBetweenValues must be a multiple of the chosen time delta's amount of minutes -- !ERROR!")
        sys.exit


# In[10]:


def normalizeWithGivenBounds(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]               


# In[11]:


def ManuallyNormalizeData01():
    global normalizedStationIndicatorVariation
    global normalized01StationIndicatorVariation
    global maxValueinDataset
    global minValueinDataset
    normalizedStationIndicatorVariation = defaultdict(list)
    normalized01StationIndicatorVariation = defaultdict(list)
    maxValueinDataset=0
    minValueinDataset=0
    max_values = np.array([])
    min_values = np.array([])
    max_values_ratio = np.array([])
    min_values_ratio = np.array([])
    
    for st in stationsOut.keys():
        if stationIndicatorVariation[st]:
            if not math.isnan(stationIndicatorVariation[st][0]):
                max_values = np.append(max_values, max(stationIndicatorVariation[st])) 
                min_values = np.append(min_values, min(stationIndicatorVariation[st]))
                max_values_ratio = np.append(max_values_ratio, max(stationIndicatorRatioVariation[st])) 
                min_values_ratio = np.append(min_values_ratio, min(stationIndicatorRatioVariation[st])) 

    totalMax = max(max_values)
    totalMin = min(min_values)
    totalMaxRatio = max(max_values_ratio)
    totalMinRatio = min(min_values_ratio)
    maxValueinDataset = totalMax
    minValueinDataset = totalMin
    
    bounds = np.array([0,1])   
    boundsRatio = np.array([-1,1])   
    for st in stationsOut.keys():
        if stationIndicatorVariation[st]:
            localMax = max(stationIndicatorVariation[st])
            localMin = min(stationIndicatorVariation[st])
            localMaxRatio = max(stationIndicatorRatioVariation[st])
            localMinRatio = min(stationIndicatorRatioVariation[st])
            
            IndicatorVariationAppliedRatio=stationIndicatorVariation.copy()
            for i in range(0, len(stationIndicatorVariation[st])):
                IndicatorVariationAppliedRatio[st][i] = stationIndicatorVariation[st][i]*stationIndicatorRatioVariation[st][i]
            normalized01StationIndicatorVariation[st] = normalizeWithGivenBounds(np.array(IndicatorVariationAppliedRatio[st]), {'actual': {'lower': totalMin, 'upper': totalMax}, 'desired': {'lower': bounds[0], 'upper': bounds[1]}})


# In[12]:


def indexValueOnLocalTime(array, st, i):
    numberOfValuesLong = array.shape[1]
    localTimeValuesArray = np.full((180, numberOfValuesLong), np.nan)
    long = float(stationsOut[st].get("longeo"))
    shiftValues = np.round((long*4)/timeBetweenValues,0)
    initialShiftValues = shiftValues
    decreasingIndex=0
    increasingIndex=0
    for y in range(np.int16(numberOfValues/numberOfDaysWithData),numberOfValues):
        if y+shiftValues>=array.shape[1]:
            localTimeValuesArray[i][np.int16(y-(y-shiftValues))-decreasingIndex] = array[i][np.int16(numberOfValues/numberOfDaysWithData)-decreasingIndex]
            decreasingIndex+=1
        else:
            localTimeValuesArray[i][np.int16(y-y+shiftValues+increasingIndex)] = array[i][y]
            increasingIndex+=1
            
    return localTimeValuesArray[i] 


# In[13]:


def makeIndicatorVariationArray(normalizedGauss, normalizedScaler):
    global indicatorVariationArray
    global localIndicatorVariationArray
    global indicatorVariationArrayLocalTime
    global localNormalized01StationIndicatorVariation
    global stationsPerLat
    
    # LINUX
    # indicatorVariationArray = np.full((180, len(normalized01StationIndicatorVariation[normalized01StationIndicatorVariation.keys()[1]])), np.nan)
    # WINDOWS
    indicatorVariationArray = np.full((180, len(normalized01StationIndicatorVariation[list(normalized01StationIndicatorVariation.keys())[1]])), np.nan)

    localIndicatorVariationArray = np.full_like(indicatorVariationArray, np.nan)
    # LINUX
    # indicatorVariationArrayLocalTime = np.full((180, len(normalized01StationIndicatorVariation[normalized01StationIndicatorVariation.keys()[1]])), np.nan)
    # localNormalized01StationIndicatorVariation = np.full((180, len(normalized01StationIndicatorVariation[normalized01StationIndicatorVariation.keys()[1]])), np.nan)
    # WINDOWS
    indicatorVariationArrayLocalTime = np.full((180, len(normalized01StationIndicatorVariation[list(normalized01StationIndicatorVariation.keys())[1]])), np.nan)
    localNormalized01StationIndicatorVariation = np.full((180, len(normalized01StationIndicatorVariation[list(normalized01StationIndicatorVariation.keys())[1]])), np.nan)


    stationsPerLat = defaultdict(list)

    intermediateLocalIndicatorVariationArray = np.empty_like(localIndicatorVariationArray)
    intermediateLocalNormalized01StationIndicatorVariation = np.empty_like(localNormalized01StationIndicatorVariation)
    alreadyFilled = False
    for st in stationsOut.keys():
        alreadyFilled=False
        for i in range(latMin, latMax):
            if not isinstance(stationsOut[st], matlab.double):
            # if not isinstance(stationsOut[st], matlab.mlarray.double):
                if i == np.round(np.int16(stationsOut[st].get("latgeo")),0):
                    if stationsOut[st].get('longeo')<40:
                        stationsPerLat[i+90].append(st)
                        if len(stationsPerLat[i+90])>1:
                            alreadyFilled=True
                        if normalizedScaler:
                            if normalized01StationIndicatorVariation[st]:
                                if alreadyFilled==True:
                                    stName = stationsPerLat[i+90][stationsPerLat[i+90].index(min(stationsPerLat[i+90]))-1]
                                    localIndicatorVariationArray[i+90] = normalized01StationIndicatorVariation[stName]
                                    localNormalized01StationIndicatorVariation[i+90] = normalized01StationIndicatorVariation[stName]
                                    indicatorVariationArrayLocalTime[i+90] = indexValueOnLocalTime(localNormalized01StationIndicatorVariation, stName, i+90)
                                    indicatorVariationArray[i+90] = localIndicatorVariationArray[i+90]
                                else:
                                    localIndicatorVariationArray[i+90] = normalized01StationIndicatorVariation[st]
                                    localNormalized01StationIndicatorVariation[i+90] = normalized01StationIndicatorVariation[st]
                                    indicatorVariationArrayLocalTime[i+90] = indexValueOnLocalTime(localNormalized01StationIndicatorVariation, st, i+90)
                                    indicatorVariationArray[i+90] = localIndicatorVariationArray[i+90]

                            else: 
                                if len(stationIndicatorVariation[st])!=0:
                                    indicatorVariationArray[i+90]=stationIndicatorVariation[st]
                                    indicatorVariationArrayLocalTime[i+90] = indexValueOnLocalTime(indicatorVariationArray, st, i+90)
                                else:
                                    None
                else:
                    None


# In[14]:


def ResizeForPlot():
    global indicatorVariationArrayLocalTime
    global indicatorVariationArray
    indicatorVariationArrayResized = np.empty([latMax-latMin, np.int16(numberOfValues-((numberOfValues/numberOfDaysWithData)*2))])
    indicatorVariationArrayLocalTimeResized = np.empty([latMax-latMin, np.int16(numberOfValues-((numberOfValues/numberOfDaysWithData)*2))])
    m=0
    for i in range(latMin+90, latMax+90):
        a=0
        for y in range(np.int16(numberOfValues/numberOfDaysWithData),np.int16(numberOfValues-(numberOfValues/numberOfDaysWithData))):
            indicatorVariationArrayResized[m][a]=indicatorVariationArray[i][y]
            a+=1
        m+=1
    indicatorVariationArray = np.empty_like(indicatorVariationArrayResized)
    indicatorVariationArray=indicatorVariationArrayResized[:]

    m=0
    for i in range(latMin+90, latMax+90):
        a=0
        for y in range(np.int16(numberOfValues/numberOfDaysWithData),np.int16(numberOfValues-(numberOfValues/numberOfDaysWithData))):
            indicatorVariationArrayLocalTimeResized[m][a]=indicatorVariationArrayLocalTime[i][y]
            a+=1
        m+=1

    indicatorVariationArrayLocalTime = np.empty_like(indicatorVariationArrayLocalTimeResized)
    indicatorVariationArrayLocalTime=indicatorVariationArrayLocalTimeResized[:]


# In[25]:


def PredictIndicatorForAllLatitudes(baseArray):
    global ReconstructedArray, latsWithoutData
    latsWithoutData = np.array([])
    predictionArray = np.empty_like(baseArray)
    predictionArray=np.copy(baseArray)
    modelTuned = {'polynomialfeatures__degree': 2, 'linearregression__fit_intercept': True, 'linearregression__normalize': True}
    for i in range(0,baseArray.shape[0]):
        specificLatitudeTimePrediction = np.full(baseArray.shape[1], np.nan)
        if math.isnan(np.sum(baseArray[i])):
            latsWithoutData = np.append(latsWithoutData, i+latMin)
            for y in range(0,baseArray.shape[1]):
                if (math.isnan(baseArray[i][y])):
                    specificLatitudeTimePrediction[y] = GetIndicatorLongPrediction(i,y,modelTuned,predictionArray)
            predictionArray[i] = specificLatitudeTimePrediction
    return predictionArray


# In[16]:


def GetIndicatorLongPrediction(latitude,longitude, params, baseArray):
    indicatorLatVariation = np.array([])
    for i in range(0, baseArray.shape[0]):
        indicatorLatVariation = np.append(indicatorLatVariation, baseArray[i][longitude])
    y = np.array(indicatorLatVariation)
    x = np.arange(0, baseArray.shape[0], 1)
    x,y = RemoveInfNan(x, y)
    return PolyRegression(x,y,params).predict(np.array(latitude).reshape(1,-1))


# In[17]:


def RemoveInfNan(latValues, indicatorValues):
    indexDeleteY = np.array([])
    for i in range(0, indicatorValues.shape[0]):
        if math.isinf(indicatorValues[i]) or math.isnan(indicatorValues[i]):
            indexDeleteY = np.append(indexDeleteY, i)
    newY = np.delete(indicatorValues, indexDeleteY)
    newX = np.delete(latValues, indexDeleteY)
    newY=newY.reshape(newY.shape[0],1)
    newX=newX.reshape(newY.shape[0],1)
    
    return newX, newY


# In[18]:


def PolyRegression(latValues, indicatorValues, params):
    poly_grid = PolynomialRegression()
    poly_grid.set_params(**params)
    poly_grid.fit(latValues, indicatorValues)
    return poly_grid


# In[19]:


def ParametersTuningPoly(baseArray,long):
    indicatorLatVariation = np.array([])
    for i in range(0, baseArray.shape[0]):
        indicatorLatVariation = np.append(indicatorLatVariation, baseArray[i][np.int16(long)])

    y = np.array(indicatorLatVariation)
    x = np.arange(0, baseArray.shape[0], 1)

    x, y = RemoveInfNan(x, y)
    
    paramsTuning = {'polynomialfeatures__degree': [2,2], 'linearregression__fit_intercept': [True, False], 'linearregression__normalize': [True, False]}
    poly_gridTuning = GridSearchCV(PolynomialRegression(), paramsTuning, cv=10, scoring='r2', verbose=0)
    poly_gridTuning.fit(x, y)
    return poly_gridTuning.best_params_


# In[20]:


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


# In[34]:


GenerateTrainingSet() 


# In[24]:


matrixInfos = np.load("{}/x_train/Y2_36_60_{}_infos.npy".format(trainingDatasetPathWindows,imageIndex))
print(matrixInfos)


# In[35]:


fig, (ax1,ax2)= plt.subplots(2,1)
fig.set_size_inches(20,10)
imageIndex=1
matrixToPlot = np.load("{}/x_train/Y2_36_60_{}.npy".format(trainingDatasetPathWindows,imageIndex))
matrixToPlotBase = np.load("{}/x_train/Y2_36_60_{}_base.npy".format(trainingDatasetPathWindows, imageIndex))

ax1.imshow(matrixToPlot, origin='lower', cmap=plt.get_cmap('jet', 20) , aspect='auto')
ax2.imshow(matrixToPlotBase, origin='lower', cmap=plt.get_cmap('jet', 20) , aspect='auto')
fig.show()

