import sys
import re
import json
import math
import numpy as np
import pandas as pd

#read tweets and convert to a list including text and id
def readJson(tweetJson):
    for line in tweetJson:
        readTweets = json.loads(line)
        tweetsJsonData[str(readTweets["id"])] = readTweets["text"]

def tweetWords(wordsList):
    counts = {}
    for word in wordsList:
        if word in counts:
            counts[word] = counts[word] + 1
        else:
            counts[word] = 1
    return counts

def tweetUnion(tweet1, tweet2):
    result = 0
    for word in tweet1:
        if word in tweet2:
            result = result + max(tweet1[word], tweet2[word])
            tweet2.pop(word, None)
        else:
            result = result + tweet1[word]
    for word in tweet2:
        result = result + tweet2[word]
    return result

def tweetIntersection(tweet1, tweet2):
    result = 0
    for word in tweet1:
        while tweet1[word] != 0 and word in tweet2:
            if word in tweet2:
                tweet2[word] = tweet2[word] - 1
                tweet1[word] = tweet1[word] - 1
                if tweet2[word] == 0:
                    tweet2.pop(word, None)
                result += 1
    return result

# calculate the Jaccard distance
def jaccard_distance(tweet_a, tweet_b):
    tweet1Words = tweetWords(tweet_a)         
    tweet2Words = tweetWords(tweet_b)           
    tweetWordsUnion = tweetUnion(dict(tweet1Words), dict(tweet2Words))
    tweetWordsIntersect = tweetIntersection(dict(tweet1Words), dict(tweet2Words))
    return 1.0 - tweetWordsIntersect*1.0/tweetWordsUnion

# read tweet and check argument number
if len(sys.argv) == 5:
    print("working")
    numOfClusters = int(sys.argv[1])
    initialSeedsFile = sys.argv[2]
    tweetsDataFile = sys.argv[3]
    outputFile = sys.argv[4]
elif len(sys.argv) == 4:
    print("Number of Clusters are 25")
    numOfClusters = 25
    initialSeedsFile = sys.argv[1]
    tweetsDataFile = sys.argv[2]
    outputFile = sys.argv[3]
else:
    print("Arguments don't match")
    sys.exit(1)

# input the initial centroids and their contents
tweetCentroidIds = {}
with open(initialSeedsFile) as tweet_centroid:
    centroids = tweet_centroid.read().rsplit(",\n")
    if len(centroids) == numOfClusters:
        for id in range(0, numOfClusters):
            tweetCentroidIds[id] = centroids[id]
    else:
        print ("Initial seed file does't match clusters")
        sys.exit(1)
tweetsJsonData = {}
with open(tweetsDataFile) as tweetJson:
    readJson(tweetJson)


def formClusters(tweetCentroidIds, tweetsJsonData):
    clusters = {}
    for index in range(len(tweetCentroidIds)):
        clusters[index] = []
    for tweet in tweetsJsonData:
        minJaccardDist = 1
        cluster = 0
        for centroidId in tweetCentroidIds:
            tweetCentroidDist = 1
            tweetCentroidDist = jaccard_distance(tweetsJsonData[tweetCentroidIds[centroidId]], tweetsJsonData[tweet])
            if tweetCentroidDist < minJaccardDist:
                minJaccardDist = tweetCentroidDist
                cluster = centroidId
        clusters[cluster].append(tweet)
    return clusters

def recalculateCentroid(cluster, tweet_data):
    centroidId = cluster[0]
    min_distance = 1
    for tweet in cluster:
        total_distance = 0
        for other_tweet in cluster:
            total_distance = total_distance + jaccard_distance(tweet_data[tweet], tweet_data[other_tweet])
        mean_distance = total_distance * 1.0 / len(cluster)
        if mean_distance < min_distance:
            min_distance = mean_distance
            centroidId = tweet
    return centroidId

# calculate the SSE
def sse(clusters, centroid_values, tweet_data):
    result = 0
    for cluster in clusters:
        for tweet in clusters[cluster]:
            result += math.pow(jaccard_distance(tweet_data[tweet], tweet_data[centroid_values[cluster]]),2)
    return result

updateCentroidIds = {}
while True:
    clusters = formClusters(tweetCentroidIds, tweetsJsonData)
    for cluster in clusters:
        updateCentroidIds[cluster] = recalculateCentroid(clusters[cluster], tweetsJsonData)
    if updateCentroidIds == tweetCentroidIds:
        sseValue = str(sse(clusters, updateCentroidIds, tweetsJsonData))
        print ("The value of SSE: " + sseValue)
        break
    else:
        tweetCentroidIds = updateCentroidIds

# format the output
fileToOutput = open(outputFile, 'w')
fileToOutput.write("The SSE Value is: ")
fileToOutput.write(sseValue)
fileToOutput.write("\n\nClusters:\n\n")
for cluster in clusters:
    fileToOutput.write(str(cluster))
    fileToOutput.write("\t")
    for tweet in clusters[cluster]:
        fileToOutput.write(tweet)
        fileToOutput.write(", ")
    fileToOutput.write("\n\n")