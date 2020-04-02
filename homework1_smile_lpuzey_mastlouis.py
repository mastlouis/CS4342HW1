import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def fPC(y, yhat):
    return np.mean(y == yhat)


def measureAccuracyOfPredictors(predictors, X, y):
    num_predictors = 0
    votes = np.zeros((predictors.shape[0], X.shape[0]))
    for i in range(len(predictors)):
        r1, c1, r2, c2 = predictors[i].astype(np.int)
        if r1 == 0 and c1 == 0 and r2 == 0 and c2 == 0 and i != 0:
            break
        num_predictors += 1
        new_vote = X[:, r1, c1] - X[:, r2, c2]
        new_vote[new_vote > 0] = 1
        new_vote[new_vote != 1] = 0
        votes[i] = new_vote
    valid_votes = votes[:num_predictors]
    if valid_votes.shape[0] > 1:
        valid_votes = valid_votes.sum(axis=0)
        valid_votes /= num_predictors
        valid_votes = np.where(valid_votes >= 0.5, 1, 0)
    # yhat = np.where(votes[:num_predictors].sum(axis=1) > 0.5, 1, 0)
    yhat = valid_votes
    return fPC(y, yhat)


def get_predictors():
    for r1 in range(24):
        for c1 in range(24):
            for r2 in range(24):
                for c2 in range(24):
                    yield np.array([r1, c1, r2, c2])


def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    final_num_features = 5
    existing_predictors = np.zeros((final_num_features, 4))
    best_score = 0
    best_predictor = None

    # Define a function to squash the votes once
    squash = lambda vote: (2 * np.sum(vote)) // np.shape(vote)[0]
    vector_squash = np.vectorize(squash)

    # Choose the 5 best predictors
    for new_predictor_number in range(final_num_features):
        votes = np.zeros((final_num_features, trainingFaces.shape[0]))

        # For each candidate predictor
        best_score = 0
        best_predictor = None
        for new_predictor in get_predictors():
            existing_predictors[new_predictor_number] = new_predictor
            score = measureAccuracyOfPredictors(existing_predictors, trainingFaces, trainingLabels)
            # existing_predictors[new_predictor_number] = np.zeros((4))
            if score >= best_score:
                best_score = score
                best_predictor = new_predictor
        existing_predictors[new_predictor_number] = best_predictor

    print('Result of Stepwise Regression: ' + str(existing_predictors))
    return existing_predictors


#Draws the features out of the testing faces file
def drawFeatures(predictors, testingFaces):
    im = testingFaces[0,:,:]
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    # Show r1,c1
    for coordinates in predictors:
        r1, c1, r2, c2  = coordinates
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5,r2 - 0.5),1,1,linewidth=2,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
    # Display the merged result
    plt.show()

#Creates an output to a textfile that calculates the accuracy of the testing
def output(trainingFaces, trainingLabels, testingFaces, testingLabels):
    samples = [400,800,1200,1600,2000]
    with open("output.txt", "a") as f:
        predictors = None
        for sampleNum in samples:
            print("{} Training Examples".format(sampleNum),file = f)

            predictors = stepwiseRegression(trainingFaces[:sampleNum],trainingLabels[:sampleNum],testingFaces,testingLabels)
            print("Predictors used:{}".format(predictors),file = f)

            accuracy = measureAccuracyOfPredictors(predictors,trainingFaces,trainingLabels)
            print("Training Accuracy with {} examples = {}".format(sampleNum, accuracy),file = f)

            accuracy = measureAccuracyOfPredictors(predictors,testingFaces,testingLabels)
            print("Testing Accuracy with {} examples = {}\n\n".format(sampleNum, accuracy),file = f)

        print("Final Accuracy of model = {}".format(accuracy),file = f)

        print("Final Predictors used:\n{}".format(predictors),file = f)


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    output(trainingFaces, trainingLabels, testingFaces, testingLabels)
