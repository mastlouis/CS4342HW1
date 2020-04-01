import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def fPC(y, yhat):
    return np.logical_xor(y == 1, yhat == 0).sum() / len(y)


def measureAccuracyOfPredictors(predictors, X, y):
    votes = np.empty((0, 4), int)
    for r1, c1, r2, c2 in predictors:
        new_vote = np.expand_dims((X[:, r1, c1] - X[:, r2, c2]) > 0, axis=0)  # Array of booleans
        votes = np.append(votes, new_vote, axis=0)
    yhat = np.where(votes.sum(axis=1) > 0.5, 1, 0)
    return fPC(y, yhat)


def get_predictors():
    for r1 in range(24):
        for c1 in range(24):
            for r2 in range(24):
                for c2 in range(24):
                    yield np.array([r1, c1, r2, c2])


def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    existing_predictors = np.empty((0, 4), int)
    best_score = 0
    best_feature = None

    # Define a function to squash the votes once
    squash = lambda vote: (2 * np.sum(vote)) // np.shape(vote)[0]
    vector_squash = np.vectorize(squash)

    # Choose the 5 best predictors
    for _ in range(5):
        votes = np.array([])

        # For each candidate predictor
        for new_predictor in get_predictors():
            best_score = 0
            best_feature = None
            new_predictor_rect = np.expand_dims(new_predictor, axis=0)
            all_predictors = np.append(existing_predictors, new_predictor_rect, axis=0)
            score = measureAccuracyOfPredictors(all_predictors, trainingFaces, trainingLabels)
            if score > best_score and new_predictor not in existing_predictors:
                score = best_score
                best_feature = new_predictor
        existing_predictors = np.append(existing_predictors, best_feature, axis=0)

    print(existing_predictors)

    show = False
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0, :, :]
        fig, ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1
        rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # Show r2,c2
        rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
        # Display the merged result
        plt.show()


def loadData(which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels


if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
