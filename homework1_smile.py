import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def fPC(y, yhat):
    return np.logical_xor(y == 1, yhat == 0).sum() / len(y)


def measureAccuracyOfPredictors(predictors, X, y):
    yhat = np.array([])
    for i, image in enumerate(X):
        yhat = np.append(yhat, ensemble_on_image(predictors, image))
    return fPC(y, yhat)


def get_predictors():
    for r1 in range(24):
        for c1 in range(24):
            for r2 in range(24):
                for c2 in range(24):
                    yield np.array([r1, c1, r2, c2])


def stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels):
    existing_predictors = {}
    best_score = 0
    best_feature = None

    # Define a function to squash the votes once
    squash = lambda vote: (2 * np.sum(vote)) // np.shape(vote)[0]
    vector_squash = np.vectorize(squash)

    # Choose the 5 best predictors
    for _ in range(5):
        for predictor in [x for x in get_predictors()]:  # Generate a list of all possible predictors, iterate over it
            best_score = 0
            best_feature = None
            votes = np.array([])

            # Get a vote from the current predictor candidate
            compare = lambda image: 1 if image[predictor[0], predictor[1]] > image[predictor[2], predictor[3]] else 0
            vector_compare = np.vectorize(compare)
            votes = np.append(votes, vector_compare(trainingFaces))

            # Get a vote from each existing predictor
            for existing_predictor in existing_predictors:
                compare_existing = lambda image: 1 if image[existing_predictor[0], existing_predictor[1]] > image[existing_predictor[2], existing_predictor[3]] else 0
                vector_compare = np.vectorize(compare_existing)
                votes = np.append(votes, vector_compare(trainingFaces))

            # Squash down all votes into predictions for each image
            smile_guesses = vector_squash(votes)

            score = fPC(trainingLabels, smile_guesses)
            if score > best_score and predictor not in existing_predictors:
                score = best_score
                best_feature = predictor
        existing_predictors.add(best_feature)

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
