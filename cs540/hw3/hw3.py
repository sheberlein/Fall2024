from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    average = np.mean(x, axis = 0)
    centered_array = x - average
    return centered_array

def get_covariance(dataset):
    transpose = np.transpose(dataset)
    xDotxTranspose = np.dot(transpose, dataset)
    
    # now we need multiply by 1/(n-1)
    s = xDotxTranspose / (dataset.shape[0] - 1)
    return s

def get_eig(S, m):
    k = len(S[0])
    largest_k_eigenvalues, eigenvectors = eigh(S, subset_by_index = [k - m, k - 1])
    
    # first, flip the eigenvalues and eigenvectors list
    largest_k_eigenvalues = largest_k_eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    # make the eigenvalues into a diagonal matrix
    eigenvalues = np.diag(largest_k_eigenvalues)
    
    return eigenvalues, eigenvectors

def get_eig_prop(S, prop):
    largest_k_eigenvalues, eigenvectors = eigh(S)
    
    # first, flip the eigenvalues and eigenvectors lists
    largest_k_eigenvalues = largest_k_eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    denominator = sum(largest_k_eigenvalues)
    
    new_eigenvalues = []
    for eigenvalue in largest_k_eigenvalues:
        p = eigenvalue / denominator
        if (p > prop):
            new_eigenvalues.append(eigenvalue)

    # re-compute the eigenvalues and eigenvectors
    describing_evalues, describing_evectors = eigh(S, subset_by_value = (new_eigenvalues[len(new_eigenvalues) - 1] - 1e-4, np.inf))
    
    # flip the eigenvalues and eigenvectors lists
    describing_evalues = describing_evalues[::-1]
    describing_evectors = describing_evectors[:, ::-1]
    
    # make the eigenvalues into a diagonal matrix
    eigenvalues = np.diag(describing_evalues)
    
    return eigenvalues, describing_evectors

def project_image(image, U):
    # each row in dataset is an image.
    weights = []
    transpose = np.transpose(U)
    
    # loop through and add each dot product of xi and UTi 
    for i in range(len(U[0])):
        weights.append(np.dot(transpose[i], image))
        
    # make an array of zeros, to be updated in the for loop
    projection = np.zeros(np.shape(image))
    
    # the projection vector is the product of the weight of xi and the value of UTi
    for i in range(len(U[0])):
        projection += weights[i] * transpose[i]
    
    return projection

def display_image(orig, proj):
    # first, reshape the images to be 64 x 64
    new_orig = orig.reshape(64, 64)
    new_proj = proj.reshape(64, 64)
    
    # Please use the format below to ensure grading consistency
    fig, [ax1, ax2] = plt.subplots(figsize=(9,3), ncols=2)
    
    # ax1
    ax1.set_title("Original")
    im1 = ax1.imshow(new_orig, aspect='equal')
    fig.colorbar(im1, ax=ax1, location = 'right')
    
    # ax2
    ax2.set_title("Projection")
    im2 = ax2.imshow(new_proj, aspect='equal')
    fig.colorbar(im2, ax=ax2, location = 'right')
    
    return fig, ax1, ax2

def perturb_image(image, U, sigma):
    # each row in dataset is an image.
    weights = []
    transpose = np.transpose(U)
    
    # loop through and add each dot product of xi and UTi 
    for i in range(len(U[0])):
        weights.append(np.dot(transpose[i], image) + np.random.normal(scale = sigma))
        
    # make an array of zeros, to be updated in the for loop
    projection = np.zeros(np.shape(image))
    
    # the projection vector is the product of the weight of xi and the value of UTi
    for i in range(len(U[0])):
        projection += weights[i] * transpose[i]
    
    return projection

def combine_image(image1, image2, U, lam):
    # each row in dataset is an image.
    weights1 = []
    transpose = np.transpose(U)
    
    # loop through and add each dot product of xi and UTi 
    for i in range(len(U[0])):
        weights1.append(np.dot(transpose[i], image1))
    
    weights2 = []
    
    for i in range(len(U[0])):
        weights2.append(np.dot(transpose[i], image2))
        
    acomb = []
    for i in range(len(weights1)):
        acomb.append(lam*weights1[i] + (1 - lam)*weights2[i])
    
    # make an array of zeros, to be updated in the for loop
    combination = np.zeros(np.shape(image1))
    
    # the projection vector is the product of the weight of xi and the value of UTi
    for i in range(len(U[0])):
        combination += acomb[i] * transpose[i]
    return combination
