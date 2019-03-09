import numpy as np
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import sys 
import os
import math


def warn(*args, **kwargs):    
    pass

def read_data(path):
    imgs = []
    for dirpath,dirnames,files in os.walk(path):
        for filename in files:
            img = plt.imread(path+filename)
            imgs.append(img)
    return np.array(imgs)

def psnr(img1, img2):
    mse = np.square(img1 - img2)
    mse = mse.sum() / (mse.size)
    psnr_result = 20 * np.log10(255 / mse ** 0.5)
    return psnr_result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc',
                        default='25',
                        dest='N_COMPONENTS',
                        help='PCA,ICA,LLE n_components, default = 25') 
    args = parser.parse_args()
    
    import warnings
    warnings.warn = warn
    
    N_COMPONENTS = int(args.N_COMPONENTS)

    train_img_path = "./training.db/"
    test_img_path = "test.tif"
    train_img = read_data(train_img_path)
    train_img_amount = train_img.shape[0]
    print("train_img_amount:",train_img_amount)

    result_dir = "./result_b/n_components_"+str(N_COMPONENTS)+"/"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        
    average_face = train_img.mean(axis=0)
    plt.imshow(average_face, cmap='gray')
    plt.title('Average Face')
    file_name = "average_face.png"
    plt.savefig(result_dir + file_name)
    #plt.subplots_adjust(left=0.15,bottom = 0.15) 
    plt.show()
   
    zero_mean_face_train = train_img - average_face
    pca = PCA( n_components = N_COMPONENTS)
    train_transform = pca.fit_transform(zero_mean_face_train.reshape(train_img_amount, -1))
   
    eigenvalues =  pca.explained_variance_
    print("Eigenvalues:", eigenvalues )

    fig = plt.figure()
    fig_size = math.ceil(math.sqrt(N_COMPONENTS))
    for i, eigenface in enumerate(pca.components_, start=1):
        fig.add_subplot(fig_size,fig_size,i)
        plt.imshow(eigenface.reshape(128, 128), cmap='gray')
        plt.title('Top ' + str(i))
        plt.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.35,wspace=0)
    file_name = "Top_" + str(N_COMPONENTS) +"_eigenfaces" + ".png"
    plt.savefig(result_dir + file_name)
    plt.show()
    
    #testing
    test_img = plt.imread(test_img_path)
    zero_mean_face_test = test_img - average_face
    test_face_transform = pca.transform(zero_mean_face_test.reshape(1, -1))
    test_face_inverse_transform = pca.inverse_transform(test_face_transform).reshape(128, 128) 
    test_face_reconstruct = test_face_inverse_transform + average_face

    #plot test img and reconstruct test img
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.title("Original Test Img")
    plt.imshow(test_img, cmap='gray')
    plt.axis('off')
    fig.add_subplot(1,2,2)
    plt.title("Reconstruct Test Img")
    plt.imshow(test_face_reconstruct, cmap='gray')
    plt.axis('off')
    plt.subplots_adjust(hspace=0.2,wspace=0.2)
    file_name = "reconstruct_img" + ".png"
    plt.savefig(result_dir + file_name)
    plt.show()
    
    #compute the top NC eigenface coefficients
    print("Top " + str(N_COMPONENTS) + " eigenface coefficients:")
    print(test_face_transform[0])
    
    #PSNR
    psnr_result = psnr(test_img,test_face_reconstruct)    
    print("psnr_result:", psnr_result)
        
    
    
    
    
