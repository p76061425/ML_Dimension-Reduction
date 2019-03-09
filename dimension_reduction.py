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

def warn(*args, **kwargs):    
    pass

def read_data(x_path,y_path):
    with open(x_path,'rb')as x_f:
        x_data = np.array( [i for i in x_f.read()[16:]] ).reshape(-1, 784)        
        
    with open(y_path, 'rb') as y_f:
        y_data = np.array([i for i in y_f.read()[8:]]).reshape(-1)    
    
    return x_data,y_data
    
def Dimension_Reduction(x_train_org,DR_TECHNIQUES,SCALING,N_COMPONENTS):
    if(DR_TECHNIQUES == "pca"):
        pca_result = dr_pca(x_train_org,SCALING,N_COMPONENTS)
        return pca_result
    elif(DR_TECHNIQUES == "ica"):
        fastICA_result = dr_fastICA(x_train_org,SCALING,N_COMPONENTS)
        return fastICA_result
    elif(DR_TECHNIQUES == "lle"):
        lle_result = dr_lle(x_train_org,SCALING,N_COMPONENTS)
        return lle_result
    elif(DR_TECHNIQUES == "all"):
        pca_result = dr_pca(x_train_org,SCALING,N_COMPONENTS)
        fastICA_result = dr_fastICA(x_train_org,SCALING,N_COMPONENTS)
        lle_result = dr_lle(x_train_org,SCALING,N_COMPONENTS)
        return [pca_result,fastICA_result,lle_result]
    else:
        print("-dr(dimension reduction) must be pca, ica or lle")
        sys.exit()
        
def dr_pca(x_train_org,SCALING,N_COMPONENTS):
    if SCALING:
        print("scaling pca...")
        pca = make_pipeline( StandardScaler(), PCA(n_components = N_COMPONENTS) )
    else:
        print("non scaling pca...")
        pca = PCA( n_components = N_COMPONENTS)
    pca_result = pca.fit_transform(x_train_org)
    print("pca_result.shape:",pca_result.shape)
    return pca_result
    
def dr_fastICA(x_train_org,SCALING,N_COMPONENTS):
    if SCALING:
        print("scaling ica...")
        fastICA = make_pipeline( StandardScaler(), FastICA(n_components = N_COMPONENTS) )
    else:
        print("non scaling ica...")
        fastICA = FastICA( n_components = N_COMPONENTS)
    fastICA_result = fastICA.fit_transform(x_train_org)
    print("fastICA_result.shape:",fastICA_result.shape)
    return fastICA_result
   
def dr_lle(x_train_org,SCALING,N_COMPONENTS):
    if SCALING:
        print("scaling lle...")
        lle = make_pipeline( StandardScaler(), LocallyLinearEmbedding(n_components = N_COMPONENTS) )
    else:
        print("non scaling lle...")
        lle = LocallyLinearEmbedding( n_components = N_COMPONENTS)
    lle_result = lle.fit_transform(x_train_org)
    print("lle_result.shape:",lle_result.shape)
    return lle_result
    
def plot_grid_scatter(x_train_org,dr_result,GRID_SIZE,DR_TECHNIQUES,NUMBER,IMG_SHOW):
    x_min = min(dr_result[:,0])
    x_max = max(dr_result[:,0])
    y_min = min(dr_result[:,1])
    y_max = max(dr_result[:,1])
    x_interval = (x_max - x_min)/GRID_SIZE
    y_interval = (y_max - y_min)/GRID_SIZE
    interval = max(x_interval,y_interval)
    
    x_list = []
    y_list = []
    for i in range(GRID_SIZE):
        #curr_x = np.logical_and(dr_result[:,0] > x_min + x_interval * i, dr_result[:,0] <= x_min + x_interval*(i+1) )
        #curr_y = np.logical_and(dr_result[:,1] > y_min + y_interval * i, dr_result[:,1] <= y_min + y_interval*(i+1) )
        curr_x = np.logical_and(dr_result[:,0] > x_min + interval * i, dr_result[:,0] <= x_min + interval*(i+1) )
        curr_y = np.logical_and(dr_result[:,1] > y_min + interval * i, dr_result[:,1] <= y_min + interval*(i+1) )
        x_list.append(curr_x)
        y_list.append(curr_y)

    fig, ax = plt.subplots()
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            box = np.logical_and(x_list[i], y_list[j])
            
            args_index = np.argwhere(box == True)
            if args_index.size != 0 :
                img_index = args_index[0][0]
                #ax.imshow(x_train_org[img_index].reshape(28,28), cmap='gray',\
                #            extent=(dr_result[img_index, 0], dr_result[img_index, 0]+ max(x_interval,y_interval) * 0.6, dr_result[img_index, 1], dr_result[img_index, 1]+ max(x_interval,y_interval)  * 0.6),\
                #            zorder=2)
                ax.imshow(x_train_org[img_index].reshape(28,28), cmap='gray',\
                            extent=(dr_result[img_index, 0], dr_result[img_index, 0]+ interval * 0.5,\
                                    dr_result[img_index, 1], dr_result[img_index, 1]+ interval* 0.5),\
                            zorder=2)
                            
                ax.scatter(dr_result[img_index,0], dr_result[img_index,1], c='r', zorder=1, s = 5)
    
    plt.scatter(dr_result[:,0], dr_result[:,1], s=5, alpha=.5, c='blue', zorder=0)
    plt.title(DR_TECHNIQUES.upper() + ' Result, '+ 'Number:' +str(NUMBER) )
    
    result_dir = "./result_a/"
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    file_name = DR_TECHNIQUES+'_result_n'+ str(NUMBER) +"_gs"+ str(GRID_SIZE) +'.png'    
    plt.axis('equal')
    plt.savefig(result_dir + file_name)
    print("save file",DR_TECHNIQUES+'_result_n'+ str(NUMBER) +"_gs"+ str(GRID_SIZE) +'.png')
    
    if IMG_SHOW:
        plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        default='true',
                        dest='SCALING',
                        help='Features scaling, default = true')
    parser.add_argument('-nc',
                        default='2',
                        dest='N_COMPONENTS',
                        help='PCA,ICA,LLE n_components, default = 2') 
    parser.add_argument('-n',
                        default='2',
                        dest='NUMBER',
                        help='The number you want to plot in Scatter plot, 0..9,all ,default = 2' )
    parser.add_argument('-gs',
                        default='10',
                        dest='GRID_SIZE',
                        help='The grid size of scatter plot for number images, default = 10')                        
    parser.add_argument('-dr',
                        default='pca',
                        dest='DR_TECHNIQUES',
                        help='The techniques of dimension reduction, default = pca, \
                        pca =(Principal Component Analysis)  ,\
                        ica =(Independent Component Analysis), \
                        lle =(Local Linear Embedding),\
                        all(pca, ica, lle) ')
    parser.add_argument('-ims',
                        default='false',
                        dest='IMG_SHOW',
                        help='Show image, default = false')
                        
    args = parser.parse_args()
    
    import warnings
    warnings.warn = warn
    
    if(args.SCALING == "true"):
        SCALING = True
    elif(args.SCALING == "false"):
        SCALING = False
    N_COMPONENTS = int(args.N_COMPONENTS)
    GRID_SIZE = int(args.GRID_SIZE)
    DR_TECHNIQUES = args.DR_TECHNIQUES    
    if(args.IMG_SHOW == "true"):     
        IMG_SHOW = True
    elif(args.IMG_SHOW == "false"):
        IMG_SHOW = False
    
    print("SCALING:",SCALING)
    print("N_COMPONENTS:",N_COMPONENTS)
    print("GRID_SIZE:",GRID_SIZE)
    print("DR_TECHNIQUES:",DR_TECHNIQUES)
    
    x_train_path = 'train-images.idx3-ubyte'
    y_train_path = 'train-labels.idx1-ubyte'
    
    if(args.NUMBER == "all"):
        for NUMBER in range(10):
            print("NUMBER:",NUMBER)
            x_train,y_train = read_data(x_train_path,y_train_path)
            x_train_org = np.array([x_train[i] for i, label in enumerate(y_train) if label == NUMBER ]) 

            dr_result = Dimension_Reduction(x_train_org,DR_TECHNIQUES,SCALING,N_COMPONENTS)
            
            if(DR_TECHNIQUES == "all"):
                pca_result = dr_result[0]
                fastICA_result = dr_result[1]
                lle_result = dr_result[2]
                
                plot_grid_scatter(x_train_org,pca_result,GRID_SIZE,"pca", NUMBER,IMG_SHOW)
                plot_grid_scatter(x_train_org,fastICA_result,GRID_SIZE,"ica",NUMBER,IMG_SHOW)
                plot_grid_scatter(x_train_org,lle_result,GRID_SIZE,"lle",NUMBER,IMG_SHOW)
            else:
                plot_grid_scatter(x_train_org,dr_result,GRID_SIZE,DR_TECHNIQUES,NUMBER,IMG_SHOW)
            print("-----------------------------------")
            
    else:
        NUMBER = int(args.NUMBER)
        print("NUMBER:",NUMBER)
        print()
        x_train,y_train = read_data(x_train_path,y_train_path)
        x_train_org = np.array([x_train[i] for i, label in enumerate(y_train) if label == NUMBER ]) 

        dr_result = Dimension_Reduction(x_train_org,DR_TECHNIQUES,SCALING,N_COMPONENTS)
        
        if(DR_TECHNIQUES == "all"):
            pca_result = dr_result[0]
            fastICA_result = dr_result[1]
            lle_result = dr_result[2]
            
            plot_grid_scatter(x_train_org,pca_result,GRID_SIZE,"pca", NUMBER,IMG_SHOW)
            plot_grid_scatter(x_train_org,fastICA_result,GRID_SIZE,"ica",NUMBER,IMG_SHOW)
            plot_grid_scatter(x_train_org,lle_result,GRID_SIZE,"lle",NUMBER,IMG_SHOW)
        else:
            plot_grid_scatter(x_train_org,dr_result,GRID_SIZE,DR_TECHNIQUES,NUMBER,IMG_SHOW)


    
  