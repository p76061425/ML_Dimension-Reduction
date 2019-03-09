# <center> Dimension Reduction </center>


* PCA, ICA and LLE comparison
* Eigenface Algorithm

## Usage
### dm_reduction.py
```sh
$ python3 dm_reduction.py [-h] 
```
| optional Options | Description |
| ---              | --- |
| -h, --help       | show this help message and exit |
| -s, SCALING       | svm kernel,default=rbf |
| -nc, N_COMPONENTS  | input the size of batch |
| -n, NUMBER     | The number you want to plot in Scatter plot, 0..9 or all , default = 2|
| -gs, GRID_SIZE | The grid size of scatter plot for number images, default = 10 |
| -dr, DR_TECHNIQUES | The techniques of dimension reduction, default = pca, pca=(Principal Component Analysis) , ica =(Independent Component Analysis), lle =(Local Linear Embedding), all(pca,ica,lle) |
|-ims, IMG_SHOW | Show image, default = false |

可以直接下
python3 dm_reduction.py -n all -dr all
會執行全部方法的全部數字，結果圖片存在result資料夾內

### eigenface_algo.py
```sh
$ python3 eigenface_algo.py [-h] 
```

| optional Options | Description |
| ---              | --- |
|-h, --help | show this help message and exit|
|-nc, N_COMPONENTS | PCA,ICA,LLE n_components, default = 25|

用-nc來指定降維的數量。</br>
結果會在result_b資料夾內。


## Development Environment
    Ubuntu 18.04.1 LTS
    Architecture:        x86_64
	Model name:          Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz

## Result Analysis

### PCA, ICA and LLE comparison
* 結果詳述於Report file中
#### 簡易範例:
![](https://i.imgur.com/UnxKpYX.png)

### Eigenface Algorithm
* 結果詳述於Report file中
#### 簡易範例:
* __Top 25 eigenfaces__
  
![](https://i.imgur.com/vTOXlEr.png)

* __Corresponding eigenvalues in a descending order.__
   
![](https://i.imgur.com/jsD1YhP.png)
 
* __Reconstruct the image__

![](https://i.imgur.com/DXR1FdM.png) </br>
PSNR= 46.70577829969278
