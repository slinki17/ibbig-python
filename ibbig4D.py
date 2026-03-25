## Szükséges könyvtárak betöltése (numpy, typing -> adatszerkezetek, matplotlib -> ábrázolás, ctypes -> C kód futtatása)
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import ctypes

## C kód beolvasása (először dll formátumra kell hozni)
clib = ctypes.CDLL('C/ibbig4d.dll')

## C-kompatibilis típusok beállítása a paraméterekre
clib.clusterCovs4DC.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),      # covTen - bemeneti tenzor
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),        # group - klaszterhez való tartozás
    ctypes.POINTER(ctypes.c_int),                                                # noCovs - oszlopok száma (1D)
    ctypes.POINTER(ctypes.c_int),                                                # noSigs - sorok száma (2D)
    ctypes.POINTER(ctypes.c_int),                                                # noSlices - szeletek száma (3D)
    ctypes.POINTER(ctypes.c_int),                                                # noTime - időpillanatok száma (4D)
    ctypes.POINTER(ctypes.c_double),                                             # alpha
    ctypes.POINTER(ctypes.c_int),                                                # noPop - populáció mérete
    ctypes.POINTER(ctypes.c_int),                                                # maxStag - maximum stagnálás
    ctypes.POINTER(ctypes.c_double),                                             # mutation - mutációs valószínűség
    ctypes.POINTER(ctypes.c_double),                                             # SR
    ctypes.POINTER(ctypes.c_int),                                                # max_SP - maximum kiválasztási valószínűség
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')       # SP - kiválasztási valószínűség
]

## Eredmények értelmezésére és ábrázolására használt osztály
class iBBiGc4D:
    
    def __init__(self,
                 Seeddata: Optional[np.ndarray] = None,
                 RowScorexNumber: Optional[np.ndarray] = None,
                 Clusterscores: Optional[Union[np.ndarray, List]] = None,
                 RowxNumber: Optional[np.ndarray] = None,
                 NumberxCol: Optional[np.ndarray] = None,
                 NumberxDepth: Optional[np.ndarray] = None,
                 NumberxTime: Optional[np.ndarray] = None,
                 Number: int = 0,
                 info: Optional[dict] = None,
                 Parameters: Optional[dict] = None):
        self.Seeddata = np.asarray(Seeddata) if Seeddata is not None else np.zeros((0, 0))
        self.RowScorexNumber = np.asarray(RowScorexNumber) if RowScorexNumber is not None else np.zeros((self.Seeddata.shape[0], 0))
        self.Clusterscores = np.asarray(Clusterscores) if Clusterscores is not None else np.array([])
        self.RowxNumber = np.asarray(RowxNumber) if RowxNumber is not None else np.zeros((self.Seeddata.shape[0], 0), dtype=bool)
        self.NumberxCol = np.asarray(NumberxCol) if NumberxCol is not None else np.zeros((0, self.Seeddata.shape[1] if self.Seeddata.size else 0), dtype=bool)
        self.NumberxDepth = np.asarray(NumberxDepth) if NumberxDepth is not None else np.zeros((0, self.Seeddata.shape[2] if self.Seeddata.size else 0), dtype=bool)
        self.NumberxTime = np.asarray(NumberxTime) if NumberxTime is not None else np.zeros((0, self.Seeddata.shape[3] if self.Seeddata.size else 0), dtype=bool)
        self.Number = int(Number)
        self.info = info or {}
        self.Parameters = Parameters or {}

    @property
    def RowScorexNumber(self):
        return self._RowScorexNumber

    @RowScorexNumber.setter
    def RowScorexNumber(self, value):
        self._RowScorexNumber = np.asarray(value) if value is not None else np.zeros((self.Seeddata.shape[0], 0))

    @property
    def Clusterscores(self):
        return self._Clusterscores

    @Clusterscores.setter
    def Clusterscores(self, value):
        self._Clusterscores = np.asarray(value) if value is not None else np.array([])

    @property
    def Seeddata(self):
        return self._Seeddata

    @Seeddata.setter
    def Seeddata(self, value):
        self._Seeddata = np.asarray(value) if value is not None else np.zeros((0, 0))

    @property
    def Parameters(self):
        return self._Parameters

    @Parameters.setter
    def Parameters(self, value):
        self._Parameters = value or {}

    @property
    def RowxNumber(self):
        return self._RowxNumber

    @RowxNumber.setter
    def RowxNumber(self, value):
        self._RowxNumber = np.asarray(value, dtype=bool) if value is not None else np.zeros((self.Seeddata.shape[0], 0), dtype=bool)

    @property
    def NumberxCol(self):
        return self._NumberxCol

    @NumberxCol.setter
    def NumberxCol(self, value):
        self._NumberxCol = np.asarray(value, dtype=bool) if value is not None else np.zeros((0, self.Seeddata.shape[1] if self.Seeddata.size else 0), dtype=bool)

    @property
    def NumberxDepth(self):
        return self._NumberxDepth

    @NumberxDepth.setter
    def NumberxDepth(self, value):
        self._NumberxDepth = np.asarray(value, dtype=bool) if value is not None else np.zeros((0, self.Seeddata.shape[2] if self.Seeddata.size else 0), dtype=bool)

    @property
    def NumberxTime(self):
        return self._NumberxTime

    @NumberxTime.setter
    def NumberxTime(self, value):
        self._NumberxTime = np.asarray(value, dtype=bool) if value is not None else np.zeros((0, self.Seeddata.shape[3] if self.Seeddata.size else 0), dtype=bool)

    @property
    def Number(self):
        return self._Number

    @Number.setter
    def Number(self, value):
        self._Number = int(value)

    @property
    def info(self):
        return self._info

    @info.setter
    def info(self, value):
        self._info = value or {}



## Jel-blokkokat a kezdeti mátrixhoz/tenzorhoz hozzáadó függvény
def addSignal(arti, startC, startR, startS, startT, endC, endR, endS, endT, densityH, densityL):

    numRows = endR - startR + 1
    numCols = endC - startC + 1
    numSlices = endS - startS + 1
    numTime = endT - startT + 1
    diffDens = densityH - densityL
    stepD = diffDens / numRows

    for i in range(startR - 1, endR):
        prob = densityH - (i - (startR - 1)) * stepD
        random_vals = (np.random.rand(numCols, numSlices, numTime) < prob).astype(int)
        arti[i, startC - 1:endC, startS - 1:endS, startT - 1:endT] = random_vals

    return arti



## Jel-blokkokat létrehozó függvény tetszőleges alsó- és felső sűrűséggel
def makeSimDesignMat(verbose=True):
    
    ## [50 x 50 x 50 x 50] méretű jel-blokkok
    #designMat = np.array([
    #    [1, 1, 1, 1, 42, 22, 40, 42, 1.0, 1.0],
    #    [26, 34, 22, 26, 46, 42, 42, 34, 1.0, 1.0],
    #    [18, 22, 26, 38, 22, 30, 50, 48, 1.0, 1.0],
    #    [1, 34, 1, 18, 30, 50, 1, 44, 1.0, 1.0],
    #    [46, 1, 34, 38, 50, 18, 50, 50, 1.0, 1.0]
    #])
    
    ## [20 x 20 x 20 x 20] méretű jel-blokkok
    designMat = np.array([
        [1, 1, 1, 1, 16, 6, 10, 11, 1.0, 1.0],
        [8, 12, 6, 8, 18, 16, 16, 12, 1.0, 1.0],
        [4, 6, 8, 14, 6, 10, 20, 19, 1.0, 1.0],
        [1, 12, 1, 4, 10, 20, 1, 17, 1.0, 1.0],
        [18, 1, 12, 14, 20, 4, 20, 20, 1.0, 1.0]
    ])

    ## [10 x 10 x 10 x 10] méretű jel-blokkok
    #designMat = np.array([
    #    [1, 1, 1, 1, 8, 3, 5, 6, 1.0, 1.0],
    #    [4, 6, 3, 4, 9, 8, 8, 6, 1.0, 1.0],
    #    [2, 3, 4, 3, 7, 5, 10, 9, 1.0, 1.0],
    #    [1, 6, 1, 2, 5, 10, 1, 8, 1.0, 1.0],
    #    [9, 1, 6, 7, 10, 2, 10, 10, 1.0, 1.0],
    #    ])

    ## Opcionális kiíratás a konzolra
    if verbose:
        cols = (designMat[:, 4] - designMat[:, 0]) + 1
        rows = (designMat[:, 5] - designMat[:, 1]) + 1
        slices = (designMat[:, 6] - designMat[:, 2]) + 1
        time = (designMat[:, 7] - designMat[:, 3]) + 1
        dens_low = designMat[:, 9]
        dens_high = designMat[:, 8]
        print("***** Summary of Design Matrix ******")
        print(np.column_stack([rows, cols, slices, time, dens_low, dens_high]))

    return designMat



## A végleges, jel-blokkokat tartalmazó bemeneti mátrixot/tenzort létrehozó függvény
def makeArtificial(nRow=20, nCol=20, nSlice=20, nTime=20, noise=0.0, verbose=True,
                   dM=None, seed=123):
    
    if dM is None:
        dM = makeSimDesignMat(verbose=verbose)

    np.random.seed(seed)

    arti = (np.random.rand(nRow, nCol, nSlice, nTime) < noise).astype(int)

    for i in range(dM.shape[0]):
        arti = addSignal(
            arti,
            int(dM[i, 0]), int(dM[i, 1]),
            int(dM[i, 2]), int(dM[i, 3]),
            int(dM[i, 4]), int(dM[i, 5]),
            int(dM[i, 6]), int(dM[i, 7]),
            dM[i, 8], dM[i, 9]
        )

    nClust = dM.shape[0]
    RN = np.zeros((nRow, nClust), dtype=bool)
    NC = np.zeros((nClust, nCol), dtype=bool)
    NS = np.zeros((nClust, nSlice), dtype=bool)
    NT = np.zeros((nClust, nTime), dtype=bool)
    
    for i in range(nClust):
        c_start, c_end = int(dM[i, 0]) - 1, int(dM[i, 4])
        r_start, r_end = int(dM[i, 1]) - 1, int(dM[i, 5])
        s_start, s_end = int(dM[i, 2]) - 1, int(dM[i, 6])
        t_start, t_end = int(dM[i, 3]) - 1, int(dM[i, 7])
        RN[r_start:r_end, i] = True
        NC[i, c_start:c_end] = True
        NS[i, s_start:s_end] = True
        NT[i, t_start:t_end] = True

    if verbose:
        print("\nCluster sizes in simulated bicluster data:")
        print(f"Number of Modules: {nClust}")
        print("Rows per module:", RN.sum(axis=0))
        print("Cols per module:", NC.sum(axis=1))
        print("Slices per module:", NS.sum(axis=1))
        print("Time per module:", NT.sum(axis=1))

    return {
        "matrix": arti,
        "RowxNumber": RN,
        "NumberxCol": NC,
        "NumberxSlice": NS,
        "NumberxTime": NT,
        "designMatrix": dM,
        "nRow": nRow,
        "nCol": nCol,
        "nSlice": nSlice,
        "nTime": nTime,
        "nClust": nClust
    }



## Sor-értékeket meghatározó függvény
def calculate_row_score_4D(col_vector, depth_vector, time_vector, binary_tensor, alpha):
    
    col_size = np.sum(col_vector)
    depth_size = np.sum(depth_vector)
    time_size = np.sum(time_vector)
    if col_size == 0 or depth_size == 0 or time_size == 0:
        return np.zeros(binary_tensor.shape[0])
    
    current_cluster = binary_tensor[:, col_vector == 1][:, :, depth_vector == 1][:, :, :, time_vector == 1]

    cluster_vals = current_cluster.reshape(binary_tensor.shape[0], -1)

    total_size = col_size * depth_size * time_size

    p_score = np.sum(cluster_vals, axis=1)
    p1 = p_score / total_size
    p0 = 1 - p1

    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)

    entropy = np.nan_to_num(entropy)

    e_score = 1 - entropy
    p_score[p1 < 0.5] = 0

    score = p_score * (e_score ** alpha)

    return score



## Sor-információt eltávolító függvény
def remove_row_information_4D(row_tensor, col_vector, depth_vector, time_vector, alpha):

    col_mask = col_vector == 1
    depth_mask = depth_vector == 1
    time_mask = time_vector == 1

    selected = row_tensor[col_mask][:, depth_mask][:, :, time_mask]

    col_size = np.sum(col_mask)
    depth_size = np.sum(depth_mask)
    time_size = np.sum(time_mask)

    if col_size == 0 or depth_size == 0 or time_size == 0:
        return row_tensor

    total_size = col_size * depth_size * time_size

    p_score = np.sum(selected)
    p1 = p_score / total_size if total_size > 0 else 0

    if p1 >= 0.4:
        p0 = 1 - p1

        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)

        entropy = np.nan_to_num(entropy)

        weight = 1 - (1 - entropy) ** alpha

        row_tensor = row_tensor.copy()
        row_tensor[np.ix_(col_mask, depth_mask, time_mask)] *= weight

    return row_tensor



## Információt eltávolító függvény, amely a sorok infomációját iteratívan törli
def remove_information_4D(col_vector, depth_vector, time_vector, tensor, alpha):

    new_tensor = np.empty_like(tensor, dtype=float)

    for r in range(tensor.shape[0]):
        new_tensor[r] = remove_row_information_4D(
            tensor[r],
            col_vector,
            depth_vector,
            time_vector,
            alpha
        )

    return new_tensor



def linear_ranking(pos, n_ind, SP):

    probab = 2 - SP + 2 * (SP - 1) * (pos - 1) / (n_ind - 1)
    return probab



def get_cum_probabilities(probabilities):
    
    probabilities = probabilities / np.sum(probabilities)
    return np.cumsum(probabilities)



## Fő függvény, amely meghívja a C kódot a beállított paraméterek szerint, majd tárolja az eredményeket
def iBBiG4D(covTen,
            n_modules,
            alpha=0.3,
            pop_size=100,
            mutation=0.08,
            stagnation=50,
            selection_pressure=1.2,
            max_sp=15,
            success_ratio=0.6):

    n_rows, n_cols, n_depth, n_time = covTen.shape

    covTen = np.ascontiguousarray(covTen, dtype=np.float64)
    raw_tensor = covTen.copy()
    
    selection_p = np.array([linear_ranking(i+1, pop_size, selection_pressure) for i in range(pop_size)])
    sp = get_cum_probabilities(selection_p)
    
    ## Paraméterek C-kompatibilissé alakítása
    alpha_c = ctypes.c_double(alpha)
    n_cols_c = ctypes.c_int(n_cols)
    n_rows_c = ctypes.c_int(n_rows)
    n_depth_c = ctypes.c_int(n_depth)
    n_time_c = ctypes.c_int(n_time)
    pop_size_c = ctypes.c_int(pop_size)
    stag_c = ctypes.c_int(stagnation)
    mutation_c = ctypes.c_double(mutation)
    success_ratio_c = ctypes.c_double(success_ratio)
    max_sp_c = ctypes.c_int(max_sp)

    number_x_col = np.zeros((0, n_cols), dtype=bool)
    number_x_depth = np.zeros((0, n_depth), dtype=bool)
    number_x_time = np.zeros((0, n_time), dtype=bool)
    row_score_x_number = np.zeros((n_rows, 0))
    row_x_number = np.zeros((n_rows, 0), dtype=bool)
    cluster_scores = []

    for i in range(n_modules):
        
        print(f"Module: {i + 1}")
        
        ## A bemeneti mátrix/tenzor C-kompatibilissé alakítása lapítással
        covFlat = np.asfortranarray(covTen).ravel(order='F')
        group = np.zeros(n_cols + n_depth + n_time, dtype=np.int32)

        ## C függvény meghívása
        clib.clusterCovs4DC(
            covFlat,
            group,
            ctypes.byref(n_cols_c),
            ctypes.byref(n_rows_c),
            ctypes.byref(n_depth_c),
            ctypes.byref(n_time_c),
            ctypes.byref(alpha_c),
            ctypes.byref(pop_size_c),
            ctypes.byref(stag_c),
            ctypes.byref(mutation_c),
            ctypes.byref(success_ratio_c),
            ctypes.byref(max_sp_c),
            np.ascontiguousarray(sp, dtype=np.float64)
        )

        ## C kód által módosított mátrix/tenzor visszaalakítása eredeti struktúrára
        covTen = covFlat.reshape((n_rows, n_cols, n_depth, n_time), order='F')

        group_cols = group[:n_cols]
        group_depth = group[n_cols:n_cols+n_depth]
        group_time = group[n_cols+n_depth:]

        ## Klaszter értékek meghatározása, sor-értékek kiszámítása és információ eltávolítás
        if np.sum(group_cols) == 0 or np.sum(group_depth) == 0 or np.sum(group_time) == 0:
            row_score = np.zeros(n_rows)
            row_vector = np.zeros(n_rows, dtype=bool)
            cluster_score = 0
        else:
            row_score = calculate_row_score_4D(group_cols, group_depth, group_time, covTen, alpha)
            row_vector = row_score > 0
            cluster_score = np.sum(row_score)
            covTen = remove_information_4D(group_cols, group_depth, group_time, covTen, alpha)

        cluster_scores.append(cluster_score)
        row_score_x_number = np.column_stack((row_score_x_number, row_score))
        row_x_number = np.column_stack((row_x_number, row_vector))
        number_x_col = np.vstack((number_x_col, group_cols > 0))
        number_x_depth = np.vstack((number_x_depth, group_depth > 0))
        number_x_time = np.vstack((number_x_time, group_time > 0))

        print(" ... done")

    ## Osztály példányosítás
    result = iBBiGc4D(
        Seeddata = raw_tensor,
        Clusterscores = np.asarray(cluster_scores, dtype=float),
        RowScorexNumber = row_score_x_number,
        NumberxCol = number_x_col,
        NumberxDepth = number_x_depth,
        NumberxTime = number_x_time,
        RowxNumber = row_x_number,
        Number = n_modules
    )

    return result



## Bemeneti mátrix/tenzor háromdimenziós megjelenítése (1 szelet a negyedik dimenzióban)
def plot_voxels(tensor4d, slice_index, axis=3):

    tensor4d = np.asarray(tensor4d)
    
    tensor3d = np.take(tensor4d, slice_index, axis=axis)
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    filled = tensor3d == 1

    ax.voxels(filled)

    n_rows, n_cols, n_depth = tensor3d.shape

    ax.set_xlim(0, n_rows)
    ax.set_ylim(0, n_rows)
    ax.set_zlim(0, n_depth)

    ax.set_xlabel("Rows", fontsize=12)
    ax.set_ylabel("Columns", fontsize=12)
    ax.set_zlabel("Depth", labelpad = 0, fontsize=12)

    ax.set_title("Simulated Binary Tensor (index = 16)", fontsize=24)

    plt.tight_layout()
    plt.show()



## iBBiG eredmények háromdimenziós megjelenítése (1 szelet a negyedik dimenzióban)
def plot_triclusters(RowxNumber, NumberxCol, NumberxDepth, NumberxTime, 
                     matrix_shape, time_index, colors=None, alpha=1.0, figsize=(8,8)):
    
    n_rows, n_cols, n_depth, n_time = matrix_shape
    n_modules = RowxNumber.shape[1]

    colors = [
        "red",
        "green",
        "orange",
        "purple",
        "magenta",
        "yellow",
        "brown",
        "lime",
        "pink",
        "gold"
    ]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')

    voxel_matrix = np.zeros((n_rows, n_cols, n_depth), dtype=bool)
    facecolors = np.empty((n_rows, n_cols, n_depth), dtype=object)

    for i in range(n_modules):
        if not NumberxTime[i, time_index]:
            continue
        rows = np.where(RowxNumber[:, i])[0]
        cols = np.where(NumberxCol[i, :])[0]
        depths = np.where(NumberxDepth[i, :])[0]
        
        color = colors[i % len(colors)]

        for r in rows:
            for c in cols:
                for d in depths:
                    voxel_matrix[r, c, d] = True
                    facecolors[r, c, d] = color

    ax.voxels(voxel_matrix, facecolors=facecolors, edgecolor=None, alpha=alpha)

    ax.set_xlim(0, n_rows)
    ax.set_ylim(0, n_cols)
    ax.set_zlim(0, n_depth)

    ax.set_xlabel("Rows")
    ax.set_ylabel("Columns")
    ax.set_zlabel("Depth", labelpad = 0)
    
    plt.tight_layout()
    
    plt.show()

#%%

## Bemeneti mátrix/tenzor létrehozása
#sim = makeArtificial(noise=0.02,verbose=False)
sim = makeArtificial(noise=0.0,verbose=False)
binary_tensor = sim["matrix"]

## Bemeneti mátrix/tenzor megjelenítése tetszőleges index-el
plot_voxels(binary_tensor, 10)
#plot_voxels(binary_tensor, 16)

#import time
#start = time.perf_counter()

## Klaszterek meghatározása tetszőleges számú modulra
clusters = iBBiG4D(binary_tensor, n_modules=5)

#end = time.perf_counter()
#print(f"Elapsed: {end-start:.9f} s")

## Eredmények megjelenítése tetszőleges index-el
plot_triclusters(
    clusters.RowxNumber,
    clusters.NumberxCol,
    clusters.NumberxDepth,
    clusters.NumberxTime,
    matrix_shape=binary_tensor.shape,
    time_index=10
)
