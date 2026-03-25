## Szükséges könyvtárak betöltése (numpy, typing -> adatszerkezetek, matplotlib -> ábrázolás, ctypes -> C kód futtatása)
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import ctypes

## C kód beolvasása (először dll formátumra kell hozni)
clib = ctypes.CDLL('C/ibbig3d.dll')

## C-kompatibilis típusok beállítása a paraméterekre
clib.clusterCovs3DC.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),      # covTen - bemeneti tenzor
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),        # group - klaszterhez való tartozás
    ctypes.POINTER(ctypes.c_int),                                                # noCovs - oszlopok száma (1D)
    ctypes.POINTER(ctypes.c_int),                                                # noSigs - sorok száma (2D)
    ctypes.POINTER(ctypes.c_int),                                                # noSlices - szeletek száma (3D)
    ctypes.POINTER(ctypes.c_double),                                             # alpha
    ctypes.POINTER(ctypes.c_int),                                                # noPop - populáció mérete
    ctypes.POINTER(ctypes.c_int),                                                # maxStag - maximum stagnálás
    ctypes.POINTER(ctypes.c_double),                                             # mutation - mutációs valószínűség
    ctypes.POINTER(ctypes.c_double),                                             # SR
    ctypes.POINTER(ctypes.c_int),                                                # max_SP - maximum kiválasztási valószínűség
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')       # SP - kiválasztási valószínűség
]

## Eredmények értelmezésére és ábrázolására használt osztály
class iBBiGc3D:
    
    def __init__(self,
                 Seeddata: Optional[np.ndarray] = None,
                 RowScorexNumber: Optional[np.ndarray] = None,
                 Clusterscores: Optional[Union[np.ndarray, List]] = None,
                 RowxNumber: Optional[np.ndarray] = None,
                 NumberxCol: Optional[np.ndarray] = None,
                 NumberxDepth: Optional[np.ndarray] = None,
                 Number: int = 0,
                 info: Optional[dict] = None,
                 Parameters: Optional[dict] = None):
        self.Seeddata = np.asarray(Seeddata) if Seeddata is not None else np.zeros((0, 0))
        self.RowScorexNumber = np.asarray(RowScorexNumber) if RowScorexNumber is not None else np.zeros((self.Seeddata.shape[0], 0))
        self.Clusterscores = np.asarray(Clusterscores) if Clusterscores is not None else np.array([])
        self.RowxNumber = np.asarray(RowxNumber) if RowxNumber is not None else np.zeros((self.Seeddata.shape[0], 0), dtype=bool)
        self.NumberxCol = np.asarray(NumberxCol) if NumberxCol is not None else np.zeros((0, self.Seeddata.shape[1] if self.Seeddata.size else 0), dtype=bool)
        self.NumberxDepth = np.asarray(NumberxDepth) if NumberxDepth is not None else np.zeros((0, self.Seeddata.shape[2] if self.Seeddata.size else 0), dtype=bool)
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
def addSignal(arti, startC, startR, startS, endC, endR, endS, densityH, densityL):

    numRows = endR - startR + 1
    numCols = endC - startC + 1
    numSlices = endS - startS + 1
    diffDens = densityH - densityL
    stepD = diffDens / numRows

    for i in range(startR - 1, endR):
        prob = densityH - (i - (startR - 1)) * stepD
        random_vals = (np.random.rand(numCols, numSlices) < prob).astype(int)
        arti[i, startC - 1:endC, startS - 1:endS] = random_vals

    return arti



## Jel-blokkokat létrehozó függvény tetszőleges alsó- és felső sűrűséggel
def makeSimDesignMat(verbose=True):
    
    ## [10 x 10 x 10] méretű jel-blokkok
    #designMat = np.array([
    #    [1, 1, 1, 8, 3, 5, 1.0, 1.0],
    #    [4, 6, 3, 9, 8, 8, 1.0, 1.0],
    #    [2, 3, 4, 3, 5, 10, 1.0, 1.0],
    #    [1, 6, 1, 5, 10, 1, 1.0, 1.0],
    #    [9, 1, 6, 10, 2, 10, 1.0, 1.0],
    #    ])
    
    ## [20 x 20 x 20] méretű jel-blokkok
    designMat = np.array([
        [1, 1, 1, 16, 6, 10, 1.0, 1.0],
        [4, 15, 8, 18, 18, 16, 1.0, 1.0],
        [4, 5, 8, 6, 10, 20, 1.0, 1.0],
        [1, 10, 1, 10, 20, 2, 1.0, 1.0],
        [18, 1, 12, 20, 4, 20, 1.0, 1.0],
        [12, 9, 15, 15, 12, 18, 1.0, 1.0]
        ])
    
    ## [20 x 20 x 20] méretű jel-blokkok
    #designMat = np.array([
    #    [1, 1, 1, 16, 6, 10, 0.95, 0.9],
    #    [4, 15, 8, 18, 18, 16, 0.95, 0.9],
    #    [4, 5, 8, 6, 10, 20, 0.95, 0.9],
    #    [1, 10, 1, 10, 20, 2, 0.95, 0.9],
    #    [18, 1, 12, 20, 4, 20, 0.95, 0.9],
    ##   [12, 9, 15, 15, 12, 18, 1.0, 1.0]
    #    ])
    
    ## [100 x 100 x 100] méretű jel-blokkok
    #designMat = np.array([
    #    [62, 12, 12, 93, 75, 37, 1.0, 1.0],
    #    [12, 62, 12, 56, 81, 37, 1.0, 1.0],
    #    [1, 1, 12, 12, 12, 70, 1.0, 1.0],
    #    [23, 23, 25, 42, 42, 70, 1.0, 1.0],
    #    [40, 40, 25, 55, 55, 70, 1.0, 1.0],
    #    [53, 53, 25, 62, 62, 70, 1.0, 1.0],
    #    [70, 70, 25, 95, 95, 70, 1.0, 1.0],
    #])

    ## Opcionális kiíratás a konzolra
    if verbose:
        rows = (designMat[:, 4] - designMat[:, 1]) + 1
        cols = (designMat[:, 3] - designMat[:, 0]) + 1
        slices = (designMat[:, 5] - designMat[:, 2]) + 1
        dens_low = designMat[:, 7]
        dens_high = designMat[:, 6]
        print("***** Summary of Design Matrix ******")
        print(np.column_stack([rows, cols, slices, dens_low, dens_high]))

    return designMat



## A végleges, jel-blokkokat tartalmazó bemeneti mátrixot/tenzort létrehozó függvény
def makeArtificial(nRow=20, nCol=20, nSlice=20, noise=0.0, verbose=True,
                   dM=None, seed=123):
    
    if dM is None:
        dM = makeSimDesignMat(verbose=verbose)

    np.random.seed(seed)

    arti = (np.random.rand(nRow, nCol, nSlice) < noise).astype(int)

    for i in range(dM.shape[0]):
        arti = addSignal(
            arti,
            int(dM[i, 0]), int(dM[i, 1]),
            int(dM[i, 2]), int(dM[i, 3]),
            int(dM[i, 4]), int(dM[i, 5]),
            dM[i, 6], dM[i, 7]
        )

    nClust = dM.shape[0]
    RN = np.zeros((nRow, nClust), dtype=bool)
    NC = np.zeros((nClust, nCol), dtype=bool)
    NS = np.zeros((nClust, nSlice), dtype=bool)

    for i in range(nClust):
        r_start, r_end = int(dM[i, 1]) - 1, int(dM[i, 4])
        c_start, c_end = int(dM[i, 0]) - 1, int(dM[i, 3])
        s_start, s_end = int(dM[i, 2]) - 1, int(dM[i, 5])
        RN[r_start:r_end, i] = True
        NC[i, c_start:c_end] = True
        NS[i, s_start:s_end] = True

    if verbose:
        print("\nCluster sizes in simulated bicluster data:")
        print(f"Number of Modules: {nClust}")
        print("Rows per module:", RN.sum(axis=0))
        print("Cols per module:", NC.sum(axis=1))
        print("Slices per module:", NS.sum(axis=2))

    return {
        "matrix": arti,
        "RowxNumber": RN,
        "NumberxCol": NC,
        "NumberxSlice": NS,
        "designMatrix": dM,
        "nRow": nRow,
        "nCol": nCol,
        "nSlice": nSlice,
        "nClust": nClust
    }



## Sor-értékeket meghatározó függvény
def calculate_row_score_3D(col_vector, depth_vector, binary_tensor, alpha):
    
    col_size = np.sum(col_vector)
    depth_size = np.sum(depth_vector)
    if col_size == 0 or depth_size == 0:
        return np.zeros(binary_tensor.shape[0])
    
    current_cluster = binary_tensor[:, col_vector == 1][:, :, depth_vector == 1]

    cluster_vals = current_cluster.reshape(binary_tensor.shape[0], -1)

    total_size = col_size * depth_size

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
def remove_row_information_3D(row_tensor, col_vector, depth_vector, alpha):

    col_mask = col_vector == 1
    depth_mask = depth_vector == 1

    selected = row_tensor[col_mask][:, depth_mask]

    col_size = np.sum(col_mask)
    depth_size = np.sum(depth_mask)

    if col_size == 0 or depth_size == 0:
        return row_tensor

    total_size = col_size * depth_size

    p_score = np.sum(selected)
    p1 = p_score / total_size if total_size > 0 else 0

    if p1 >= 0.5:
        p0 = 1 - p1

        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)

        entropy = np.nan_to_num(entropy)

        weight = 1 - (1 - entropy) ** alpha

        row_tensor = row_tensor.copy()
        row_tensor[np.ix_(col_mask, depth_mask)] *= weight

    return row_tensor



## Információt eltávolító függvény, amely a sorok infomációját iteratívan törli
def remove_information_3D(col_vector, depth_vector, tensor, alpha):

    new_tensor = np.empty_like(tensor, dtype=float)

    for r in range(tensor.shape[0]):
        new_tensor[r] = remove_row_information_3D(
            tensor[r],
            col_vector,
            depth_vector,
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
def iBBiG3D(covTen,
            n_modules,
            alpha=0.3,
            pop_size=100,
            mutation=0.08,
            stagnation=50,
            selection_pressure=1.2,
            max_sp=15,
            success_ratio=0.6):

    n_rows, n_cols, n_depth = covTen.shape

    covTen = np.ascontiguousarray(covTen, dtype=np.float64)
    raw_tensor = covTen.copy()
    
    selection_p = np.array([linear_ranking(i+1, pop_size, selection_pressure) for i in range(pop_size)])
    sp = get_cum_probabilities(selection_p)

    ## Paraméterek C-kompatibilissé alakítása
    alpha_c = ctypes.c_double(alpha)
    n_cols_c = ctypes.c_int(n_cols)
    n_rows_c = ctypes.c_int(n_rows)
    n_depth_c = ctypes.c_int(n_depth)
    pop_size_c = ctypes.c_int(pop_size)
    stag_c = ctypes.c_int(stagnation)
    mutation_c = ctypes.c_double(mutation)
    success_ratio_c = ctypes.c_double(success_ratio)
    max_sp_c = ctypes.c_int(max_sp)

    number_x_col = np.zeros((0, n_cols), dtype=bool)
    number_x_depth = np.zeros((0, n_depth), dtype=bool)
    row_score_x_number = np.zeros((n_rows, 0))
    row_x_number = np.zeros((n_rows, 0), dtype=bool)
    cluster_scores = []

    for i in range(n_modules):
        print(f"Module: {i + 1}")
        
        ## A bemeneti mátrix/tenzor C-kompatibilissé alakítása lapítással
        covFlat = np.asfortranarray(covTen).ravel(order='F')
        group = np.zeros(n_cols + n_depth, dtype=np.int32)

        ## C függvény meghívása
        clib.clusterCovs3DC(
            covFlat,
            group,
            ctypes.byref(n_cols_c),
            ctypes.byref(n_rows_c),
            ctypes.byref(n_depth_c),
            ctypes.byref(alpha_c),
            ctypes.byref(pop_size_c),
            ctypes.byref(stag_c),
            ctypes.byref(mutation_c),
            ctypes.byref(success_ratio_c),
            ctypes.byref(max_sp_c),
            np.ascontiguousarray(sp, dtype=np.float64)
        )

        ## C kód által módosított mátrix/tenzor visszaalakítása eredeti struktúrára
        covTen = covFlat.reshape((n_rows, n_cols, n_depth), order='F')

        group_cols = group[:n_cols]
        group_depth = group[n_cols:]

        ## Klaszter értékek meghatározása, sor-értékek kiszámítása és információ eltávolítás
        if np.sum(group_cols) == 0 or np.sum(group_depth) == 0:
            row_score = np.zeros(n_rows)
            row_vector = np.zeros(n_rows, dtype=bool)
            cluster_score = 0
        else:
            row_score = calculate_row_score_3D(group_cols, group_depth, covTen, alpha)
            row_vector = row_score > 0
            cluster_score = np.sum(row_score)
            covTen = remove_information_3D(group_cols, group_depth, covTen, alpha)

        cluster_scores.append(cluster_score)
        row_score_x_number = np.column_stack((row_score_x_number, row_score))
        row_x_number = np.column_stack((row_x_number, row_vector))
        number_x_col = np.vstack((number_x_col, group_cols > 0))
        number_x_depth = np.vstack((number_x_depth, group_depth > 0))

        print(" ... done")

    ## Osztály példányosítás
    result = iBBiGc3D(
        Seeddata = raw_tensor,
        Clusterscores = np.asarray(cluster_scores, dtype=float),
        RowScorexNumber = row_score_x_number,
        NumberxCol = number_x_col,
        NumberxDepth = number_x_depth,
        RowxNumber = row_x_number,
        Number = n_modules
    )

    return result



## Bemeneti mátrix/tenzor háromdimenziós megjelenítése
def plot_voxels(matrix3d):

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')

    filled = matrix3d == 1

    ax.voxels(filled)

    n_rows, n_cols, n_depth = matrix3d.shape

    ax.set_xlim(0, n_rows)
    ax.set_ylim(0, n_rows)
    ax.set_zlim(0, n_depth)

    ax.set_xlabel("Rows")
    ax.set_ylabel("Columns")
    ax.set_zlabel("Depth", labelpad = 0)

    plt.tight_layout()

    plt.show()



## iBBiG eredmények háromdimenziós megjelenítése
def plot_triclusters(RowxNumber, NumberxCol, NumberxDepth, matrix_shape,
                     colors=None, alpha=1.0, figsize=(8,8)):
    
    n_rows, n_cols, n_depth = matrix_shape
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

    voxel_matrix = np.zeros(matrix_shape, dtype=bool)
    facecolors = np.empty(matrix_shape, dtype=object)

    for i in range(n_modules):
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

## Bemeneti mátrix/tenzor megjelenítése
plot_voxels(binary_tensor)

#import time
#start = time.perf_counter()

## Klaszterek meghatározása tetszőleges számú modulra
clusters = iBBiG3D(binary_tensor, n_modules=7)

#end = time.perf_counter()
#print(f"Elapsed: {end-start:.9f} s")

## Eredmények megjelenítése
plot_triclusters(
    clusters.RowxNumber, 
    clusters.NumberxCol, 
    clusters.NumberxDepth, 
    matrix_shape=binary_tensor.shape
)
