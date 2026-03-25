## Szükséges könyvtárak betöltése (numpy, typing -> adatszerkezetek; matplotlib -> ábrázolás; ctypes -> C kód futtatása)
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import ctypes

## C kód beolvasása (először dll formátumra kell alakítani)
clib = ctypes.CDLL('C/ibbig.dll')

## C-kompatibilis típusok beállítása a paraméterekre
clib.clusterCovsC.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),     # covMat - bemeneti mátrix
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),       # group - klaszterhez való tartozás
    ctypes.POINTER(ctypes.c_int),                                               # noCovs - oszlopok száma (1D)
    ctypes.POINTER(ctypes.c_int),                                               # noSigs - sorok száma (2D)
    ctypes.POINTER(ctypes.c_double),                                            # alpha
    ctypes.POINTER(ctypes.c_int),                                               # noPop - populáció mérete
    ctypes.POINTER(ctypes.c_int),                                               # maxStag - maximum stagnálás
    ctypes.POINTER(ctypes.c_double),                                            # mutation - mutációs valószínűség
    ctypes.POINTER(ctypes.c_double),                                            # SR
    ctypes.POINTER(ctypes.c_int),                                               # max_SP - maximum kiválasztási lehetőség
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')      # SP - kiválasztási lehetőség
]

## Eredmények értelmezésére és ábrázolására használt osztály
class iBBiGc:
    
    def __init__(self,
                 Seeddata: Optional[np.ndarray] = None,
                 RowScorexNumber: Optional[np.ndarray] = None,
                 Clusterscores: Optional[Union[np.ndarray, List]] = None,
                 RowxNumber: Optional[np.ndarray] = None,
                 NumberxCol: Optional[np.ndarray] = None,
                 Number: int = 0,
                 info: Optional[dict] = None,
                 Parameters: Optional[dict] = None):
        self.Seeddata = np.asarray(Seeddata) if Seeddata is not None else np.zeros((0, 0))
        self.RowScorexNumber = np.asarray(RowScorexNumber) if RowScorexNumber is not None else np.zeros((self.Seeddata.shape[0], 0))
        self.Clusterscores = np.asarray(Clusterscores) if Clusterscores is not None else np.array([])
        self.RowxNumber = np.asarray(RowxNumber) if RowxNumber is not None else np.zeros((self.Seeddata.shape[0], 0), dtype=bool)
        self.NumberxCol = np.asarray(NumberxCol) if NumberxCol is not None else np.zeros((0, self.Seeddata.shape[1] if self.Seeddata.size else 0), dtype=bool)
        self.Number = int(Number)
        self.info = info or {}
        self.Parameters = Parameters or {}



## Jel-blokkokat a kezdeti mátrixhoz/tenzorhoz hozzáadó függvény
def addSignal(arti, startC, startR, endC, endR, densityH, densityL):

    numRows = endR - startR + 1
    numCols = endC - startC + 1
    diffDens = densityH - densityL
    stepD = diffDens / numRows

    for i in range(startR - 1, endR):
        prob = densityH - (i - (startR - 1)) * stepD
        random_vals = (np.random.rand(numCols) < prob).astype(int)
        arti[i, startC - 1:endC] = random_vals

    return arti



## Jel-blokkokat létrehozó függvény tetszőleges alsó- és felső sűrűséggel
def makeSimDesignMat(verbose=True):
    
    ## [10 x 10] méretű jel-blokkok
    #designMat = np.array([
    #    [2, 3, 8, 5, 1.0, 1.0],
    #    [4, 1, 6, 9, 1.0, 1.0]
    #])
    
    ## [400 x 400] méretű jel-blokkok
    designMat = np.array([
        [251, 51, 275, 300, 1.0, 1.0],
        [51, 251, 225, 325, 1.0, 1.0],
        [1, 1, 50, 50, 1.0, 1.0],
        [46, 46, 85, 85, 1.0, 1.0],
        [81, 81, 110, 110, 1.0, 1.0],
        [106, 106, 125, 125, 1.0, 1.0],
        [151, 151, 190, 190, 1.0, 1.0],
    ])
    
    ## [400 x 400] méretű jel-blokkok
    #designMat = np.array([
    #    [251, 51, 275, 300, 0.9, 0.55],
    #    [51, 251, 225, 325, 0.9, 0.55],
    #    [1, 1, 50, 50, 0.9, 0.55],
    #    [46, 46, 85, 85, 0.9, 0.55],
    #    [81, 81, 110, 110, 0.9, 0.55],
    #    [106, 106, 125, 125, 0.9, 0.55],
    #    [151, 151, 190, 190, 0.9, 0.55],
    #])

    ## Opcionális kiíratás a konzolra
    if verbose:
        rows = (designMat[:, 3] - designMat[:, 1]) + 1
        cols = (designMat[:, 2] - designMat[:, 0]) + 1
        dens_low = designMat[:, 5]
        dens_high = designMat[:, 4]
        print("***** Summary of Design Matrix ******")
        print(np.column_stack([rows, cols, dens_low, dens_high]))

    return designMat



## A végleges, jel-blokkokat tartalmazó bemeneti mátrixot/tenzort létrehozó függvény
def makeArtificial(nRow=400, nCol=400, noise=0.0, verbose=True,
                   dM=None, seed=123):
    
    if dM is None:
        dM = makeSimDesignMat(verbose=verbose)

    np.random.seed(seed)

    arti = (np.random.rand(nRow, nCol) < noise).astype(int)

    for i in range(dM.shape[0]):
        arti = addSignal(
            arti,
            int(dM[i, 0]), int(dM[i, 1]),
            int(dM[i, 2]), int(dM[i, 3]),
            dM[i, 4], dM[i, 5]
        )

    nClust = dM.shape[0]
    RN = np.zeros((nRow, nClust), dtype=bool)
    NC = np.zeros((nClust, nCol), dtype=bool)

    for i in range(nClust):
        r_start, r_end = int(dM[i, 1]) - 1, int(dM[i, 3])
        c_start, c_end = int(dM[i, 0]) - 1, int(dM[i, 2])
        RN[r_start:r_end, i] = True
        NC[i, c_start:c_end] = True

    if verbose:
        print("\nCluster sizes in simulated bicluster data:")
        print(f"Number of Modules: {nClust}")
        print("Rows per module:", RN.sum(axis=0))
        print("Cols per module:", NC.sum(axis=1))

    return {
        "matrix": arti,
        "RowxNumber": RN,
        "NumberxCol": NC,
        "designMatrix": dM,
        "nRow": nRow,
        "nCol": nCol,
        "nClust": nClust
    }



## Sor-értékeket meghatározó függvény
def calculate_row_score(col_vector, binary_matrix, alpha):
    
    col_size = np.sum(col_vector)
    if col_size == 0:
        return np.zeros(binary_matrix.shape[0])
    
    current_cluster = binary_matrix[:, col_vector == 1]
    p_score = np.sum(current_cluster, axis=1)
    p1 = p_score / col_size
    p0 = 1 - p1

    with np.errstate(divide='ignore', invalid='ignore'):
        entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
    entropy = np.nan_to_num(entropy)
    
    e_score = 1 - entropy
    p_score[p1 < 0.5] = 0
    score = p_score * (e_score ** alpha)
    
    return score



## Sor-információt eltávolító függvény
def remove_row_information(row_score, col_vector, alpha):
   
    col_size = np.sum(col_vector)
    current_cluster = row_score[col_vector == 1]
    p_score = np.sum(current_cluster)
    p1 = p_score / col_size if col_size > 0 else 0
    
    if p1 >= 0.5:
        p0 = 1 - p1
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -p1 * np.log2(p1) - p0 * np.log2(p0)
        entropy = np.nan_to_num(entropy)
        weight = 1 - (1 - entropy) ** alpha
        row_score = row_score * np.where(col_vector == 1, weight, 1)
    
    return row_score



## Információt eltávolító függvény, amely a sorok infomációját iteratívan törli
def remove_information(col_vector, covMat, alpha):
    
    new_matrix = np.apply_along_axis(remove_row_information, 1, covMat, col_vector, alpha)
    np.transpose(new_matrix)
    return new_matrix



def linear_ranking(pos, n_ind, SP):

    probab = 2 - SP + 2 * (SP - 1) * (pos - 1) / (n_ind - 1)
    return probab



def get_cum_probabilities(probabilities):
    
    probabilities = probabilities / np.sum(probabilities)
    return np.cumsum(probabilities)



## Fő függvény, amely meghívja a C kódot a beállított paraméterek szerint, majd tárolja az eredményeket
def iBBiG(covMat,
          n_modules,
          alpha=0.3,
          pop_size=100,
          mutation=0.08,
          stagnation=50,
          selection_pressure=1.2,
          max_sp=15,
          success_ratio=0.6):

    n_rows, n_cols = covMat.shape
    covMat = np.ascontiguousarray(covMat, dtype=np.float64)
    raw_matrix = covMat.copy()
    
    selection_p = np.array([linear_ranking(i + 1, pop_size, selection_pressure) for i in range(pop_size)])
    sp = get_cum_probabilities(selection_p)

    ## Paraméterek C-kompatibilissé alakítása
    alpha_c = ctypes.c_double(alpha)
    n_cols_c = ctypes.c_int(n_cols)
    n_rows_c = ctypes.c_int(n_rows)
    pop_size_c = ctypes.c_int(pop_size)
    stag_c = ctypes.c_int(stagnation)
    mutation_c = ctypes.c_double(mutation)
    success_ratio_c = ctypes.c_double(success_ratio)
    max_sp_c = ctypes.c_int(max_sp)
    
    number_x_col = np.zeros((0, n_cols), dtype=bool)
    row_score_x_number = np.zeros((n_rows, 0))
    row_x_number = np.zeros((n_rows, 0), dtype=bool)
    cluster_scores = []

    for i in range(n_modules):
        
        print(f"Module: {i + 1}")
        
        ## A bemeneti mátrix/tenzor C-kompatibilissé alakítása lapítással
        covMat = np.asfortranarray(covMat, dtype=np.float64).ravel(order='F')
        group = np.zeros(n_cols, dtype=np.int32)
        
        ## C függvény meghívása
        clib.clusterCovsC(
            covMat,
            group,
            ctypes.byref(n_cols_c),
            ctypes.byref(n_rows_c),
            ctypes.byref(alpha_c),
            ctypes.byref(pop_size_c),
            ctypes.byref(stag_c),
            ctypes.byref(mutation_c),
            ctypes.byref(success_ratio_c),
            ctypes.byref(max_sp_c),
            np.ascontiguousarray(sp, dtype=np.float64)
        )
        
        ## C kód által módosított mátrix/tenzor visszaalakítása eredeti struktúrára
        covMat = covMat.reshape((n_rows, n_cols), order='F')
        
        ## Klaszter értékek meghatározása, sor-értékek kiszámítása és információ eltávolítás
        if np.sum(group) == 0:
            row_score = np.zeros(n_rows)
            row_vector = np.zeros(n_rows, dtype=bool)
            cluster_score = 0
        else:
            row_score = calculate_row_score(group, covMat, alpha)
            row_vector = row_score > 0
            cluster_score = np.sum(row_score)
            covMat = remove_information(group, covMat, alpha)

        cluster_scores.append(cluster_score)
        row_score_x_number = np.column_stack((row_score_x_number, row_score))
        row_x_number = np.column_stack((row_x_number, row_vector))
        number_x_col = np.vstack((number_x_col, group > 0))
        
        print(" ... done")

    ## Osztály példányosítás
    result = iBBiGc(
        Seeddata = raw_matrix,
        Clusterscores = np.asarray(cluster_scores, dtype=float),
        RowScorexNumber = row_score_x_number,
        NumberxCol = number_x_col,
        RowxNumber = row_x_number,
        Number = n_modules
    )

    return result



## iBBiG eredmények kétdimenziós megjelenítése
def plot_iBBiG_blocks(Seeddata, RowxNumber, NumberxCol,
                      title="iBBiG Discovered Modules",
                      use_fill=False,
                      fill_alpha=0.35,
                      colors=None,
                      figsize=(8.75,6.25),
                      tick_interval=50):

    Seeddata = np.asarray(Seeddata)
    RowxNumber = np.asarray(RowxNumber, dtype=bool) if RowxNumber is not None else np.zeros((Seeddata.shape[0],0), dtype=bool)
    NumberxCol = np.asarray(NumberxCol, dtype=bool) if NumberxCol is not None else np.zeros((0, Seeddata.shape[1]), dtype=bool)

    if colors is None:
        base = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = list(base) + ["orange","purple","cyan","magenta","lime","brown"]

    n_modules = RowxNumber.shape[1] if RowxNumber.size else 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(Seeddata, cmap="gray_r", aspect='auto', origin='upper')

    drawn_modules = []

    for i in range(n_modules):
        row_mask = RowxNumber[:, i]
        col_mask = NumberxCol[i, :]
        if not row_mask.any() or not col_mask.any():
            continue
        color = colors[i % len(colors)]

        rows = np.where(row_mask)[0]
        cols = np.where(col_mask)[0]

        for r in rows:
            for c in cols:
                if Seeddata[r, c] > 0:
                    if use_fill:
                        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                             facecolor=color, edgecolor=None,
                                             alpha=fill_alpha, linewidth=0)
                    else:
                        rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                             facecolor='none', edgecolor=color,
                                             linewidth=1.2)
                    ax.add_patch(rect)

        drawn_modules.append(i)

    ax.set_title(title, fontsize=30, fontweight="bold")
    
    n_rows, n_cols = Seeddata.shape
    ax.set_xticks(np.arange(0, n_cols, tick_interval))
    ax.set_yticks(np.arange(0, n_rows, tick_interval))
    ax.tick_params(axis='both', which='both', labelsize=16)

    plt.tight_layout()
    plt.show()

#%%

## Bemeneti mátrix/tenzor létrehozása
sim = makeArtificial(noise=0.0,verbose=False)
binary_matrix = sim["matrix"]

## Bemeneti mátrix/tenzor megjelenítése
plt.imshow(binary_matrix, cmap="gray_r", aspect="auto")
plt.title("Simulated Binary Matrix (Artificial Data)", fontsize = 15, fontweight = "bold")
plt.show()

#import time
#start = time.perf_counter()

## Klaszterek meghatározása tetszőleges számú modulra
clusters = iBBiG(binary_matrix, n_modules=7)

#end = time.perf_counter()
#print(f"Elapsed: {end-start:.9f} s")

## Eredmények megjelenítése
plot_iBBiG_blocks(clusters.Seeddata,clusters.RowxNumber,clusters.NumberxCol)
