from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# sns.set_palette('Set4')
# .....               +  commenti/print e pulizia progetto/controlli vari

def stampa_info(dataset, num_rig=20):
    print('\nPrime ' + str(num_rig) + ' righe del dataset:\t' + (dataset.dataframeName if
          hasattr(dataset, 'dataframeName') else '') + '\n')
    print(dataset.head(num_rig))
    print('\n\nCampi descritti dalle colonne del dataset:\n', dataset.columns)
    print('\nDimensioni del dataset:\t', dataset.shape)
    print('\n\nStatistiche dataset:\n')
    print(dataset.describe(include='all'))
    print('\n\nInformazioni dataset:')
    print(dataset.info())
    print('\n\n')



def controlla_val(dataset):
    print('\nDataset:\n')
    print(dataset.head())
    print('\n\nControllo presenza valori nulli per colonna:\n')
    print(dataset.isnull().sum())
    print('\nControllo numero tuple ridondanti:\n')
    print(dataset.duplicated().sum())
    print('\n')



def stampa_num_val(dataset, col):
    print('Numero valori nelle colonne di tipo categorico\n')
    for i in col:
        print(f'\n+ -\t\t-- \tConteggio valori di {i}: \t--\t\t- +\n\n{dataset[i].value_counts()}\n'
              f'\n+------------------------------------------------------------+\n')
    print('\n\n')



def codifica_dati(dati, col_interessate):
    le = LabelEncoder()
    ds = dati.copy()

    for col in col_interessate:
        val_ma = ds.index[ds[col].isnull()]
        ds[col] = le.fit_transform(ds[col])

        i = 0
        while i < len(val_ma):
            ds.at[val_ma[i], col] = np.NaN
            i += 1

    return ds



def scala_dati(col, dati, dati_test=None):
    norm = MinMaxScaler()
    stan = StandardScaler()

    if dati_test is None:
        dati_n = dati.copy()
        dati_s = dati.copy()

        dati_n[col] = norm.fit_transform(dati[col])
        dati_s[col] = stan.fit_transform(dati[col])

        return dati_n, dati_s

    else:

        dati_n1 = dati.copy()
        dati_n2 = dati_test.copy()
        dati_s1 = dati.copy()
        dati_s2 = dati_test.copy()

        dati_n1[col] = norm.fit_transform(dati[col])
        dati_n2[col] = norm.transform(dati_test[col])
        dati_s1[col] = stan.fit_transform(dati[col])
        dati_s2[col] = stan.transform(dati_test[col])

        return dati_n1, dati_n2, dati_s1, dati_s2



def integra_val_mancanti(dataset, col_interessate, k=3):
    norm = MinMaxScaler()
    elab = KNNImputer(n_neighbors=k)

    dati_elab = dataset.copy()
    dati_elab[col_interessate] = pd.DataFrame(norm.fit_transform(dati_elab[col_interessate]), columns=col_interessate)
    dati_elab = pd.DataFrame(elab.fit_transform(dati_elab), columns=dati_elab.columns)
    dati_elab[col_interessate] = pd.DataFrame(norm.inverse_transform(dati_elab[col_interessate]), columns=col_interessate)

    return dati_elab



def osserva_test_modello(mod, X_test, y_test):
    pred = mod.predict(X_test)

    plt.figure(figsize=(7, 7))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Purples')
    plt.xlabel("Val Predetti")
    plt.ylabel("Val Reali")

    print('\n+------------------------------------------------------------------------------------------+')
    print('\n+\tRisultati test per\t ' + (mod.__annotations__ if hasattr(mod, '__annotations__') else str(mod)) + ' :\n\n')
    print(classification_report(y_test, pred))
    print('+------------------------------------------------------------------------------------------+\n')

    plt.show()
    plt.close()



def osserva_metriche_classificatore(val_risultanti, mod=None, metriche_test=None, metriche_train=None):
    if metriche_train is None and metriche_test is None:
        metriche_train = ['train_accuracy', 'train_precision', 'train_recall', 'train_f1']
        metriche_test = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']

    print('\n\n <>  Valutazione di ' + (mod.__annotations__ if hasattr(mod, '__annotations__') else str(mod)) + ' : ')
    print('\t\t\t\t\t\t\t\t\t\t\t\t[ numero valori per metrica: ' + str(len(val_risultanti[list(val_risultanti.keys())[0]])) + ' ]\n\n')

    print('> Risultati metriche su test set:')
    for val in metriche_test:
        print('\n\t' + val.upper() + ' :\t\t' + str(val_risultanti[val]))
        print('\t' + val.upper() + ' Media: ' + str(round(val_risultanti[val].mean(), 3)) + '  \tMax: ' +
              str(round(max(val_risultanti[val]), 3)) + '  \tMin: ' + str(round(min(val_risultanti[val]), 3)) + '\n')

    if metriche_train is not None:
        stampa = '\t[ '
        print('\n\n\n< Risultati metriche su train set:\n')
        for val in metriche_train:
            stampa = stampa + val.upper() + ' Medio: ' + str(round(val_risultanti[val].mean(), 3)) + ' \t'
        stampa = stampa + ' ] \n'
        print(stampa)

    val_x = [k for k in range(1, len(val_risultanti[list(val_risultanti.keys())[0]]) + 1)]
    print('\n   <>\n\n')

    plt.title('Visualizzazione variazione valori per ' + metriche_test[len(metriche_test) - 1].capitalize() + ':       '
              '  [media: ' + str(round(val_risultanti[metriche_test[len(metriche_test) - 1]].mean(), 2)) + ']')
    plt.plot(val_x, val_risultanti[metriche_test.pop()], color='m', marker='D')
    plt.xticks(ticks=val_x, labels=val_x)
    plt.grid()
    plt.show()
    plt.close()



def visualizza_bilanciamento(col, desc='Controllo bilanciamento dei valori:'):
    dct = col.value_counts(ascending=True).to_dict()
    genere = list(dct.keys())
    numero = list(dct.values())

    plt.figure(figsize=(10, 5))
    plt.title(desc)
    plt.hlines(y=genere, xmin=0, xmax=numero, color='slateblue')
    plt.plot(numero, genere, 'o', color='slateblue')
    plt.tick_params(left=False)
    sns.despine(left=True)
    plt.show()
    plt.close()



def visualizza_distribuzione_conteggio(col, desc='Controllo distribuzione dei valori:'):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle(desc)
    plt.subplots_adjust(wspace=0.6)
    ax[0].pie(np.array(col.iloc[:, 0].value_counts()), labels=list(col.iloc[:, 0].value_counts().to_dict().keys()), autopct='%1.2f%%', shadow=True)
    sns.violinplot(x=col.iloc[:, 0], y=col.iloc[:, 1], hue=(col.iloc[:, 2] if len(col.columns) == 3 else None), split=True, inner='quartile', ax=ax[1])
    plt.show()
    plt.close()



def mostra_conteggi(col_interessate, dataset, evidenzia=None):
    num_val = len(col_interessate)
    num_it = 1
    da_mostrare = 4

    if num_val > 4:
        num_it = (num_val - (num_val % 4))/4
        if num_val % 4 != 0:
            num_it += 1

    i = 0
    col_mostrate = 0
    while i < num_it:
        if (col_mostrate + 4) > num_val:
            da_mostrare = num_val - col_mostrate



        fig, ax = plt.subplots(da_mostrare, 2, figsize=(17, 17))
        plt.suptitle('Conteggio valori colonne rispetto \'' + str(evidenzia) + '\'      [' + str(int(i+1)) + '/'
                     + str(int(num_it)) + ']', size=17)

        if da_mostrare == 1:
            plt.subplots_adjust(top=0.75, bottom=0.5)
            sns.countplot(data=dataset, y=col_interessate[col_mostrate], ax=ax[0])
            sns.countplot(data=dataset, y=col_interessate[col_mostrate], ax=ax[1], hue=evidenzia)

        else:
            plt.subplots_adjust(wspace=0.6, hspace=0.7, bottom=0.06, left=0.12, right=0.92)

            j = 0
            while j < da_mostrare:
                sns.countplot(data=dataset, y=col_interessate[col_mostrate + j], ax=ax[j, 0])
                sns.countplot(data=dataset, y=col_interessate[col_mostrate + j], ax=ax[j, 1], hue=evidenzia)
                j += 1



        plt.show()
        plt.close()
        col_mostrate += 4
        i += 1



def rivela_dispersione_ver3d(dati, etichette, larghezza=10, lunghezza=7, titolo='Dispersione in 3D:'):
    fig = plt.figure(figsize=(larghezza, lunghezza))
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.text2D(0.05, 0.95, titolo, transform=ax.transAxes, fontsize=15)

    if len(etichette) == 4:
        labels = dati[etichette[3]].unique()[0]
        colors = (LabelEncoder().fit_transform(dati[etichette[3]]) if dati[etichette[3]].dtype == 'object' else dati[etichette[3]].to_numpy())
    else:
        labels = None
        colors = labels

    ax.scatter(dati[etichette[0]], dati[etichette[1]], dati[etichette[2]], c=colors, s=15, label=labels)
    ax.set_xlabel(etichette[0])
    ax.set_ylabel(etichette[1])
    ax.set_zlabel(etichette[2])
    ax.legend()
    plt.show()
    plt.close()



def mostra_distribuzioni(col_interessate, dataset, evidenzia=None):
    dati_vis = dataset.copy()
    dati_vis['fake_col'] = ''
    num_val = len(col_interessate)
    num_it = 1
    da_mostrare = 3

    if num_val > 3:
        num_it = (num_val - (num_val % 3)) / 3
        if num_val % 3 != 0:
            num_it += 1

    i = 0
    col_mostrate = 0
    while i < num_it:
        if (col_mostrate + 3) > num_val:
            da_mostrare = num_val - col_mostrate



        fig, ax = plt.subplots(da_mostrare, 3, figsize=(33, 33))
        plt.suptitle('Distribuzione valori colonne con \'' + str(evidenzia) + '\' in evidenza          [' + str(int(i+1))
                     + '/' + str(int(num_it)) + ']', size=17)

        if da_mostrare == 1:
            plt.subplots_adjust(top=0.75, bottom=0.5)
            sns.histplot(data=dati_vis, x=col_interessate[col_mostrate], ax=ax[0], kde=True, hue=evidenzia)
            sns.boxplot(data=dati_vis, y=col_interessate[col_mostrate], ax=ax[1], hue=evidenzia, orient='v',
                        notch=True, flierprops=dict(markerfacecolor='r', marker='D'), width=0.4)
            ax[1].set_xlabel(col_interessate[col_mostrate])
            ax[1].set_ylabel('')
            ax[1].grid(axis='y', linestyle='--', visible=True)
            sns.violinplot(data=dati_vis, x=col_interessate[col_mostrate], y='fake_col', ax=ax[2], split=True, hue=evidenzia)
            ax[2].set_ylabel('')

        else:
            plt.subplots_adjust(hspace=0.35, bottom=0.07)

            j = 0
            while j < da_mostrare:
                sns.histplot(data=dati_vis, x=col_interessate[col_mostrate + j], ax=ax[j, 0], kde=True, hue=evidenzia)
                sns.boxplot(data=dati_vis, y=col_interessate[col_mostrate + j], ax=ax[j, 1], hue=evidenzia, orient='v',
                            notch=True, flierprops=dict(markerfacecolor='r', marker='D'), width=0.4)
                ax[j, 1].set_xlabel(col_interessate[col_mostrate + j])
                ax[j, 1].set_ylabel('')
                ax[j, 1].grid(axis='y', linestyle='--', visible=True)
                sns.violinplot(data=dati_vis, x=col_interessate[col_mostrate + j], y='fake_col', ax=ax[j, 2], split=True, hue=evidenzia)
                ax[j, 2].set_ylabel('')
                j += 1



        plt.show()
        plt.close()
        col_mostrate += 3
        i += 1