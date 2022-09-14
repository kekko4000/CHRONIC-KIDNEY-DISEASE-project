import warnings 
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics.cluster import completeness_score, v_measure_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.model_selection import cross_val_score, train_test_split


from utili_varie import *

percorso_dati = 'kidney_disease.csv'
warnings.filterwarnings('ignore')


dataset = pd.read_csv(percorso_dati)
dataset.dataframeName = 'Dati Pazienti A Rischio Malattie Renali'




stampa_info(dataset)
controlla_val(dataset)

dataset = dataset.drop('id', axis=1)
dataset.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                   'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                   'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                   'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema',
                   'anemia', 'class']

dataset['packed_cell_volume'] = pd.to_numeric(dataset['packed_cell_volume'], errors='coerce')
dataset['white_blood_cell_count'] = pd.to_numeric(dataset['white_blood_cell_count'], errors='coerce')
dataset['red_blood_cell_count'] = pd.to_numeric(dataset['red_blood_cell_count'], errors='coerce')



col_categoriche = [col for col in dataset.columns if (dataset[col].dtype == 'object' or col == 'albumin' or
                   col == 'sugar' or col == 'specific_gravity')]
col_continue = [col for col in dataset.columns if col not in col_categoriche]
print('\nLe colonne di tipologia categorica sono:\n')
print(col_categoriche)
print('\n')
print('Le colonne di tipologia continua sono:\n')
print(col_continue)
print('\n\n')

for col in col_categoriche:
    print('Valori contenuti in ' + str(col) + ' :\t ' + str(dataset[col].unique()))
dataset['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
dataset['coronary_artery_disease'].replace(to_replace={'\tno': 'no'}, inplace=True)
dataset['class'].replace(to_replace={'ckd\t': 'ckd'}, inplace=True)


stampa_info(dataset)
visualizza_bilanciamento(dataset['bacteria'], 'Confronto numero pazienti per presenza di batteri:')
visualizza_bilanciamento(dataset['anemia'], 'Confronto numero pazienti per anemia:')
visualizza_bilanciamento(dataset['specific_gravity'], 'Confronto numero pazienti per livello di densità relativa:')
visualizza_distribuzione_conteggio(dataset[['hypertension', 'age', 'class']], 
                                   'Distribuzione problemi d\'ipertensione rispetto età pazienti:')
visualizza_distribuzione_conteggio(dataset[['anemia', 'haemoglobin', 'class']], 
                                   'Distribuzione casi di anemia rispetto livelli di emoglobina:')
visualizza_distribuzione_conteggio(dataset[['bacteria', 'white_blood_cell_count', 'class']], 
                                   'Distribuzione presenza di batteri rispetto numero di globuli bianchi:')
visualizza_distribuzione_conteggio(dataset[['coronary_artery_disease', 'packed_cell_volume', 'class']], 
                                   'Distribuzione CAD rispetto quantità di cellule nel sangue:')

plt.figure(figsize=(7, 7))
plt.title('Conteggio numero casi malattie renali:', x=0.5, y=1.1)
plt.pie(np.array(dataset['class'].value_counts()), labels=['ckd', 'not ckd'], explode=[0.1, 0], autopct='%1.1f%%', shadow=True)
plt.show()
plt.close()


mostra_conteggi(col_categoriche, dataset, 'class')
stampa_num_val(dataset, col_categoriche)
rivela_dispersione_ver3d(dataset, ['age', 'blood_pressure', 'haemoglobin'] + list(['class']), 11, 10, 'Visualizzazione di alcuni valori continui:          [1/4]')
rivela_dispersione_ver3d(dataset, ['red_blood_cell_count', 'blood_urea', 'blood_glucose_random'] + list(['class']), 11, 10, '[2/4]')
rivela_dispersione_ver3d(dataset, ['sodium', 'serum_creatinine', 'potassium'] + list(['class']), 11, 10, '[3/4]')               # class: 0 -> ckd --> purple
rivela_dispersione_ver3d(dataset, ['blood_pressure', 'white_blood_cell_count', 'packed_cell_volume'] + list(['class']), 11, 10, '[4/4]')
mostra_distribuzioni(col_continue, dataset, 'class')

sns.pairplot(data=dataset, hue='class', aspect=1.5, markers=['D', 'X'], plot_kws={'s': 15}, corner=True)
plt.show()
plt.close()

plt.figure(figsize=(11, 11))
sns.heatmap(dataset.corr(), annot=True, linewidth=1.7, fmt='0.2f', cmap='Purples')
plt.show()
plt.close()



dataset = codifica_dati(dataset, [col for col in dataset.columns if dataset[col].dtype == 'object'])
dataset = integra_val_mancanti(dataset, col_continue, 2)
controlla_val(dataset)
stampa_info(dataset)

X = dataset.drop(['class'], axis=1)
y = dataset['class']



# Clustering:

df_utile = X.copy()
df_norm, df_stan = scala_dati(col_continue, df_utile)
ds_clustering = df_stan.copy()

i = 2
pca_test = None
while i < 10:
    pca_test = PCA(i)
    pca_test.fit_transform(ds_clustering)
    i += 1

print('Autovalori:')
print(pca_test.explained_variance_)
print('\n\n')
plt.title('Scree Plot:')
plt.plot(pca_test.explained_variance_, marker='o')
plt.xlabel('Numero Autovalori:')
plt.ylabel('Grandezza Autovalori:')
plt.show()
plt.close()


ssd = []
poss_numero_clusters = [2,3,4,5,6,7,8,9,10,11]
pca = PCA(5)
df_kmed = pca.fit_transform(ds_clustering)

for num_clusters in poss_numero_clusters:
    kmedoids = KMedoids(n_clusters=num_clusters, method='pam', max_iter=100, init='k-medoids++', random_state=1)
    kmedoids.fit(df_kmed)
    ssd.append(kmedoids.inertia_)

    media_silhouette = silhouette_score(df_kmed, kmedoids.labels_)
    print('Con n_clusters={0}, il valore di silhouette {1}'.format(num_clusters, media_silhouette))

print('\n\n')
plt.title('Curva a gomito:')
plt.plot(ssd)
plt.grid()
plt.show()
plt.close()

kmedoids = KMedoids(n_clusters=3, method='pam', max_iter=100, init='k-medoids++', random_state=1)
label = kmedoids.fit_predict(df_kmed)
etichette_kmed = np.unique(label)
df_kmed = np.array(df_kmed)


plt.figure(figsize=(9, 9))
plt.title('Clustering con k-Medoids: ')
for k in etichette_kmed:
    plt.scatter(df_kmed[label == k, 0], df_kmed[label == k, 1], label=k)
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:, 1], s=200, c='k', label='Medoide')
plt.legend()
plt.show()
plt.close()

ds_clustering['cluster'] = label

fig = plt.figure(figsize=(11, 10))
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.text2D(0.05, 0.95, 'Clusters:', transform=ax.transAxes, fontsize=15)
ax.scatter(df_kmed[:, 0], df_kmed[:, 1], df_kmed[:, 2], c=ds_clustering['cluster'].to_numpy(), s=20)
plt.show()
plt.close()

print('\n\nValutazione:\n')
print('Omogeneità  : ', homogeneity_score(y, label))
print('Completezza : ', completeness_score(y, label))
print('V_measure   : ', v_measure_score(y, label))



# Classificazione:

classificatori = []
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=54)
Xn_train, Xn_test, Xs_train, Xs_test = scala_dati(col_continue, X_train, X_test)

risultati_Knn = []
for k in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    valutazioni = cross_val_score(knn, Xn_train, y_train, cv=5, scoring='f1')
    risultati_Knn.append(valutazioni.mean())
val_x = [k for k in range(1, 20)]
plt.plot(val_x, risultati_Knn, color='g')
plt.xticks(ticks=val_x, labels=val_x)
plt.grid()
plt.show()
plt.close()

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(Xn_train, y_train)
knn.__annotations__ = 'K-NearestNeighbors clf           [1]'
classificatori.append(knn)
osserva_test_modello(knn, Xn_test, y_test)


parametri = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
svm = RandomizedSearchCV(SVC(), parametri, scoring='f1')
svm.fit(Xn_train, y_train)
svm = svm.best_estimator_
svm.__annotations__ = 'C-SupportVectorMachine clf       [2]'
classificatori.append(svm)
osserva_test_modello(svm, Xn_test, y_test)


parametri = {'criterion': ['gini', 'entropy', 'log_loss']}
dtc = RandomizedSearchCV(DecisionTreeClassifier(), parametri, scoring='f1', n_iter=3)
dtc.fit(Xs_train, y_train)
dtc = dtc.best_estimator_
dtc.__annotations__ = 'DecisionTree clf                 [3]'
classificatori.append(dtc)
osserva_test_modello(dtc, Xs_test, y_test)

parametri = {'n_estimators': [25, 50, 75, 100, 150, 200, 250]}
rfc = RandomizedSearchCV(RandomForestClassifier(), parametri, scoring='f1', n_iter=7)
rfc.fit(Xs_train, y_train)
rfc = rfc.best_estimator_
rfc.__annotations__ = 'RandomForest clf                 [4]'
classificatori.append(rfc)
osserva_test_modello(rfc, Xs_test, y_test)



print('\n\n  <>\n')
X_norm, X_stan = scala_dati(col_continue, X)
f1_knn = []
f1_svm = []
f1_dtc = []
f1_rfc = []


k = 5
j = k
i = 0
while i < 3:
    print('\n--Valutazione con Stratified K-Fold Cross Validation:\t\t----\t\t[Run #' + str(i + 1) + ' | k=' + str(k) + ']--\n\n')
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=j)
    val_risultanti = cross_validate(knn, X_norm, y, cv=skf, scoring=('accuracy', 'precision', 'recall', 'f1'), return_train_score=True)
    osserva_metriche_classificatore(val_risultanti, knn)
    f1_knn.append(val_risultanti['test_f1'].mean())

    val_risultanti = cross_validate(svm, X_norm, y, cv=skf, scoring=('accuracy', 'precision', 'recall', 'f1'), return_train_score=True)
    osserva_metriche_classificatore(val_risultanti, svm)
    f1_svm.append(val_risultanti['test_f1'].mean())

    val_risultanti = cross_validate(dtc, X_stan, y, cv=skf, scoring=('accuracy', 'precision', 'recall', 'f1'), return_train_score=True)
    osserva_metriche_classificatore(val_risultanti, dtc)
    f1_dtc.append(val_risultanti['test_f1'].mean())

    val_risultanti = cross_validate(rfc, X_stan, y, cv=skf, scoring=('accuracy', 'precision', 'recall', 'f1'), return_train_score=True)
    osserva_metriche_classificatore(val_risultanti, rfc)
    f1_rfc.append(val_risultanti['test_f1'].mean())

    print('------------------------------------------------------------------------------------------------------\n\n')
    k += 5
    i += 1


print('\n\n - I risultati finali della valutazione sono i seguenti:\n\n')
print('--\tF1 Media delle CV eseguite\t--\n')
print(knn.__annotations__.upper() + ' :\t' + str(sum(f1_knn)/len(f1_knn)) + '\n')
print(svm.__annotations__.upper() + ' :\t' + str(sum(f1_svm)/len(f1_svm)) + '\n')
print(dtc.__annotations__.upper() + ' :\t' + str(sum(f1_dtc)/len(f1_dtc)) + '\n')
print(rfc.__annotations__.upper() + ' :\t' + str(sum(f1_rfc)/len(f1_rfc)) + '\n')
print('\n------------------------------------------------------------------------------------------------------\n\n\n')



# Rete Bayesiana:

df_RBayes = pd.DataFrame(np.array(dataset.copy(), dtype=int), columns=dataset.columns)


k2 = K2Score(df_RBayes)
hc_k2 = HillClimbSearch(df_RBayes)
modello_k2 = hc_k2.estimate(scoring_method=k2)

rete_bayesiana = BayesianNetwork(modello_k2.edges())
rete_bayesiana.fit(df_RBayes)

inferenza = VariableElimination(rete_bayesiana)
prob_notckd = inferenza.query(variables=['class'],
                              evidence={'age': 50, 'blood_pressure': 70, 'albumin': 0, 'sugar': 0, 'red_blood_cells': 1,
                                        'pus_cell': 1, 'pus_cell_clumps': 0, 'bacteria': 1, 'white_blood_cell_count': 7000,
                                        'blood_glucose_random': 130, 'blood_urea': 60, 'serum_creatinine': 2, 'sodium': 147,
                                        'haemoglobin': 13, 'packed_cell_volume': 45, 'red_blood_cell_count': 5, 'hypertension': 0,
                                        'diabetes_mellitus': 0, 'coronary_artery_disease': 0, 'appetite': 0, 'pedal_edema': 0, 'anemia': 0})

print('\nProbabilità per un individuo di non avere un disturbo cronico ai reni: ')
print(prob_notckd, '\n')


prob_ckd = inferenza.query(variables=['class'],
                           evidence={'age': 55, 'blood_pressure': 70, 'albumin': 0, 'sugar': 0, 'red_blood_cells': 1,
                                     'pus_cell': 1, 'pus_cell_clumps': 0, 'bacteria': 1, 'white_blood_cell_count': 7000,
                                     'blood_glucose_random': 130, 'blood_urea': 60, 'serum_creatinine': 1, 'sodium': 135,
                                     'haemoglobin': 13, 'packed_cell_volume': 45, 'red_blood_cell_count': 5, 'hypertension': 0,
                                     'diabetes_mellitus': 0, 'coronary_artery_disease': 0, 'appetite': 0, 'pedal_edema': 0, 'anemia': 0})

print('\nProbabilità per un individuo di avere un disturbo cronico ai reni: ')
print(prob_ckd, '\n\n')