import umap
import numpy as np
import matplotlib.pyplot as plt

distances = np.load('distances.npy')
y = np.load('y.npy')
subjects = np.load('subject_ids.npy')
unique_subjects = sorted(np.unique(subjects))
SUB_IND = -3
# mask = (subjects == unique_subjects[-3]) | (subjects == unique_subjects[-2]) | (subjects == unique_subjects[-1])

# distances = distances[np.ix_(mask, mask)]
# y = y[mask]
# from sklearn.manifold import MDS

# trans = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distances)
# plt.scatter(trans[:, 0], trans[:, 1], s= 5, c=subjects, cmap='Spectral')
# plt.title(f'{unique_subjects[SUB_IND]} Embedding of the training set by MDS', fontsize=24)
# plt.colorbar()
# plt.show()


trans = umap.UMAP(n_neighbors=1000, metric='precomputed', n_epochs=200).fit(distances)
plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s= 5, c=y, cmap='Spectral')
plt.title(f'{unique_subjects[SUB_IND]} Embedding of the training set by UMAP', fontsize=24)
plt.colorbar()
plt.show()