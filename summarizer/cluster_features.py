from typing import List

import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class ClusterFeatures(object):
    """
    Basic handling of clustering features.
    """

    def __init__(
        self,
        features: ndarray,
        base_features: ndarray,
        algorithm: str = 'kmeans',
        pca_k: int = None,
        random_state: int = 12345
    ):
        """
        :param features: the embedding matrix created by bert parent
        :param base_features: the embedding matrix created by bert parent for the example sentences
        :param algorithm: Which clustering algorithm to use
        :param pca_k: If you want the features to be ran through pca, this is the components number
        :param random_state: Random state
        """

        if pca_k:
            self.features = PCA(n_components=pca_k).fit_transform(features)
            self.base_features = PCA(n_components=pca_k).fit_transform(base_features)
        else:
            self.features = features
            self.base_features = base_features

        self.algorithm = algorithm
        self.pca_k = pca_k
        self.random_state = random_state

    def __get_model(self, k: int = 2):
        """
        Retrieve clustering model from the example predefined sentences

        :param k: amount of clusters
        :return: Clustering model

        """

        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)
        return KMeans(n_clusters=k, random_state=self.random_state)

    def __get_centroids(self, model):
        """
        Retrieve centroids of model
        :param model: Clustering model
        :return: Centroids
        """

        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_

    def __find_closest_args(self, centroids: np.ndarray,k: int = 2):
        """
        Find the closest arguments to centroids of the example sentences 

        :param centroids: Centroids to find closest
        :param k: amount of sentences per cluster to look for
        :return: Closest arguments
        """

        sentences_per_cluster = [int(k/len(centroids))]*len(centroids)
        idx = 0
        if k - sum(sentences_per_cluster) > len(sentences_per_cluster):
            sentences_per_cluster = [a+1 for a in sentences_per_cluster]
        while k - sum(sentences_per_cluster) > 0:
            sentences_per_cluster[idx] += 1
            idx += 1

        print(sentences_per_cluster)
        
        args = {}
        used_idx = []

        for j, centroid in enumerate(centroids):

            scored_ids = []

            for i, feature in enumerate(self.features):
                value = np.linalg.norm(feature - centroid)
                if i in used_idx:
                    value = 100000000
                    pass
                scored_ids.append(value)
            
            args[j] = np.argsort(scored_ids)[:sentences_per_cluster[j]]
            print(f"Cluster {j}:")
            for i in range(len(args[j])):
                print(f"Sentence {args[j][i] + 1}:\t{np.sort(scored_ids)[i]}")
            used_idx.extend(args[j])
            # print(used_idx)

        # print(args)
        return args

    def cluster(self, nr_sentences: int = 4, clusters: int = 2) -> List[int]:
        """
        Clusters sentences
        :param nr_sentences: Sentences to output at summary
        :param clusters: Clusters to use from base examples
        :return: Sentences index that qualify for summary
        """
        # new_ratio = ratio/clusters
        # k = 1 if new_ratio * len(self.features) < 1 else int(len(self.features) * new_ratio)
        # print(f"Defined ratio:\t{ratio} ({k * clusters} sentence{'s' if k > 1 else ''})")

        model = self.__get_model(clusters).fit(self.base_features)
        centroids = self.__get_centroids(model)
        cluster_args = self.__find_closest_args(centroids,nr_sentences)
        # sorted_values = sorted(cluster_args.values())
        return cluster_args

    def __call__(self, nr_sentences: int = 4, clusters: int = 2) -> List[int]:
        return self.cluster(nr_sentences,clusters)
