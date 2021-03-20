from pytorch_metric_learning.utils import accuracy_calculator
import numpy as np


class AccCalculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_1(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 1, self.avg_of_avgs)

    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 5, self.avg_of_avgs)

    def calculate_precision_at_10(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 10, self.avg_of_avgs)

    def calculate_knn_labels(self, knn_labels, query_labels, **kwargs):
        return knn_labels

    def retrieval_at_k(self, k, knn_labels, query_labels):
        curr_knn_labels = knn_labels[:, :k]
        accuracy_per_sample = np.apply_along_axis(any, axis=1, arr=(curr_knn_labels == query_labels[:, None]))
        return accuracy_calculator.maybe_get_avg_of_avgs(accuracy_per_sample, query_labels, self.avg_of_avgs)

    def calculate_retrieval_at_1(self, knn_labels, query_labels, **kwargs):
        return self.retrieval_at_k(1, knn_labels, query_labels)

    def calculate_retrieval_at_10(self, knn_labels, query_labels, **kwargs):
        return self.retrieval_at_k(10, knn_labels, query_labels)

    def calculate_retrieval_at_100(self, knn_labels, query_labels, **kwargs):
        return self.retrieval_at_k(100, knn_labels, query_labels)

    def calculate_duplicates(self, knn_distances, **kwargs):
        duplicates = knn_distances[:, :1].squeeze(1) == 0
        return sum(duplicates) / len(duplicates)

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_1", "precision_at_5", "precision_at_10", "retrieval_at_1",
                                         "retrieval_at_10", "retrieval_at_100", "knn_labels"]
