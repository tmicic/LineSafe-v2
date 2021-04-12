import torch
import warnings 

class Metrics:

    def __init__(self, comparison_metric='accuracy', metric_is_percentage=True) -> None:
            
            self.comparison_metric = None
            self.metric_is_percentage = metric_is_percentage    # if true, when printing to screen, will multiple metric by 100

            self.set_comparison_metric(comparison_metric)
            self.reset()

    def set_comparison_metric(self, metric_name):

        avail = [a for a in (dir(self) + list(self.__dict__.keys())) if not (a.startswith('__') or a.startswith('set') or a in ['update', 'reset', 'get_metric_value'])] 

        if metric_name in avail:
            self.comparison_metric = metric_name
        else:
            raise ValueError(f'Invalid metric. Please select a valid matric from {avail}.')
    
    def get_metric_value(self):
        return getattr(self, self.comparison_metric)()


    def negative_count(self):
        return self.count - self.positive_count

    def reset(self):
        self.count = torch.tensor(0.)
        self.positive_count = torch.tensor(0.)
        self.true_positives = torch.tensor(0.)
        self.true_negatives = torch.tensor(0.)
        self.false_positives = torch.tensor(0.)
        self.false_negatives = torch.tensor(0.)
        self.loss = None

    def __eq__(self, o: object) -> bool:
        return (self.get_metric_value() == o.get_metric_value()).item()

    def __ne__(self, o: object) -> bool:
        return (self.get_metric_value() != o.get_metric_value()).item()

    def __lt__(self, o: object) -> bool:
        return (self.get_metric_value() < o.get_metric_value()).item()

    def __gt__(self, o: object) -> bool:
        return (self.get_metric_value() > o.get_metric_value()).item()

    def __le__(self, o: object) -> bool:
        return (self.get_metric_value() <= o.get_metric_value()).item()

    def __ge__(self, o: object) -> bool:
        return (self.get_metric_value() >= o.get_metric_value()).item()

 

    def update(self, count, positive_count, true_positives, true_negatives, false_positives, false_negatives, loss=None):

        self.count += count
        self.positive_count += positive_count
        self.true_positives += true_positives
        self.true_negatives += true_negatives
        self.false_positives += false_positives
        self.false_negatives += false_negatives
        self.loss = loss

        if true_negatives+true_positives+false_negatives+false_positives != count:
            warnings.warn('TN, TP, FP, FN do not sum to match count during a metrics update! Metrics will not be accurate!', RuntimeWarning)
    
    def p(self):
        return self.positive_count

    def n(self):
        return self.negative_count()

    def sensitivity(self):
        return self.true_positives / self.p()

    def specificity(self):
        return self.true_negatives / self.n()

    def f1_score(self):
        return 2 * ((self.positive_predictive_value() * self. true_positive_rate()) / (self.positive_predictive_value() + self.true_positive_rate()))

    def  false_positive_rate(self):
        return self.false_positives / self.n()

    def false_negative_rate(self):
        return self.false_negatives / self.p()
    
    def accuracy(self):
        return (self.true_positives + self.true_negatives) / self.count
    
    def balananced_accuracy(self):
        return (self.true_positive_rate() + self.true_negative_rate()) / 2.
    
    def recall(self):
        return self.sensitivity()

    def hit_rate(self):
        return self.sensitivity()

    def true_positive_rate(self):
        return self.sensitivity()

    def selectivity(self):
        return self.specificity()
    
    def true_negative_rate(self):
        return self.specificity()

    def positive_predictive_value(self):
        return self.true_positives / (self.true_positives + self.false_positives)

    def precision(self):
        return self.positive_predictive_value()

    def negative_predictive_value(self):
        return self.true_negatives / (self.true_negatives + self.false_negatives)

    def false_negative_rate(self):
        return self.false_negatives / self.p()
    
    def miss_rate(self):
        return self.false_negative_rate()

    def false_positive_rate(self):
        return self.false_positives / self.n()

    def fall_out(self):
        return self.false_positive_rate()

    def false_discovery_rate(self):
        return 1. - self.positive_predictive_value()

    def false_omission_rate(self):
        return 1. - self.negative_predictive_value()

    def prevalence_threshold(self):
        return (torch.sqrt(self.true_positive_rate()*(-self.true_negative_rate()+1)) + self.true_negative_rate() - 1.) / (self.true_positive_rate() + self.true_negative_rate() - 1)

    def threat_score(self):
        return self.true_positives / (self.true_positives + self.false_negatives + self.false_positives)

    def critical_success_index(self):
        return self.threat_score()

    def matthews_correlation_coefficient(self):
        return ((self.true_positives * self.true_negatives) - (self.false_positives * self.false_negatives)) / torch.sqrt((self.true_positives+self.false_positives)*(self.true_positives+self.false_negatives)*(self.true_negatives+self.false_positives)*(self.true_negatives+self.false_negatives))

    def fowlkes_mallows_index(self):
        return torch.sqrt((self.positive_predictive_value() * self.true_positive_rate()))

    def informedness(self):
        return self.true_positive_rate() + self.true_positive_rate() - 1.

    def bookmaker_informedness(self):
        return self.informedness()

    def markedness(self):
        return self.positive_predictive_value() + self.negative_predictive_value() - 1.

    def delta_p(self):
        return self.markedness()

    def __str__(self) -> str:
        return f'{self.comparison_metric}: {self.get_metric_value().item() * (100 if self.metric_is_percentage else 1.):.4f}{("%" if self.metric_is_percentage else "")}'

    def __repr__(self) -> str:
        return self.__str__()


if __name__ == '__main__':
    pass





