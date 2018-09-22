"""
Pairwise causal models base class
Author: Diviyan Kalainathan
Date : 7/06/2017
"""
import networkx as nx
from sklearn.preprocessing import scale
from pandas import DataFrame, Series
from ...utils.Settings import SETTINGS


class PairwiseModel(object):
    """Base class for all graph causal inference models.

    Usage for undirected graphs and raw data. All causal discovery
    models out of observational data base themselves on this class. Its main
    feature is the predict function that executes a function according to the
    given arguments.
    """

    def __init__(self):
        """Init."""
        super(PairwiseModel, self).__init__()

    def predict(self, x, *args, **kwargs):
        """Generic predict method.

        Chooses to execute according to the type of the arguments given.

        1. If ``args[0]`` is a ``networkx.Graph``, then ``self.orient_graph`` is executed.
        2. If ``x`` is a ``pandas.DataFrame``, then ``self.predict_dataset`` is executed.
        3. If ``x`` is a ``pandas.Series``, then ``self.predict_proba`` is executed.

        Args:
            df_data (pandas.DataFrame): DataFrame containing the observational data.
            args[0] (networkx.DiGraph or networkx.Graph or None): Prior knowledge on the causal graph.

        .. warning::
           Requirement : Name of the nodes in the graph must correspond to the
           name of the variables in df_data

        """
        if len(args) > 0:
            if type(args[0]) == nx.Graph or type(args[0]) == nx.DiGraph:
                return self.orient_graph(x, *args, **kwargs)
            else:
                return self.predict_proba(x, *args, **kwargs)
        elif type(x) == DataFrame:
            return self.predict_dataset(x, *args, **kwargs)
        elif type(x) == Series:
            return self.predict_proba(x.iloc[0], x.iloc[1], *args, **kwargs)

    def predict_proba(self, a, b, idx=0, **kwargs):
        """Predict the causal direction of a pair of variables.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        raise NotImplementedError

    def predict_dataset(self, x, **kwargs):
        """Predict the causal direction of all pairs in the dataset.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        printout = kwargs.get("printout", None)
        pred = []
        res = []
        x.columns = ["A", "B"]
        for idx, row in x.iterrows():
            a = scale(row['A'].reshape((len(row['A']), 1)))
            b = scale(row['B'].reshape((len(row['B']), 1)))

            pred.append(self.predict_proba(a, b, idx))

            if printout is not None:
                res.append([row['SampleID'], pred[-1]])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)
        return pred

    def orient_graph(self, df_data, graph, printout=None, nb_runs=6, **kwargs):
        """Orient an undirected graph.

        .. note::
           Not implemented: will be implemented by the model classes.
        """
        if type(graph) == nx.DiGraph:
            edges = [a for a in list(graph.edges()) if (a[1], a[0]) in list(graph.edges())]
            oriented_edges = [a for a in list(graph.edges()) if (a[1], a[0]) not in list(graph.edges())]
            for a in edges:
                if (a[1], a[0]) in list(graph.edges()):
                    edges.remove(a)
            output = nx.DiGraph()
            for i in oriented_edges:
                output.add_edge(*i)

        elif type(graph) == nx.Graph:
            edges = list(graph.edges())
            output = nx.DiGraph()

        else:
            raise TypeError("Data type not understood.")

        res = []

        for idx, (a, b) in enumerate(edges):
            weight = self.predict_proba(
                df_data[a].values.reshape((-1, 1)), df_data[b].values.reshape((-1, 1)), idx=idx,
                nb_runs=nb_runs, **kwargs)
            if weight > 0:  # a causes b
                output.add_edge(a, b, weight=weight)
            else:
                output.add_edge(b, a, weight=abs(weight))
            if printout is not None:
                res.append([str(a) + '-' + str(b), weight])
                DataFrame(res, columns=['SampleID', 'Predictions']).to_csv(
                    printout, index=False)

        for node in list(df_data.columns.values):
            if node not in output.nodes():
                output.add_node(node)

        return output
