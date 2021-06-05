import abc
import base64
from io import BytesIO

import pandas as pd
from matplotlib.figure import Figure


class AnalysisResult(abc.ABC):
    """ Encapsulates result of analysis in a way that is easy to render
    """

    @abc.abstractmethod
    def render_html(self) -> str:
        """ Renders result to an HTML
        :return: Ready to-be-rendered HTML containing analysis result. It is a safe string
        """


class DataFrameAnalysisResult(AnalysisResult):
    def __init__(self, dataframe: pd.DataFrame):
        """
        :param dataframe: DataFrame to be displayed as analysis result
        """
        self._dataframe = dataframe

    def render_html(self) -> str:
        return self._dataframe.to_html(table_id='stats')


class FigureAnalysisResult(AnalysisResult):
    def __init__(self, figure: Figure):
        """
        :param figure: Matplotlib's figure to be displayed as analysis result
        """
        self._figure = figure

    def render_html(self) -> str:
        buffer = BytesIO()
        self._figure.savefig(buffer, format="png")
        data = base64.b64encode(buffer.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'>"
