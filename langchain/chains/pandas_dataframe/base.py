"""Chain for interacting with Pandas DataFrame."""
from __future__ import annotations

from typing import Any, Dict, List

from matplotlib.axes import Axes
from pandas import DataFrame
from pydantic import BaseModel, Extra, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.pandas_dataframe.prompt import PLOT_PROMPT, PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


def _evaluate_code(code: str, df: DataFrame = None) -> Any:
    # Import common libraries
    from datetime import datetime, timedelta  # noqa: F401

    import matplotlib.pyplot as plt  # noqa: F401
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import plotly.express as px  # noqa: F401

    try:
        try:
            return eval(code)
        except SyntaxError:
            return exec(code)
    except (NameError, KeyError) as e:
        return e


class PandasDataFrameChain(Chain, BaseModel):
    """Chain for interacting with a Pandas DataFrame.

    Example:
        .. code-block:: python

            df = DataFrame(...)
            df_chain = PandasDataFrameChain(llm=OpenAI(), dataframe=df)
    """

    llm: BaseLLM
    """LLM wrapper to use."""
    dataframe: DataFrame = Field(exclude=True)
    """Pandas DataFrame to connect to."""
    prompt: BasePromptTemplate = PROMPT
    """Prompt to use to translate natural language to Pandas."""
    plot_prompt: BasePromptTemplate = PLOT_PROMPT
    """Prompt to use to translate Pandas plots to Plotly."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    code_key: str = "code"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.code_key, self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        self.callback_manager.on_text("Question: ", verbose=self.verbose)
        self.callback_manager.on_text(
            inputs[self.input_key], color="blue", verbose=self.verbose
        )
        llm_inputs = {
            "input": inputs[self.input_key],
            "column_names": self.dataframe.columns.to_list(),
            "stop": ["\nCode:"],
        }
        code = llm_chain.predict(**llm_inputs).strip()
        if ".plot" in code:
            plot_chain = LLMChain(llm=self.llm, prompt=self.plot_prompt)
            llm_inputs = {
                "input": code,
                "stop": ["\nPlotly:"],
            }
            code = plot_chain.predict(**llm_inputs).strip()
        self.callback_manager.on_text("\nCode: ", verbose=self.verbose)
        self.callback_manager.on_text(code, color="yellow", verbose=self.verbose)
        result = _evaluate_code(code, df=self.dataframe)
        if isinstance(result, Exception):
            self.callback_manager.on_text("\nResult: ", verbose=self.verbose)
            self.callback_manager.on_text(
                result.__class__.__name__, color="pink", verbose=self.verbose
            )
        else:
            self.callback_manager.on_text("\nResult: ", verbose=self.verbose)
            self.callback_manager.on_text(
                str(result), color="green", verbose=self.verbose
            )
        return {self.code_key: code, self.output_key: result}

    @property
    def _chain_type(self) -> str:
        return "pandas_dataframe_chain"
