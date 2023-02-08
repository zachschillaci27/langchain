"""Chain for interacting with Pandas DataFrame."""
from __future__ import annotations

from typing import Any, Dict, List

from pandas import DataFrame
from pydantic import BaseModel, Extra, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.pandas_dataframe.prompt import PANDAS_PROMPT, PLOT_PROMPT
from langchain.llms.base import BaseLLM
from langchain.prompts.base import BasePromptTemplate


def _evaluate_code(code: str, df: DataFrame = None) -> Any:
    # Import common libraries that are likely to be used
    # in the machine-generated code. These should follow
    # the most frequently used aliases.
    from datetime import datetime, timedelta  # noqa: F401

    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401

    if _code_contains_plotly_plot(code):
        try:
            import plotly.express as px  # noqa: F401
        except ImportError:
            raise ImportError(
                "Plotly is not installed. Please install it with `pip install plotly`."
            )

    try:
        try:
            return eval(code)
        except SyntaxError:
            return exec(code)
    except (NameError, KeyError) as e:
        return e


def _code_contains_pandas_plot(code: str) -> bool:
    return ".plot" in code


def _code_contains_plotly_plot(code: str) -> bool:
    return "px." in code


class PandasDataFrameChain(Chain, BaseModel):
    """Chain for interacting with a Pandas DataFrame.

    Example:
        .. code-block:: python

            df = DataFrame(...)
            df_chain = PandasDataFrameChain(llm=OpenAI(), dataframe=df)
    """

    dataframe: DataFrame = Field(exclude=True)
    """Pandas DataFrame to use."""
    pandas_generator: LLMChain
    """Pandas chain to generate executable Pandas code."""
    plot_generator: LLMChain
    """Plot chain to generate interactive Plotly figures."""
    code_key: str = "code"
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

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

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        dataframe: DataFrame,
        pandas_prompt: BasePromptTemplate = PANDAS_PROMPT,
        plot_prompt: BasePromptTemplate = PLOT_PROMPT,
        verbose: bool = False,
    ) -> PandasDataFrameChain:
        return cls(
            dataframe=dataframe,
            pandas_generator=LLMChain(llm=llm, prompt=pandas_prompt),
            plot_generator=LLMChain(llm=llm, prompt=plot_prompt),
            verbose=verbose,
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        self.callback_manager.on_text("Question: ", verbose=self.verbose)
        self.callback_manager.on_text(
            inputs[self.input_key], color="blue", verbose=self.verbose
        )
        llm_inputs = {
            "input": inputs[self.input_key],
            "column_names_and_types": str(self.dataframe.dtypes),
            "stop": ["\nCode:"],
        }
        # Store all LLM-generated code responses in a list
        code = [self.pandas_generator.predict_and_parse(**llm_inputs)]
        # Check if this output is a Pandas DataFrame plot, if so
        # run the Plotly chain to translate it.
        if _code_contains_pandas_plot(code[-1]):
            llm_inputs = {
                "input": code[-1],
                "stop": ["\nPlotly:"],
            }
            code.append(self.plot_generator.predict_and_parse(**llm_inputs))
        self.callback_manager.on_text("\nCode: ", verbose=self.verbose)
        self.callback_manager.on_text(code[-1], color="yellow", verbose=self.verbose)
        # Evaluate the code and a handle common exceptions automatically
        result = _evaluate_code(code[-1], df=self.dataframe)
        if isinstance(result, Exception):
            self.callback_manager.on_text("\nResult: ", verbose=self.verbose)
            self.callback_manager.on_text(
                f"""{result.__class__.__name__}: {result}.""",
                color="pink",
                verbose=self.verbose,
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
