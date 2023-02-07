# flake8: noqa
from langchain.prompts.base import CodeOutputParser
from langchain.prompts.prompt import PromptTemplate

_PANDAS_TEMPLATE = """Given an input question, create syntactically correct Python code to run using the Pandas framework. 

Respond only in executable Python code using the following format:

Question: "Question here"
Code: "Python code to run"

The variable name of the DataFrame is `df`. The column names and data types of the DataFrame are as follows:
{column_names_and_types}

Be sure to access the columns names exactly as they are named and use operations appropriate to the data type.

If generating a plot, do not make any unnecessary aesthetic contributions, such as coloring or figure size modifications.

Do not use any in place operations and be sure to return the final value directly.

Question: {input}
Code: """

PANDAS_PROMPT = PromptTemplate(
    input_variables=["input", "column_names_and_types"],
    template=_PANDAS_TEMPLATE,
    output_parser=CodeOutputParser(),
)

_PLOT_TEMPLATE = """Convert the following Pandas DataFrame plot to Plotly Express.

Here are some examples:

EXAMPLE 1:
===================================================================================
Pandas: df.groupby("x")["y"].mean().plot(kind="bar")
Plotly: px.bar(df.groupby("x").mean().reset_index(), x="x", y="y", barmode="group")
===================================================================================

EXAMPLE 2:
===================================================================================
Pandas: df.groupby("x")["y"].sum().plot(kind="scatter", color="z")
Plotly: px.scatter(df.groupby("x").sum().reset_index(), x="x", y="y", color="z")
===================================================================================

Pandas: {input}
Plotly: """

PLOT_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=_PLOT_TEMPLATE,
    output_parser=CodeOutputParser(),
)
