# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_TEMPLATE = """Given an input question, create syntactically correct Python code to run using the Pandas framework.
Respond only in executable Python code using the following format:

Question: "Question here"
Code: "Python code to run"

The variable name of the DataFrame is `df`. The column names are:
{column_names}

Do not use any in place operations and be sure to return the final value directly.

Question: {input}
Code: """

PROMPT = PromptTemplate(
    input_variables=["input", "column_names"],
    template=_DEFAULT_TEMPLATE,
)

_PLOT_TEMPLATE = """Convert the following Pandas DataFrame plot to Plotly Express. Consider the following examples for reference:    

# Examples:
Pandas: df.groupby('x')['y'].mean(numeric_only=True).plot(kind="bar")
Plotly: px.bar(df.groupby("x").mean(numeric_only=True).reset_index(), x="x", y="y", barmode="group")

Pandas: df.groupby('x')['y'].sum(numeric_only=True).plot(kind="scatter", color="z")
Plotly: px.scatter(df.groupby("x").sum(numeric_only=True).reset_index(), x="x", y="y", color="z")
#

Pandas: {input}
Plotly: """

PLOT_PROMPT = PromptTemplate(
    input_variables=["input"],
    template=_PLOT_TEMPLATE,
)
