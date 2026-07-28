import os

file_path = "backend/tests/test_workflow.py"
with open(file_path, "r") as f:
    content = f.read()

content = content.replace(
    'patch("backend.app.agent.workflow.router", new=mock_router)',
    'patch("backend.app.agent.workflow.get_router", return_value=mock_router)'
)
content = content.replace(
    'patch("backend.app.agent.workflow.retriever", new=mock_retriever)',
    'patch("backend.app.agent.workflow.get_retriever", return_value=mock_retriever)'
)
content = content.replace(
    'patch("backend.app.agent.workflow.sql_engine", new=mock_sql_engine)',
    'patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_sql_engine)'
)
content = content.replace(
    'patch("backend.app.agent.workflow.sql_engine", new=mock_engine)',
    'patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_engine)'
)
content = content.replace(
    'patch("backend.app.agent.workflow.llm", new=mock_llm)',
    'patch("backend.app.agent.workflow.get_llm", return_value=mock_llm)'
)
content = content.replace(
    'patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_sql_engine), \\',
    'patch("backend.app.agent.workflow.get_sql_engine", return_value=mock_engine), \\'
)

with open(file_path, "w") as f:
    f.write(content)

print("Tests patched successfully!")
