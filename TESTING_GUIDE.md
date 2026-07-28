# 🧪 Testing Guide: Dual-Mode Agentic RAG Chatbot

Welcome to the testing guide for the **Dual-Mode Agentic RAG Chatbot**. This document outlines how to properly validate the capabilities of the system's dynamic routing, retrieval, and SQL generation engines.

---

## 🎯 Understanding the Routing Engine

This application features an intelligent **LangGraph-based Router** that classifies your prompt and dynamically chooses the correct execution path:

1. **RAG Route (Retrieval-Augmented Generation)**: Triggered when asking about policies, guidelines, or unstructured text. The agent searches embedded PDF documents using ChromaDB.
2. **SQL Route (Text-to-SQL)**: Triggered when asking for statistics, counts, or structured database records. The agent writes and executes raw SQLite queries.
3. **Hybrid Route (RAG + SQL)**: Triggered for complex queries requiring BOTH unstructured policy knowledge and structured database records.

---

## 📝 Sample Test Questions

Below are categorized test prompts you can paste into the chat interface to validate the system. 

### 1. RAG Queries (Document Retrieval)
These questions test the system's ability to search through PDF documents (like HR manuals or Return Policies) to find text-based answers.
* *"Summarize the company's leave policy."*
* *"What is the warranty period for laptops?"*
* *"How many days do I have to return a defective item?"*
* *"What is the policy regarding sick leave?"*

**Expected Behavior:** The agent should display the `RAG` badge, stream a text response based on the PDF files in the `Dataset/` folder, and cite its sources.

### 2. SQL Queries (Database Queries)
These questions test the Text-to-SQL engine, converting your English question into a SQL query to fetch data from the `orders.db` database.
* *"How many pending orders do we currently have?"*
* *"What is the total revenue for the last month?"*
* *"Which customer has the highest number of completed orders?"*
* *"List the top 5 most expensive products we sell."*

**Expected Behavior:** The agent should display the `SQL` badge. Underneath the response, you should be able to expand the **"View Generated SQL"** toggle to inspect the exact SQLite query it ran against the database.

### 3. Hybrid Queries (RAG + SQL)
These questions are the most advanced. The Agent will realize it needs BOTH policy information from the PDFs and current order data from the SQL database to answer correctly.
* *"Which of our customers with pending orders have purchased items that are still under the standard warranty period?"*
* *"Can you list the recent orders that are eligible for a return based on our returns policy?"*

**Expected Behavior:** The agent should display the `Hybrid` badge. It will simultaneously query the SQLite database and search ChromaDB, merging both contexts before streaming the final response.

---

## 🚦 Troubleshooting & Local Execution

If you are running into Docker space constraints (`ENOSPC`) or binding issues (`Errno 10048`), you can run the application natively on Windows using Python and Node.js.

### 1. Start the FastAPI Backend
Ensure your virtual environment is activated and dependencies are installed.
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*Note: If you are running locally without Redis, ensure `ENABLE_MEMORY=False` in your `.env` file to prevent connection timeouts.*

### 2. Start the Next.js Frontend
Open a **new terminal window**, navigate to the frontend directory, and start the development server using Webpack.
```powershell
cd frontend
npm run dev --webpack
```

You can then access the chat interface at **http://localhost:3000**.
