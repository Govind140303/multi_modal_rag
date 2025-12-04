# multi_modal_rag
Local RAG — Document QA Demo

This project lets you ask questions about a long PDF and instantly get the most relevant answers.
It works by breaking the document into chunks, generating embeddings, building a FAISS vector index, and using a Streamlit app to search and display the best-matching passages.


HOW TO USE IT

✅ 1. Install Anaconda (recommended)

Download and install from:
https://www.anaconda.com/download

This avoids all Windows errors with FAISS & pyarrow.

✅ 2. Open Anaconda Prompt

Search in Start Menu → "Anaconda Prompt"

✅ 3. Create Environment

Inside Anaconda Prompt, run:

conda create -n mmr python=3.10 -y
conda activate mmr


This makes a clean environment for your project.

✅ 4. Install Required Libraries

FAISS, pyarrow, streamlit, and sentence-transformers must be installed using conda:

conda install -c conda-forge faiss-cpu pyarrow streamlit sentence-transformers -y


Then install remaining pip dependencies:

pip install -r requirements.txt --no-deps
pip install blinker

✅ 5. Go to Your Project Folder

Run:

cd C:\Users\govin\multi_modal_rag


(Or wherever the project is located.)

✅ 6. Make Sure FAISS Index Exists

Look inside:

embeddings_out/


It MUST contain:

faiss.index

meta.json

If these are missing → run the pipeline:

python ingest_simple.py
python embed_simple.py
python build_and_query.py

⭐ 7. Run the Application

Use this command:

python -m streamlit run app_demo.py


IMPORTANT
Always run Streamlit using python -m so it uses the correct environment.

⭐ 8. Open the App in Browser

After the command runs, the terminal will show:

Local URL: http://localhost:8501


Click that link or open it in your browser.

⭐ 9. How to Use the App
Step 1

Click "Load index & model"
This loads:

the FAISS index

metadata

the embedding model

Step 2

Enter a question in the text box. Example:

"What is Qatar's GDP?"

"Government revenue projections?"

"Oil production trends?"

Step 3

Click Search

⭐ 10. What the App Returns

You will see:

The top-K most relevant chunks

Page number

Snippet of the text

Relevance percentage (0–100%)

A downloadable CSV of results

Example:

1. Page 24 — 92.5% relevance
Qatar's GDP increased in ...

🎉 THAT’S IT
