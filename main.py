# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from fastapi.middleware.cors import CORSMiddleware

import os

# After loading environment variables
api_key = os.environ['OPENAI_API_KEY']

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins for testing; update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index at startup
embeddings = OpenAIEmbeddings()
faiss_index = FAISS.load_local("faiss_index",
                               embeddings,
                               allow_dangerous_deserialization=True)
retriever = faiss_index.as_retriever()
llm = OpenAI()
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)


class QueryRequest(BaseModel):
    query: str


class Source(BaseModel):
    url: str
    paragraph: int
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = qa_chain({"query": request.query})
        answer = result["result"]
        source_documents = result["source_documents"]

        # Prepare sources
        sources = []
        for doc in source_documents:
            url = doc.metadata.get('url', 'N/A')
            paragraph = doc.metadata.get('paragraph', 'N/A')
            snippet = doc.page_content[:200]
            sources.append({
                "url": url,
                "paragraph": paragraph,
                "snippet": snippet
            })

        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app)