from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import config

class RAGChain:
    def __init__(self, retriever):
        """
        Initializes the RAG Chain with a retriever and a Gemini LLM.
        """
        # 1. Initialize your LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3
        )

        # 2. Define the modern ChatPromptTemplate
        system_prompt = (
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n"
            "Context:\n{context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # 3. Build the document chain and the final retrieval chain
        self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(retriever, self.question_answer_chain)

    def ask(self, question: str) -> dict:
        """
        Passes the user question to the chain and returns a dictionary
        containing the answer and the formatted sources.
        """
        # Invoke the chain
        response = self.rag_chain.invoke({"input": question})
        
        # Extract and format the sources from the retrieved context
        formatted_sources = []
        for doc in response.get("context", []):
            formatted_sources.append({
                "file": doc.metadata.get("source", "Unknown Document"),
                "page": doc.metadata.get("page", "N/A"),
                # Grab the first 150 characters of the document chunk as an excerpt
                "excerpt": doc.page_content[:150].replace('\n', ' ') + "..." 
            })

        # Return the exact dictionary format that main.py expects
        return {
            "answer": response["answer"],
            "sources": formatted_sources
        }