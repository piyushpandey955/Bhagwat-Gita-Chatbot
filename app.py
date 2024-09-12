from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as pc
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings=download_hugging_face_embeddings()

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
p = pc(api_key=PINECONE_API_KEY)
index1 = p.Index("medical-bot")
index_name = "medical-bot"

#loading the index
docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="/Users/piyushpandey955/Desktop/AIML/gen ai/medical chatbot project/Medical-Chatbot/model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.75}
                )


# Use the Pinecone vectorstore from Langchain
vectorstore = Pinecone(
    index=pinecone.Index(index_name, host='https://medical-bot-9d82pte.svc.aped-4627-b74a.pinecone.io'),
    embedding_function=embeddings.embed_query,
    text_key="text"
    )

# Set up the retriever with Pinecone and Langchain
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    query_embedding = embeddings.embed_query(input)
    print(input)
    # Query Pinecone using the keyword arguments format
    result = index1.query(vector=query_embedding, top_k=2, include_metadata=True)
    

    # Check if results are found
    if result and 'matches' in result:
        # Loop through the top results
        for match in result['matches']:
            # Extract metadata if available (assuming text is stored in metadata)
            if 'metadata' in match and 'text' in match['metadata']:
                # Extract and print the relevant text
                answer = match['metadata']['text']
                print("Match:", match['metadata']['text'])
            else:
                print("No metadata or text field found in this match.")
    else:
        print("No results found.")
    return str(answer)


if __name__ =='__main__':
    app.run(debug=True)


