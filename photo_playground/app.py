import streamlit as st
import uuid
from utils import *
from dotenv import load_dotenv
import os

load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
pinecode_environment=os.environ.get("PINECONE_ENVIRONMENT")

#Session variable to ensure we don't duplicate
if "unique_id" not in st.session_state:
  st.session_state[""] = "" #set it's empty

def main():
  st.set_page_config(page_title="NourishAI Assistant")
  st.title("NourishAI Assistant")
  st.subheader("I can help you in regconizing food portion and scoring it")

  #upload the food photos
  photos = st.file_uploader("Upload photos here, only PNG, JPEG allowed", 
                           type=["png","jpeg"],
                           accept_multiple_files=True)
  submit = st.button("Analyze Photos")

  if submit:
    with st.spinner("NourishAI is processing your photos...") as spinner:

      ### Creating a uniqueID for this session, 
      # so that we can use to query and get only the user uploaded files from PINECONE
      st.session_state["unique_id"] = uuid.uuid4().hex
      st.write("-->", st.session_state["unique_id"])
      
      ### Create a documents list of all the users uploaded photos
      docs = create_docs(photos, st.session_state["unique_id"])
      st.write(len(docs))
      st.write(docs)

      ### Create embedding instances
      embeddings = create_embeddings_load_data()

      ### Send to Pinecone
      push_to_pinecone(docs=docs,
                       pinecone_index_name=pinecone_index_name,
                       embeddings=embeddings
                       )



    st.success("Your meal looks so yummy!!!")

  
  



if __name__ == "__main__":
  main()
