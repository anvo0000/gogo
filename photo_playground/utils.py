from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain

from PIL import Image
import numpy as np
import time
import open_clip
from io import BytesIO
import base64


### Photo Zone
def get_photo_arr(photo_name):
    img = Image.open(photo_name)
    pic_arr = np.asarray(img)
    return pic_arr

def resize_and_compress_image(np_array, max_size=(100, 100), quality=85):
    """
    Resize and compress a numpy array image.
    
    Parameters:
    np_array (numpy.ndarray): The numpy array to resize and compress.
    max_size (tuple): The maximum size for the image.
    quality (int): The quality of the compression (1-100).
    
    Returns:
    str: The base64 encoded string of the resized and compressed image.
    """
    pil_img = Image.fromarray(np_array)
    pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG", quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# iterate over files in 
# that user uploaded Photos, one by one
def create_docs(user_photo_list, unique_id):
    docs=[]
    for filename in user_photo_list:
        
        chunks = get_photo_arr(filename)
        img_str = resize_and_compress_image(chunks)
        # img_str = img_str[0:100]
        #Adding items to our list - Adding data & its metadata
        docs.append(Document(
            page_content=img_str,
            metadata={"name": filename.name,
                      "id":filename.file_id,
                      "type=":filename.type,
                      "size":filename.size,
                      "unique_id":unique_id},
        ))
        
    return docs


#Create embeddings instance
def create_embeddings_load_data():
    #embeddings = OpenAIEmbeddings()
    
    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")
    
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', 
    #                                                          pretrained='laion2b_s34b_b79k')
    # embeddings = model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    # embeddings = preprocess(Image.open("./images/hippo.png")).unsqueeze(0)

    return embeddings


#Function to push data to Vector Store - Pinecone here
def push_to_pinecone(docs,pinecone_index_name,embeddings):
  PineconeVectorStore.from_documents(
     documents=docs,
     embedding=embeddings,
     index_name=pinecone_index_name
  )
    

#Function to pull infrmation from Vector Store - Pinecone here
def pull_from_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings):
    # For some of the regions allocated in pinecone which are on free tier, the data takes upto 10secs for it to available for filtering
    #so I have introduced 20secs here, if its working for you without this delay, you can remove it :)
    #https://docs.pinecone.io/docs/starter-environment
    print("20secs delay...")
    time.sleep(20)
    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index



#Function to help us get relavant documents from vector store - based on user input
def similar_docs(query,k,pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,unique_id):

    pinecone.init(
    api_key=pinecone_apikey,
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = pull_from_pinecone(pinecone_apikey,pinecone_environment,index_name,embeddings)
    similar_docs = index.similarity_search_with_score(query, int(k),{"unique_id":unique_id})
    #print(similar_docs)
    return similar_docs


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    #llm = HuggingFaceHub(repo_id="bigscience/bloom", model_kwargs={"temperature":1e-10})
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])

    return summary



def main():
  rs = create_docs()
  print(rs)


if __name__ == "__main__":
  main()   