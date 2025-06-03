import os
import re

from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.chains import ConversationalRetrievalChain

from doc_descriptions import DOCUMENT_DESCRIPTIONS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing in the .env file.")
os.environ["OPENAI_API_KEY"] = api_key


llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature = 0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# vector_store = Chroma(
#     collection_name="university_data",
#     embedding_function=embeddings,
#     persist_directory="./university_info_db",
# )

DATA_FOLDER = "data"


def classify_document_by_question(question):
    """
    Returns a list of document files based on keywords extracted from the question
    """
    question = question.lower()
    relevant_files = []
    for filename, metadata in DOCUMENT_DESCRIPTIONS.items():
        keywords = metadata.get("keywords", [])
        if any(keyword.lower() in question.lower() for keyword in keywords):
            relevant_files.append(filename)
    if not relevant_files:
        relevant_files = list(DOCUMENT_DESCRIPTIONS.keys()) # return all files in no keyword found (god help us)
    return list(set(relevant_files))


def create_knowledge_base(embeddings, selected_files):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_docs = []
    for filename in selected_files:
        filepath = os.path.join(DATA_FOLDER, filename)
        loader = TextLoader(filepath, encoding="utf-8")
        docs = loader.load()
        splits = text_splitter.split_documents(docs)

        metadata = DOCUMENT_DESCRIPTIONS[filename]

        enhanced_content = f"Document Description: {metadata['description']}\n"
        enhanced_content += f"Type: {metadata['type']}\n"
        if metadata['type'] == "schedule":
            enhanced_content += f"Program: {metadata.get('program', '')}\n"
            enhanced_content += f"Year: {metadata.get('year', '')}\n"
            if 'semigroup' in metadata:
                enhanced_content += f"Semigroup: {metadata.get('semigroup', '')}\n"
            if 'courses' in metadata:
                enhanced_content += f"Courses: {', '.join(metadata['courses'])}\n"

        enhanced_content += "\nDOCUMENT CONTENT:\n" + docs[0].page_content
        # assign the enhanced content and metadata to all splits
        for doc in splits:
            doc.metadata = {
                "source_file": filename,
                **{k: ", ".join(map(str, v)) for k, v in metadata.items()}
            }
            doc.page_content = enhanced_content

        all_docs.extend(splits)
    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        collection_name="university_data",
        persist_directory="./university_info_db"
    )
    return vector_store


def get_retriever_for_question(question):
    files = classify_document_by_question(question)
    print(f"[SYS] Relevant files: {files}")
    knowledge_base = create_knowledge_base(embeddings, selected_files=files)
    return knowledge_base.as_retriever()


#prompt = hub.pull("rlm/rag-prompt")
'''prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers based on the following context. Whenever you have links in your context, please include them in your answer and tell the user to visit the links for more information. Please answer in the language of the question."),
    ("user", "Context:\n{context}\n\nConversation History:\n{chat_history}\n\nQuestion:\n{question}")
])'''
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks related to the university Alexadru Ioan Cuza from Iasi. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know and redirect them to \"https://www.info.uaic.ro/\" for more information. Use five sentences maximum and keep the answer concise. Also, please include links relevant to the questions when you have them."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}\nAnswer:")
])
#You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
#memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=FileChatMessageHistory("chat_history.json"), return_messages=True)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print("You said:", query)
        return query
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return ""
    except sr.RequestError as e:
        print(f"Speech Recognition error: {e}")
        return ""

def get_answer(question, session_id):
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=FileChatMessageHistory(f"history/chat_history-{session_id}.json"), return_messages=True)

    retriever = get_retriever_for_question(question)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    result = qa_chain.invoke({"question": question})
    answer = result["answer"]

    if 'source_documents' in result:
        sources = list({doc.metadata['source_file'] for doc in result['source_documents']})
        answer += f"\n\n(Source: {', '.join(sources)})"
    return answer



if __name__ == "__main__":
    while(True):
        question = input("Ask: ").strip()
        # print(question)
        print(get_answer(question, 0))
        #print("updated memory: ", memory.load_memory_variables({}))


