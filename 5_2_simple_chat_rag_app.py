import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

try:
    from langchain_text_splitters import CharacterTextSplitter
except ImportError:
    from langchain.text_splitter import CharacterTextSplitter


load_dotenv()

APP_TITLE = "개인 문서 기반 Q&A"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def require_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(".env 파일 또는 환경변수에 OPENAI_API_KEY를 설정해주세요.")
        st.stop()
    return api_key


def extract_text_from_pdf(uploaded_file) -> str:
    try:
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as error:
        st.error(f"PDF에서 텍스트를 추출하는 중 오류가 발생했습니다: {error}")
        return ""


def split_text_to_documents(text: str) -> list[Document]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return text_splitter.create_documents([text])


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_name: str, file_bytes: bytes):
    pdf_file = BytesIO(file_bytes)
    pdf_file.name = file_name
    document_text = extract_text_from_pdf(pdf_file)

    if not document_text:
        return None, 0

    documents = split_text_to_documents(document_text)
    embeddings = OpenAIEmbeddings(api_key=require_openai_api_key())
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, len(documents)


def get_rag_response(user_query: str, vectorstore) -> tuple[str, list[Document]]:
    retrieved_docs = vectorstore.similarity_search(user_query, k=3)
    retrieved_text = "\n\n".join(
        f"문서 {index + 1}:\n{document.page_content}"
        for index, document in enumerate(retrieved_docs)
    )

    prompt = (
        "제공된 문서 내용만 근거로 사용자의 질문에 한국어로 답하세요.\n"
        "문서에 없는 내용은 추측하지 말고 '문서에서 확인할 수 없습니다'라고 답하세요.\n\n"
        f"질문:\n{user_query}\n\n"
        f"검색된 문서:\n{retrieved_text}"
    )

    chat_model = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0,
        api_key=require_openai_api_key(),
    )
    response = chat_model.invoke(prompt)
    return response.content, retrieved_docs


st.set_page_config(page_title=APP_TITLE)
st.title(APP_TITLE)

require_openai_api_key()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "PDF 문서를 업로드한 뒤 문서 내용에 대해 질문해주세요.",
        }
    ]

uploaded_file = st.file_uploader("PDF 문서를 업로드하세요", type=["pdf"])

if uploaded_file:
    with st.spinner("문서를 읽고 벡터 인덱스를 만드는 중입니다..."):
        vectorstore, chunk_count = build_vectorstore(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )

    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.success(f"{chunk_count}개의 문서 조각을 생성했습니다.")
else:
    st.session_state.vectorstore = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_query = st.chat_input("업로드한 문서를 기반으로 질문하세요")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.write(user_query)

    if not st.session_state.get("vectorstore"):
        assistant_response = "먼저 PDF 문서를 업로드해주세요."
        retrieved_docs = []
    else:
        with st.spinner("문서에서 근거를 찾고 답변을 생성하는 중입니다..."):
            assistant_response, retrieved_docs = get_rag_response(
                user_query,
                st.session_state.vectorstore,
            )

    with st.chat_message("assistant"):
        st.write(assistant_response)

        if retrieved_docs:
            with st.expander("검색된 근거 문서 보기"):
                for index, document in enumerate(retrieved_docs, start=1):
                    st.markdown(f"**문서 {index}**")
                    st.write(document.page_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )

#uv add streamlit
#uv run streamlit run .\04_vectordb\11_5_2_simple_chat_rag_app.py