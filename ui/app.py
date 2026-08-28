import requests
import streamlit as st

GATEWAY_URL = "http://localhost:8080"
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Athenaeum", page_icon="📚")
st.title("📚 Athenaeum")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_id = st.text_input("User ID", value="amith")

st.header("Upload a document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "html", "htm"])

if uploaded_file is not None:
    if st.button("Upload"):
        if not user_id.strip():
            st.error("Enter a User ID before uploading.")
        else:
            try:
                response = requests.post(
                    f"{GATEWAY_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    data={"user_id": user_id},
                    timeout=30,
                )
                if response.ok:
                    body = response.json()
                    st.success(f"Queued for processing. Job ID: {body.get('job_id')}")
                else:
                    st.error(f"Upload failed ({response.status_code}): {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the upload service: {e}")

st.header("Ask a question")

for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if entry["sources"]:
            st.caption("Sources:")
            for s in entry["sources"]:
                if s.get("page") is not None:
                    st.caption(f"- {s['filename']}, page {s['page']}")
                else:
                    st.caption(f"- {s['filename']}")

question = st.chat_input("Ask a question about your uploaded documents")

if question:
    if not user_id.strip():
        st.error("Enter a User ID before asking a question.")
    else:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"question": question, "user_id": user_id},
                        timeout=60,
                    )
                    if response.ok:
                        body = response.json()
                        st.write(body["answer"])
                        sources = body.get("sources", [])
                        if sources:
                            st.caption("Sources:")
                            for s in sources:
                                if s.get("page") is not None:
                                    st.caption(f"- {s['filename']}, page {s['page']}")
                                else:
                                    st.caption(f"- {s['filename']}")
                        st.session_state.chat_history.append(
                            {"question": question, "answer": body["answer"], "sources": sources}
                        )
                    else:
                        st.error(f"Query failed ({response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the query service: {e}")
