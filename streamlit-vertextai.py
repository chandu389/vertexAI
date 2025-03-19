#!/usr/bin/env python
# coding: utf-8

__import__('pysqlite3')
import sys
from typing import Any, Dict
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import json
import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import vertexai
from langchain_community.embeddings import HuggingFaceEmbeddings
from google.oauth2 import service_account


# First load the JSON string
raw_credentials = st.secrets["gcp"]["credentials"]


st.write(raw_credentials)

try:
    # Decode the first level (string to dict)
    decoded_credentials: Dict[str, Any] = json.loads(raw_credentials)
    st.write("Decoded Credentials:", decoded_credentials)
except json.JSONDecodeError as e:
    st.error(f"Error decoding JSON: {e}")
    st.stop()

decoded_credentials["private_key"] = st.secrets["gcp"]["private_key"]
creds = service_account.Credentials.from_service_account_info(decoded_credentials)

vertexai.init(
    project=credentials["project_id"],
    location="us-central1",
    credentials=creds
)
llm = VertexAI(
    model="gemini-1.0-pro-002"
)

embeddings = HuggingFaceEmbeddings()


vector_store = Chroma(
    collection_name="test_db", # Collection Name where to store the data .
    embedding_function=embeddings
)

retriver = vector_store.as_retriever(search_type="mmr", search_kwargs={'k':20})


custom_prompt_template = """
You are an expert in ETSI VNFD and helping vendors to identify the suitable VNFD configurations.
Give only yaml configs which is used for creating the SOL001 VNFD. This yaml should readily usable to the vendors. 
This yaml is to be used with Cisco NFVO and ESC. Provider is always Cisco. Below are some of the details to be used always.

provider: Cisco
template_name: cisco_generic_vnf
template_author: Cisco
product_name: STC A2G
product_info_name: STC A2G
node_type should be cisco_generic_vnf
VDU naming convention should be like vdu1, vdu2 etc..
Connection points(interfaces) should be like vdu1_cp1, vdu1_cp2 for vdu1 and vdu2_cp1, vdu2_cp2 for vdu2 etc..
If volumes present, volume name should be like vdu1_volume1, vdu1_volume2 for vdu1 and vdu2_volume1, vdu2_volume2 for vdu2 etc..

The SOL001 Yaml file must contain the below details:
1. tosca_definitions_version
2. imports
3. metadata
4. dsl_definitions
5. node_types
6. topology_template
     --> node_templates -> vnf
     --> vdu
        --> properties
        --> sw_image_data
        --> vdu_profile
        --> configurable_properties
    --> interfaces(connection points)
    --> policies
        --> tosca.policies.nfv.InstantiationLevels
        --> tosca.policies.nfv.VduInstantiationLevels
        --> tosca.policies.nfv.VduInitialDelta
        --> tosca.policies.nfv.ScalingAspects
        --> tosca.policies.nfv.VduScalingAspectDeltas
        --> tosca.policies.nfv.AntiAffinityRule (If anti-affinity is needed between VDUs)
        --> tosca.groups.nfv.PlacementGroup (If anti-affinity is needed then placement group is also needed)

Don't ignore any of the above sections. If any of the sections are not present, please add them with default values.
    
STRICTLY follow these rules:
- Output **only** valid YAML.
- Do **not** include explanations, comments, or additional text.
- Do **not** use code blocks (```)—just raw YAML.

context : {context}

conversation history: {chat_history}

input: {question}

"""


custom_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=custom_prompt_template,
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the ConversationalRetrievalChain
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriver,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# Streamlit app
st.title("VNFD Designer")
st.write("Enter your query to generate the VNFD configuration:")

query = st.text_area("Query", """
Generate a VNFD with 2 VDUs.
1. Descriptor ID is STCA2G.
2. First VDU will only have 1 interface and 2nd VDU will have 2 interfaces.
3. First VDU minimum instances is 1 and maximum 10. Second VDU minimun and maximum is 1
4. Intra VDU anti affinity is not needed

Please provide the tosca sol001 yaml configuration for the above requirements.
""")

if st.button("Generate Configuration"):
    chat_history = memory.load_memory_variables({})["chat_history"]
    response = conv_chain({"question": query, "chat_history": chat_history})
    st.write("Generated Configuration:")
    st.code(response["answer"], language="yaml")