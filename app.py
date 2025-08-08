from flask import Flask, request, jsonify, send_from_directory, send_file, make_response
import os, glob, json, re, time, io, shutil, requests, hashlib, traceback, secrets
from dotenv import load_dotenv
from threading import Thread, Timer
from functools import lru_cache, wraps
from collections import OrderedDict
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import logging

load_dotenv()

# ... (rest of unchanged code above)

# ==== AUDIO FILE SERVING ====
from flask import make_response

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    response = make_response(send_from_directory('static/audio', filename))
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ... (rest of unchanged code below)