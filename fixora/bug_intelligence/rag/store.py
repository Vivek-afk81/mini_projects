## Builds a FAISS index from doc chunks and saves it to disk.
## FAISS lets us search thousands of embeddings instantly
## by finding the closest vectors to a query.
import os
import json
import numpy as np
import faiss
from core.embedder import embed_text