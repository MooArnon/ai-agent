###########
# Imports #
##############################################################################

import json

from ai_agent.embedding import OpenAIEmbedding
from ai_agent.database import QdrantVectorDatabase

##############################################################################

ai_agent = OpenAIEmbedding()
qdrant = QdrantVectorDatabase()
text = ["Hello, wow and see"]

vector_payload = ai_agent.get_text_embeddings(
    texts=text,
    chunk_size=10,
    chunk_overlap=1
)

with open('result.json', "w") as f:
    json.dump(vector_payload, f, indent=4)

qdrant.ingest_data(
    vector_payload,
    "test"
)


query = qdrant.query_data(
    query_text="hello",
    embedding_client=ai_agent,
    collection_name="test"
)
print(query)

data = qdrant.get_by_article_id(
    article_id="xxx",
    collection_name="test",
)
print(data)


# qdrant.create_article_id_index('test', 'article_id')
##############################################################################
