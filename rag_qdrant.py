
import json

def read_doc():
    with open('documents.json', 'rt') as f_in:
        docs_raw = json.load(f_in)

    documents = []

    for course_dict in docs_raw:
        for doc in course_dict['documents']:
            doc['course'] = course_dict['course']
            documents.append(doc)

    return documents

documents= read_doc()

from qdrant_client import QdrantClient, models
client = QdrantClient("http://localhost:6333")

from fastembed import TextEmbedding
TextEmbedding.list_supported_models()


EMBEDDING_DIMENSIONALITY = 512

# for model in TextEmbedding.list_supported_models():
#     if model["dim"] == EMBEDDING_DIMENSIONALITY:
#         print(json.dumps(model, indent=2))


model_handle = "jinaai/jina-embeddings-v2-small-en"

# Define the collection name
collection_name = "qdrant-rag"

# Create the collection with specified vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
        distance=models.Distance.COSINE  # Distance metric for similarity search
    )
)

points = []
id = 0



for doc in documents:
    point = models.PointStruct(
        id=id,
        vector=models.Document(text=doc['text'], model=model_handle), #embed text locally with "jinaai/jina-embeddings-v2-small-en" from FastEmbed
        payload={
            "text": doc['text'],
            "section": doc['section'],
            "course": doc['course']
        } #save all needed metadata fields
    )
    points.append(point)
    id += 1

client.upsert(
    collection_name=collection_name,
    points=points
)


def search(query, limit=1):
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle 
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )
    return results


import random
course_piece = random.choice(documents)
print(json.dumps(course_piece, indent=2))
result = search(course_piece['question'])
print(f"Query: {course_piece['question']}")
print(result)
print(search("What if I submit homeworks late?").points[0].payload['text'])


client.create_payload_index(
    collection_name=collection_name,
    field_name="course",
    field_schema="keyword" # exact matching on string metadata fields
)

def search_in_course(query, course="mlops-zoomcamp", limit=1):
    results = client.query_points(
        collection_name=collection_name,
        query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
            text=query,
            model=model_handle
        ),
        query_filter=models.Filter( # filter by course name
            must=[
                models.FieldCondition(
                    key="course",
                    match=models.MatchValue(value=course)
                )
            ]
        ),
        limit=limit, # top closest matches
        with_payload=True #to get metadata in the results
    )
    return results

print(search_in_course("What if I submit homeworks late?", "mlops-zoomcamp").points[0].payload['text'])
print(search_in_course("What if I submit homeworks late?", "data-engineering-zoomcamp").points[0].payload['text'])
