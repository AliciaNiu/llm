
import json
import uuid

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
client.get_collections()


def build_client(collection_name, client, model_handle):
    from fastembed import TextEmbedding
    TextEmbedding.list_supported_models()

    EMBEDDING_DIMENSIONALITY = 512

    # for model in TextEmbedding.list_supported_models():
    #     if model["dim"] == EMBEDDING_DIMENSIONALITY:
    #         print(json.dumps(model, indent=2))


    # client.delete_collection(collection_name=collection_name) # delete the collection if it exists

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
    return client


def search(collection_name, client, model_handle, query, limit=1):
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


# import random
# course_piece = random.choice(documents)
# print(json.dumps(course_piece, indent=2))
# result = search(course_piece['question'])
# print(f"Query: {course_piece['question']}")
# print(result)
# print(search("What if I submit homeworks late?").points[0].payload['text'])


def search_in_course(collection_name, client, model_handle, query, course="mlops-zoomcamp", limit=5):
    client.create_payload_index(
        collection_name=collection_name,
        field_name="course",
        field_schema="keyword" # exact matching on string metadata fields
    )

    points = client.query_points(
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
    results = []
    for point in points.points:
        results.append(point.payload)
    return results

# print(search_in_course("What if I submit homeworks late?", "mlops-zoomcamp").points[0].payload['text'])
# print(search_in_course("What if I submit homeworks late?", "data-engineering-zoomcamp").points[0].payload['text'])


def build_prompt(query, search_results):
    prompt_template = """
        You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
        Use only the facts from the CONTEXT when answering the QUESTION.

        QUESTION: {question}

        CONTEXT: 
        {context}
        """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt



def llm(prompt):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


def rag(query, course="mlops-zoomcamp", limit=5):
    # Define the collection name
    collection_name = "qdrant-rag"
    model_handle = "jinaai/jina-embeddings-v2-small-en"
    client = build_client(collection_name, client, model_handle)
    search_results = search_in_course(collection_name, client, model_handle, query, course, limit=limit).points
    if not search_results:
        return "No relevant information found."
    
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    
    return answer


def build_sparse_client(client):
    collections = client.get_collections().collections
    # collections = [
    #     CollectionDescription(name='qdrant-sparse'),
    #     CollectionDescription(name='qdrant-rag')
    # ]
    existing_collections = [col.name for col in collections]
    print(f"Existing collections: {existing_collections}")

    if "qdrant-sparse" not in existing_collections:
        # Create the collection with specified sparse vector parameters
        client.recreate_collection(
            collection_name="qdrant-sparse",
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )


    # Send the points to the collection
    client.upsert(
        collection_name="qdrant-sparse",
        points=[
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "bm25": models.Document(
                        text=doc["text"], 
                        model="Qdrant/bm25",
                    ),
                },
                payload={
                    "text": doc["text"],
                    "section": doc["section"],
                    "course": doc["course"],
                }
            )
            for doc in documents
        ]
    )

    return client

def sparse_search(client, query: str, limit: int = 1) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name="qdrant-sparse",
        query=models.Document(
            text=query,
            model="Qdrant/bm25",
        ),
        using="bm25",
        limit=limit,
        with_payload=True,
    )

    return results.points


def build_hybrid_client(client):
    collections = client.get_collections().collections
    # collections = [
    #     CollectionDescription(name='qdrant-sparse'),
    #     CollectionDescription(name='qdrant-rag')
    # ]
    existing_collections = [col.name for col in collections]
    collection_name = "qdrant-sparse-and-dense"
    # Create the collection with both vector types
    if collection_name not in existing_collections:
        print(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                # Named dense vector for jinaai/jina-embeddings-v2-small-en
                "jina-small": models.VectorParams(
                    size=512,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            }
        )

    client.upsert(
        collection_name="qdrant-sparse-and-dense",
        points=[
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector={
                    "jina-small": models.Document(
                        text=doc["text"],
                        model="jinaai/jina-embeddings-v2-small-en",
                    ),
                    "bm25": models.Document(
                        text=doc["text"], 
                        model="Qdrant/bm25",
                    ),
                },
                payload={
                    "text": doc["text"],
                    "section": doc["section"],
                    "course": doc["course"],
                }
            )
            for doc in documents
        ]
    )
    return client


def multi_stage_search(client, query: str, limit: int = 1) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name="qdrant-sparse-and-dense",
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="jinaai/jina-embeddings-v2-small-en",
                ),
                using="jina-small",
                # Prefetch ten times more results, then
                # expected to return, so we can really rerank
                limit=(10 * limit),
            ),
        ],
        query=models.Document(
            text=query,
            model="Qdrant/bm25", 
        ),
        using="bm25",
        limit=limit,
        with_payload=True,
    )

    return results.points


def rrf_search(client, query, limit: int = 1) -> list[models.ScoredPoint]:
    results = client.query_points(
        collection_name="qdrant-sparse-and-dense",
        prefetch=[
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="jinaai/jina-embeddings-v2-small-en",
                ),
                using="jina-small",
                limit=(5 * limit),
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="Qdrant/bm25",
                ),
                using="bm25",
                limit=(5 * limit),
            ),
        ],
        # Fusion query enables fusion on the prefetched results
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        with_payload=True,
    )

    return results.points

def sparse_rag():
    client = QdrantClient("http://localhost:6333")

    sp_client = build_sparse_client(client)
    query = 'pandas'
    # query = 'Qdrant'
    search_results = sparse_search(sp_client, query)
    
    print(search_results)
    if search_results:
        print(search_results[0].payload['text'])
        print(search_results[0].score)


def hybrid_rag():
    client = QdrantClient("http://localhost:6333")

    hybrid_client = build_hybrid_client(client)
    query = 'What if I submit homeworks late?'
    # query = 'Qdrant'
    search_results = rrf_search(hybrid_client, query)
    
    print(search_results)
    if search_results:
        print(search_results[0].payload['text'])
        print(search_results[0].score)

