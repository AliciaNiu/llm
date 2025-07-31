import minsearch
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




def elastic_search(query):
    from elasticsearch import Elasticsearch
    es_client = Elasticsearch('http://localhost:9200')

    documents = read_doc()
    # index_settings = {
    #     "settings": {
    #         "number_of_shards": 1,
    #         "number_of_replicas": 0
    #     },
    #     "mappings": {
    #         "properties": {
    #             "text": {"type": "text"},
    #             "section": {"type": "text"},
    #             "question": {"type": "text"},
    #             "course": {"type": "keyword"} 
    #         }
    #     }
    # }

    index_name = "course-questions"

    # es_client.indices.create(index=index_name, body=index_settings)

    # from tqdm.auto import tqdm
    for doc in documents:
        es_client.index(index=index_name, body=doc)

    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    print(prompt)
    answer = llm(prompt)
    return answer

query = "I just saw the course, can I still enroll?"
query = "How do I get the certificate?"
rag(query)