#import JSON module
import json

#import numpy
import numpy as np

#import openAI
from openai import OpenAI
client = OpenAI()

#turn listings into one string
def listings_string(item):
    return " ".join([
        item.get("title", ""),
        item.get("category", ""),
        item.get("subcategory", ""),
        " ".join(item.get("attributes", [])),
        item.get("condition", ""),
        item.get("colour", ""),
        item.get("location", ""),
        str(item.get("price", ""))
    ])

#open JSON listings dataset
with open("500_listings_realistic.json", "r", encoding="utf-8") as file:
    #load listings into python structure
    listings = json.load(file)

#create dictionary for listing vectors
listing_vectors = {}

#loop through listings and create embeddings
for item in listings:
    embedding = client.embeddings.create(
        #select embedding model
        model="text-embedding-3-small",
        #input listing string
        input=listings_string(item)
    ).data[0].embedding

    #store embedding and its norm 
    listing_vectors[str(item["id"])] = {
        "embedding": embedding,
        "norm": float(np.linalg.norm(embedding))
    }
    #print number of listings embedded
    print("Embedded:", item["id"])

#create new JSON file for listing vectors
with open("listing_vectors.json", "w", encoding="utf-8") as file:
    json.dump(listing_vectors, file)

