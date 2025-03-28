#!/usr/bin/env python
# coding: utf-8

# This is a starter notebook for the project, you'll have to import the libraries you'll need, you can find a list of the ones available in this workspace in the requirements.txt file in this workspace.

import os
from langchain_openai.chat_models import ChatOpenAI
import csv
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_experimental.open_clip.open_clip import OpenCLIPEmbeddings

from langchain_chroma.vectorstores import Chroma
from langchain_community.vectorstores import LanceDB
import chromadb
from langchain.chains import RetrievalQA
import json
from IPython.display import Markdown, display
import re
from PIL import Image
import shutil

os.environ["OPENAI_API_KEY"] = "voc-6285383661266773632486678b43fb075333.75365710"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

regenerate_listings = False
multi_modal_mode = False

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

example_listing = """
Neighborhood: Green Oaks
Price: $800,000
Bedrooms: 3
Bathrooms: 2
House Size: 2,000 sqft

Description: Welcome to this eco-friendly oasis nestled in the heart of Green Oaks. This charming 3-bedroom, 2-bathroom home boasts energy-efficient features such as solar panels and a well-insulated structure. Natural light floods the living spaces, highlighting the beautiful hardwood floors and eco-conscious finishes. The open-concept kitchen and dining area lead to a spacious backyard with a vegetable garden, perfect for the eco-conscious family. Embrace sustainable living without compromising on style in this Green Oaks gem.

Neighborhood Description: Green Oaks is a close-knit, environmentally-conscious community with access to organic grocery stores, community gardens, and bike paths. Take a stroll through the nearby Green Oaks Park or grab a cup of coffee at the cozy Green Bean Cafe. With easy access to public transportation and bike lanes, commuting is a breeze.
"""

base_prompt = ("Generate a realistic and detailed real estate listing. You can refer to the provided example for how to "
               "structure the real estate listing:\n {example}")
prompt = PromptTemplate.from_template(
    base_prompt + "\n\n Do not repeat already generated listings: {generated_listings}")
summarization_prompt = PromptTemplate.from_template(
    "Summarize briefly in less than 10 words, capturing essential details of the specified real estate listing: {listing}")

generation_chain = prompt | llm
summarization_chain = summarization_prompt | llm

listings = []
summaries = []

generate_listings = not os.path.exists("generated-real-estate-listings.csv") or regenerate_listings

if generate_listings:
    # Generate 10 real-estate listings using LLM.
    # We generate 1 listing at a time to account for LLM's context window, and pass summaries of already generated listings
    # to prevent generating repeated/duplicated listings.
    for i in range(10):
        listing = generation_chain.invoke({"example": example_listing, "generated_listings": summaries})
        listings.append(listing.content)
        summary = summarization_chain.invoke({"listing": listing.content})
        summaries.append(summary.content)

    print(f"Generated {len(listings)} real estate listings!\n\n")

    # Save listings as csv file
    with open('generated-real-estate-listings.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        for index, listing in enumerate(listings):
            updated_listing = listing.replace('\n', '|')
            writer.writerow([f"Listing Number: {index}|{updated_listing}"])

# Load real estate listings from CSV file in a Vector database.
loader = CSVLoader(file_path='generated-real-estate-listings.csv')
listings_data = loader.load()

embeddings = OpenAIEmbeddings()
shutil.rmtree('./chroma')
persistent_client = chromadb.PersistentClient()
db = Chroma(
    client=persistent_client,
    collection_name="real-estate-listings",
    embedding_function=embeddings,
)

db.add_documents(documents=listings_data)

if multi_modal_mode:
    # Load real estate images in a Vector database.
    clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-g-14", checkpoint="laion2b_s34b_b88k")
    db_images = LanceDB(
        table_name="real_estate_images",
        embedding=clip_embeddings,
    )
    image_uris = [ f"real-estate-image{row + 1}.jpg" for row in range(10) ]
    db_images.add_images(uris=image_uris)

questions = [
    "How big do you want your house to be?"
    "What are 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
    "What exterior color would you like your house to have?"
]
answers = [
    "A comfortable 3-bedroom house with a spacious kitchen and a cozy living room.",
    "A quiet neighborhood, good local schools, and shopping malls.",
    "A backyard for gardening, a two-car garage, and a modern, energy-efficient heating system.",
    "Easy access to a reliable bus line, nearby parks, and bike-friendly roads.",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
    "Tuscan yellow will be cool."
    ]

visual_questions = [
    "What exterior color would you like your house to have?"
]
visual_answers = [
    "Tuscan yellow will be cool."
]

user_preferences = [{"criteria": question, "user_response": answer} for question, answer in zip(questions, answers)]
user_visual_preferences = [{"criteria": question, "user_response": answer} for question, answer in zip(visual_questions, visual_answers)]

summarization_prompt = PromptTemplate.from_template(
    "Summarize briefly, capturing essential details of user preferences based on provided questions and answers: {user_preferences}")
summarization_chain = summarization_prompt | llm

user_preference_summary = summarization_chain.invoke({"user_preferences": user_preferences})
user_visual_preference_summary = summarization_chain.invoke({"user_preferences": user_visual_preferences})

display({'text/plain': f"## Summary of user preferences\n{user_preference_summary.content}\n",
         'text/markdown': f"## Summary of user preferences\n{user_preference_summary.content}\n"},
        raw=True)

rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())

selection_prompt = ChatPromptTemplate.from_messages(
    [("system",
      "You are a real-estate agent. You MUST answer based on the provided context. Based on the real-estate listings in "
      "the context and user's preferences, select the best matching real-estate listings. Only output a json array containing "
      "listing numbers for matching listings."),
     ("human", "{question}")]
)

max_attempts = 3
attempts = 0
matching_listings = []
while (len(matching_listings) == 0 and attempts < max_attempts):
    llm_response = rag.invoke(selection_prompt.format(question=user_preference_summary.content))
    print(llm_response)
    matching_listings = json.loads(llm_response['result'].replace("```json\n", "").replace("```", ""))
    attempts += 1

print(f"Listing numbers from LLM: {matching_listings}")

if multi_modal_mode:
    image_results = db_images.similarity_search_by_vector(db_images._embedding.embed_query(user_visual_preference_summary.content))
    print(f"Image results: {image_results}")

description_prompt = ChatPromptTemplate.from_messages(
    [("system",
      "You are a real-estate agent. Adapt the specified real estate listing description, so that it is tailored towards "
      "the user preference. Ensure that the description only includes factual information that was present in the specified "
      "real estate listing. You do not need to start with the phrase \"Adapted real estate listing\""),
     ("human", "Real estate listing:\n{description}\n User preference:\n{user_preference}")]
)

recommendations = []

for index, matching_listing in enumerate(matching_listings):
    row = int(matching_listing)
    if row == 0:
        description = re.sub(r":.*", "", listings_data[0].page_content)
    else:
        description = re.sub(r".*:", "", listings_data[row - 1].page_content)

    recommendation = llm.invoke(
        description_prompt.format(description=description, user_preference=user_preference_summary.content))
    recommendations.append(recommendation.content)

display({'text/plain': "## Recommendations from LLM\n",
         'text/markdown': "## Recommendations from LLM\n"},
        raw=True)
for matching_listing, recommendation in zip(matching_listings, recommendations):
    row = int(matching_listing)
    display(Markdown(f"### Listing {row}\n"))
    img = Image.open(f"real-estate-image{row + 1}.jpg")
    img.show()
    display(Markdown(f""))
    display({'text/plain': recommendation,
             'text/markdown': recommendation},
            raw=True)
