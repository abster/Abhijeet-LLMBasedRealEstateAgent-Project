#!/usr/bin/env python
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
from IPython.display import Markdown, display
import re

os.environ["OPENAI_API_KEY"] = "<Your-OpenAI-Key>"
os.environ["OPENAI_API_BASE"] = "https://openai.vocareum.com/v1"

regenerate_listings = False
multi_modal_mode = True

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

listing_field_names = ['Listing Number', 'Neighborhood', 'Price', 'Bedrooms', 'Bathrooms', 'House Size','Description', 'Neighborhood Description']

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
    print("Generating real estate listings...")

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
        writer = csv.writer(csvfile, delimiter="|", quoting=csv.QUOTE_STRINGS)
        writer.writerow(listing_field_names)
        for index, listing in enumerate(listings):
            updated_listing = re.sub(r"\n+", "|", listing)
            listing_fields = updated_listing.split("|")
            updated_listing_fields = [ listing_field.replace(f"{listing_field_name}: ", "")
                                       for listing_field_name, listing_field in zip(listing_field_names[1:], listing_fields) ]
            writer.writerow([ index + 1 ] + updated_listing_fields)
        print(f"The generated real estate listings were written to generated-real-estate-listings.csv.\n\n")

# Load real estate listings from CSV file in a Vector database.
loader = CSVLoader(file_path='generated-real-estate-listings.csv', csv_args={ 'delimiter': '|',
                                                                              'fieldnames': listing_field_names })
print("Loading real estate listings...")
listings_data = loader.load()

embeddings = OpenAIEmbeddings()

if not os.path.exists("./chroma"):
    os.makedirs("./chroma")
    os.chmod("./chroma", 0o777) # permissions 0o777

persistent_client = chromadb.PersistentClient()
db = Chroma(
    client=persistent_client,
    collection_name="real-estate-listings-table",
    embedding_function=embeddings,
)

db.add_documents(documents=listings_data, ids = [str(id) for id in range(11)])

print("Real estate listings added to vector database.")

if multi_modal_mode:
    print("Loading real estate listing images...")

    # Load real estate images in a Vector database.
    clip_embeddings = OpenCLIPEmbeddings()
    db_images = LanceDB(
        table_name="real_estate_images",
        embedding=clip_embeddings,
    )
    image_uris = [ f"real-estate-image{row + 1}.jpg" for row in range(10) ]
    db_images.add_images(uris=image_uris, ids = [id for id in range(10)])
    print("Real estate listing images added to vector database.")

questions = [
    "How big do you want your house to be?"
    "What are the 3 most important things for you in choosing this property?",
    "Which amenities would you like?",
    "Which transportation options are important to you?",
    "How urban do you want your neighborhood to be?",
]
answers = [
    "A luxurious 4-bedroom house with a spacious kitchen and a cozy living room.",
    "An upscale neighborhood, top rated schools, and nearby shopping.",
    "An outdoor pool, patio, and spa like bathroom.",
    "Easy access to parks, walking trails and club houses",
    "A balance between suburban tranquility and access to urban amenities like restaurants and theaters.",
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

print("Generating a summary of user preferences...")

user_preference_summary = summarization_chain.invoke({"user_preferences": user_preferences})
user_visual_preference_summary = summarization_chain.invoke({"user_preferences": user_visual_preferences})

print(f"Summary of user preferences:\n{user_preference_summary.content}\n")
print(f"Summary of visual preferences:\n{user_visual_preference_summary.content}\n")

non_visual_search_results = db.as_retriever().invoke(user_preference_summary.content)
matching_listings = [ int(result.id) for result in non_visual_search_results ]

print(f"Listing ids matching user preference (non-visual): {matching_listings}")

if multi_modal_mode:
    print("Using vector database for images to select top listings based on visual preferences from user...")
    visual_search_results = db_images.similarity_search_by_vector(db_images._embedding.embed_query(user_visual_preference_summary.content))
    visual_matches = [ int(result.metadata["id"]) + 1 for result in visual_search_results ]
    print(f"Listing ids matching user preference (visual): {visual_matches}")
    intersection = list(set(matching_listings) & set(visual_matches))
    print(f"Listing ids matching both user preference (non-visual) and user preference(visual): {intersection}")
    if len(intersection) == 0:
        print(f"Sticking with listing ids matching user preference (non-visual): {matching_listings}")
    else:
        matching_listings = intersection

description_prompt = ChatPromptTemplate.from_messages(
    [("system",
      "You are a real-estate agent. Enhance the specified real estate listing description, so that it is tailored towards "
      "the user preference. Ensure that the description only includes factual information that was present in the specified "
      "real estate listing. You do not need to start with the phrase \"Enhanced real estate listing\"."),
     ("human", "Real estate listing:\n{description}\n User preference:\n{user_preference}")]
)

recommendations = []

for index, matching_listing in enumerate(matching_listings):
    row = int(matching_listing)
    print(f"Enhancing description for the listing {matching_listing} to match user preferences...")
    recommendation = llm.invoke(description_prompt.format(description=listings_data[row].page_content,
                                  user_preference=user_preference_summary.content))
    recommendations.append(recommendation.content)

display({'text/plain': "## Recommendations from LLM\n",
         'text/markdown': "## Recommendations from LLM\n"},
        raw=True)
for matching_listing, recommendation in zip(matching_listings, recommendations):
    row = int(matching_listing)
    display({'text/plain': f"### Listing {row}\n",
             'text/markdown': f"### Listing {row}\n"},
            raw=True)
    image_uri = f"./real-estate-image{row}.jpg"
    display(Markdown(f"![Picture for listing]({image_uri})"))
    display({'text/plain': recommendation,
             'text/markdown': recommendation},
            raw=True)