#run in terminal:
#pip install openai
#pip install numpy
#setx OPENAI_API_KEY "your_api_key_here"
#Key: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

#import JSON module
import json

#import Flask
from flask import Flask, request
#create Flask app
app = Flask(__name__)

#import openAI
from openai import OpenAI
client = OpenAI()

#import numpy
import numpy as np

#open JSON listings dataset
with open('500_listings_realistic.json', 'r', encoding="utf-8") as file:
    #load listings into python structure
    listings = json.load(file)

#embedding - load listing vectors
with open("listing_vectors.json", "r", encoding="utf-8") as file:
    listing_vectors = json.load(file)

#test number of listing vectors
#print("Embeddings loaded:", len(listing_vectors))

#llm interperets user query 
def interpret_user_query(search_query: str):
    
    #creating first prompt to be sent to LLM
    prompt= (
    
    #provide context of task
    "\nYou are interpreting a user's search query for an online second-hand marketplace.\n"
    "\nThe interpreted query will also be used for semantic search (embedding) and metadata filtering.\n\n"     
    
    #provide the user's natural language search query
    +"User's entered search query:" + search_query + "\n\n"
    
    #define llm task
    +"Your task:\n"
    +"1. Correct any spelling errors.\n"
    +"2. Interpret the user's intent. Keeping it short and clear.\n"
    +"3. Extract any relevant filters. ONLY if clearly stated.\n\n"
    
    #define what llm should return
    +"Return JSON only in this exact format:\n"
    +"  {\n"
            #interpreted query
    +'      "interpreted_query": <string>, \n'
            #filters
    +'      "filters": {\n'
    +'          "min_price": <integer or null>,\n'
    +'          "max_price": <integer or null>,\n'
    +'          "location": <string or null>,\n'
    +'          "condition": <string or null>,\n'
    +'          "colour": <string or null>,\n'
    +'          "brand": <string or null>,\n'
    +'          "size": <string or null>\n'
    +"      }\n"
    +"  }\n\n"
    
    #rules for filters:
    #price filters
    +"Price rule:\n"
    +"If a price (min or max) is stated, extract the exact number, do not modify.\n\n"
    #location filter
    +"Location rule:\n"
    +"If user states a location (e.g. near Birmingham), only extract the location name (e.g. Birmingham).\n"
    +"Do not include the word 'near' or similar words.\n\n"
    #condition filter
    +"Condition rule:\n" 
    +"If user states condition (e.g. good condition), only extract the condition label (e.g. good).\n"
    +"Do not include the word 'condition'.\n\n"
    #brand filter
    +"Brand rule:\n" 
    +"Only set brand if clear product brand stated (e.g. Nike, Apple...).\n"
    +"Brand usually appears in the listings title or attributes.\n\n"
    #size filter
    +"Size rule:\n" 
    +"Only set size if it's in terms of clothing (e.g. S, extra-large, UK10, 32).\n"
    +"Do not include other descriptive adjectives such as 'big'.\n\n"
    
    #define output constraints
    +"If a filter is not clearly stated or you are unsure, return null.\n"
    +"Do not invent information.\n"
    +"Return JSON only. Do not include any extra text.\n"
    )

    #sending propmpt to LLM
    response = client.responses.create(
        #specify which LLM model to use
        model="gpt-5-nano",
        #provide prompt
        input=prompt
    )

    #convert llm response from JSON into python dictionary
    return json.loads(response.output_text)

#apply metadata filtering to listings using interpreted filters
def filter_listings(listings, filters):

    #create list for filtered listings
    filtered_listings = []

    #loop through and filter listings
    for item in listings:

        #listings start off as valid
        match = True

        # min price filter
        if filters.get("min_price") is not None:
             #if listings violates filter
             if item.get("price", 0) < filters["min_price"]:
                #listing becomes invalid
                match = False

        # max price filter
        if filters.get("max_price") is not None:
            if item.get("price", 0) > filters["max_price"]:
                match = False
        
        # location filter
        if filters.get("location"):
            #convert filter into lowercase and compare with listing
            if item.get("location", "").lower() != filters["location"].lower():
                match = False
        
        # condition filter
        if filters.get("condition"):
            #if the stated condition is one of the accepted values
            if filters["condition"].lower() in ["new", "good", "fair", "poor", "bad"]:
                if filters["condition"].lower() not in item.get("condition", "").lower():
                    match = False

        # colour filter
        if filters.get("colour"):
            #extracts colour from filter and also listings
            colour_value = filters["colour"].lower()
            item_colour = item.get("colour", "").lower()
            #if colour is not mentioned
            if colour_value not in item_colour:
                match = False
        
        # brand filter
        if filters.get("brand"):
            # convert listing title and attributes into lowercase
            title_text = item.get("title", "").lower()
            attributes_text = " ".join(item.get("attributes", [])).lower()
            #checks if filtered brand is mentioned in title or attributes
            if (filters["brand"].lower() not in title_text and
                filters["brand"].lower() not in attributes_text):
                match = False
        
        #size filter
        if filters.get("size"):
            #convert user input into lowercase and remove any spaces
            size_value = filters["size"].lower().strip()
            title_text = item.get("title", "").lower()
            attributes_text = " ".join(item.get("attributes", [])).lower()
            #allow different word variations for sizing
            #if user types:
            if size_value in ["xs", "extra small"]:
                #allow:
                valid_size = ["xs", "extra small"]
            elif size_value in ["s", "small"]:
                valid_size = ["s", "small"]
            elif size_value in ["m", "medium"]:
                valid_size = ["m", "medium"]
            elif size_value in ["l", "large"]:
                valid_size = ["l", "large"] 
            elif size_value in ["xl", "extra large"]:
                valid_size = ["xl", "extra large"]
            elif size_value in ["xxl", "extra extra large"]:
                valid_size = ["xxl", "extra extra large"]
            elif size_value in ["xxxl", "extra extra extra large"]:
                valid_size = ["xxxl", "extra extra extra large"]                     
            #if size is not in list, use what was entered
            else:
                valid_size = [size_value]
            #assume size does not match by default
            size_match = False
            #loop through entered sizes
            for listing_size in valid_size:
                #if size appears in listing title or attribute
                if listing_size in title_text or listing_size in attributes_text:
                    #it's valid
                    size_match = True
            #if listing does not match size, invalid
            if not size_match:
                match = False

        #add valid listings to list
        if match:
            filtered_listings.append(item)

    #return filtered listings
    return filtered_listings

#llm based search
def llm_search(search_query: str):

    #interpret user search query using LLM 
    query_data = interpret_user_query(search_query)

    #apply metadata filtering
    filtered_listings = filter_listings(listings, query_data["filters"])
    #test metadata filtering
    #print("After metadata filter:", len(filtered_listings))

    #test embedding pt.1
    #print("Top 3 items before embedding:")
    #for item in filtered_listings[:3]:
        #print(item["id"], item.get("title", ""))

    #embedding - create query vector
    query_vector = client.embeddings.create(
        #select embedding model
        model="text-embedding-3-small",
        #input interpreted query
        input=query_data.get("interpreted_query", "")
    ).data[0].embedding
    
    #compute query norm once
    query_norm = np.linalg.norm(query_vector)

    #cosine similarity
    filtered_listings = sorted(
        filtered_listings,
        key=lambda item:(
            np.dot(query_vector, listing_vectors[str(item["id"])]["embedding"])
            / (query_norm * listing_vectors[str(item["id"])]["norm"])
        ),
        reverse=True
    #keep only top 40 listings
    )[:40]
    #test embedding results
    #print("After embedding Top-K:", len(filtered_listings))

    #test embedding pt.2
    #print("Top 3 items after embedding:")
    #for item in filtered_listings[:3]:
        #print(item["id"], item.get("title", ""))

    #create list to store listing chunks
    listing_chunks = []
    
    #loop through filtered listings
    for item in filtered_listings:
        #build text chunk for each listing
        chunk = (
            "ID:" + str(item["id"]) + " | " +
            item.get("title", "") + " | " +
            "£" + str(item.get("price", "")) + " | " +
            item.get("location", "") + " | " +
            item.get("condition", "") + " | " +
            item.get("colour", "") + " | " +
            ", ".join(item.get("attributes", []))
        )
        #add chunk to list
        listing_chunks.append(chunk)

    #combine all chunks into single string
    listings_string = "\n".join(listing_chunks)

    #creating second prompt to be sent to LLM
    prompt = (
    
    #provide context of task
    "\nHelp refine search results for second-hand items online.\n" 
    +"\nThe user has described the item they are looking for in natural language.\n" 
    +"\nThe system has already retrieved the 50 most relevant candidate listings using semantic search (embeddings).\n\n" 
    
    #provide user's search query
    +"User's search query: " + search_query + "\n"
    #provide the interpreted query
    +"Interpreted query:" + query_data["interpreted_query"] + "\n\n"
    
    #provide all item listings
    +"Candidate listings (top-k from embedding retrieval):\n" + listings_string + "\n\n"
   
    #define llm task
    +"\nSelect the most relevant listings, only from the candidate listings.\n"
    +"\nConsider all relevant factors when selecting listings.\n\n" 
   
    #define what LLM should return
    +"Return the selected listings, ONLY in the following JSON format:\n" 
    +"  {\n"
    +'      "matches": [\n'
    +"          {\n"
    +'               "id": <integer>,\n'
    +'               "title": <string>,\n'
    +'               "category": <string>,\n'
    +'               "subcategory": <string>,\n'
    +'               "price": <integer>,\n'
    +'               "condition": <string>,\n'
    +'               "location": <string>,\n'
    +'               "colour": <string>,\n'
    +'               "attributes": [<string>, <string>...]\n'
    +"           }\n"
    +"     ]\n"
    +"  }\n\n"

    #specify how many listings to return
    +"Select and return maximum 25 most relevant listings only. Do not return more than 25.\n"
    +"If fewer than 25 are genuine matches, return fewer.\n\n"
    
    #Constraints:
    +"Constraints:\n"
    +"1.Only return items that genuinely match what the user is asking for.\n"
    +"2.Stay within the same general item type.\n"
    +"For example, a search for 'jumper' could return sweatshirts, but not dresses, trousers, or shoes.\n"
    +"A search for 'jacket' may include coats as they are both outerwear, but it should not return jumpers.\n"
    +"If there are no suitable matches, return an empty matches array.\n"
    +"3.Only use the provided candidate listings.\n"
    +"4.Do not invent new listings or IDs.\n"
    )

    #sending propmpt to LLM
    response = client.responses.create(
        #specify which LLM model to use
        model="gpt-5-nano",
        #provide prompt
        input=prompt
    )

    #convert LLM response from JSON into python dictionary
    llm_response_python = json.loads(response.output_text)

    #get matching listings, interpreted query and filters from response
    return {
        "matches": llm_response_python.get("matches", []),
        "interpreted_query": query_data["interpreted_query"],
        "filters": query_data["filters"]
    }

#create route for home page
@app.route("/", methods=["GET", "POST"])
def user_interface():
    
    #creating a html
    html = """
    <!-- document type -->
    <!doctype html>
    <html>
      
      <head>
        <style>

            /* entire page */
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                overflow: hidden;
            }

            /* page heading */
            .header {
                background: linear-gradient(90deg,#5d89ba, #1f4f8a, #5d89ba);
                color: white;
                padding: 30px 0;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 0;
                overflow: hidden;
            }

            /* search bar container */
            .search_container {
                text-align: center;
                margin-top: 45px;
            }

            /* search input */
            .search_input {
                width: 400px;
                padding: 12px;
                font-size: 16px;
                border: 1px solid #000000;
                border-radius: 6px;
            }

            /* search button */
            .search_button {
                color: white;
                padding: 12px 18px;
                font-size: 16px;
                background-color: #2f5f96;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                margin-left: 8px;
                font-weight: 600;
            }

            /* search button - hover */
            .search_button:hover {
                background-color: #3c6fa8;
            }

            /* display search query */
            .display_search_query {
                font-size: 18px;
                font-weight: 550;
                margin-bottom: 25px; 
                margin-left: 40px;
                text-align: left;
                margin-top: 40px;
            }

            /* display llm interpretation */
            .display_interpretation {
                font-size: 15px;
                margin-top: 15px;
                margin-left: 40px;
                text-align: left;
            }

            /* display llm filters */
            .display_filters {
                font-size: 15px;
                margin-bottom: 25px; 
                margin-left: 40px;
                text-align: left;
            }

            /* results container */
            .results_container {
                width: 75%;
                margin-left: 40px;
                margin-top: 35px;
                border-collapse: collapse;
                font-size: 16px;
                max-height: calc(100vh - 380px);;        
                overflow-y: auto;   
            }

            /* results table */
            .results_table {
                width: 100%;
                margin: 0;        
            }

            /* results table - heading */
            .results_table th {
                background-color: #d7e3f2;
                padding: 12px;
                text-align: left;
            }

            /* results table - data */
            .results_table td {
                background-color: #F5F5F5;
                padding: 12px;
                border-top: 2px solid #FFFFFF;
            }

            /* listing title */
            details summary {
                cursor: pointer;
                font-size: 16px;
            }

            /* listing attributes */
            details div {
                margin-top: 6px;
                font-size: 14px;
            }

            /* results table - every 2nd row */
            .results_table tr:nth-child(even) td {
                background-color: #F8F8F8;
            }

            /* no search entered message */
            .no_search_message {
                margin-left: 40px;
                margin-top: 40px;
                width: 75%;
                padding: 16px 22px;
                background-color: #F0F0F0;
                border-radius: 6px;
                font-size: 16px;
            }

            /* no results found message */
            .no_results_message {
                margin-left: 40px;
                margin-top: 40px;
                width: 75%;
                padding: 16px 22px;
                background-color: #F0F0F0;
                border-radius: 6px;
                font-size: 16px;
            }

        </style>
      </head>

      <body>

        <!-- page heading -->
        <div class="header">
            LLM-Based Search of Product Listings
        </div>

        <div class="search_container">
            <form method="post">
            <!-- search bar -->
            <input 
                class="search_input"
                name="user_search_query"
                placeholder="Search for second-hand listings" 
            />
            <!-- search button -->
            <button class="search_button" type="submit">Search</button>
            </form>
        </div>
    """

    #if user clicks search button
    if request.method == "POST":

        #getting value entered by user
        user_search_query = request.form.get("user_search_query", "")

        #if nothing was entred:
        if not user_search_query:
            html += (
                "<div class='no_search_message'>"
                "No search was entered"
                "<br><br>"
                "Please enter an item description using the search bar above"
                "</div>"
            )
                        
        else:

            #prints message to show what it is searching for
            html += (
                "<div class='display_search_query'>"
                "Showing results for: '" + user_search_query + "'"
                "</div>"
            )

            #gather list of matching listings
            result = llm_search(user_search_query)
            matches = result["matches"]
            interpreted_query = result["interpreted_query"]
            filters = result["filters"]

            #display interpreted query 
            html += (
                "<div class='display_interpretation'>"
                "<strong>Interpreted as: </strong>'" + interpreted_query + "'"
                "</div>"
            )

            #display filters
            html += (
                "<p class='display_filters'><strong>Filters:</strong> "
                "Min price: " + str(filters.get("min_price")) + ", "
                "Max price: " + str(filters.get("max_price")) + ", "
                "Location: " + str(filters.get("location")) + ", "
                "Condition: " + str(filters.get("condition")) + ", "
                "Colour: " + str(filters.get("colour")) + ", "
                "Brand: " + str(filters.get("brand")) + ", "
                "Size: " + str(filters.get("size")) +
                "</p>"
            )

            #if no matching listings
            if len(matches) == 0:
                html += (
                    "<div class='no_results_message'>"
                    "No matches found"
                    "<br><br>"
                    "Try adjusting your search"
                    "</div>"
                )

            #display results
            else:

                # start scroll container
                html += "<div class='results_container'>"

                html += """
                <table class="results_table">
                <tr>
                    <th>ID</th>
                    <th>Listing</th>
                    <th>Price</th>
                    <th>Condition</th>
                    <th>Location</th>
                </tr>
                """

                for item in matches:
                    item_id = item.get("id", "")
                    title = item.get("title", "")
                    price = item.get("price", "")
                    location = item.get("location", "")
                    condition = item.get("condition", "")
                    attributes = ", ".join(item.get("attributes", []))

                    html += (
                        "<tr>"
                        "<td>" + str(item_id) + "</td>"
                        "<td>"
                            "<details>"
                                "<summary>" + str(title) + "</summary>"
                                "<div>" + str(attributes) + "</div>"
                            "</details>"
                        "</td>"
                        "<td>£" + str(price) + "</td>"
                        "<td>" + str(condition) + "</td>"
                        "<td>" + str(location) + "</td>"
                        "</tr>"
                    )

                html += "</table>"
                html += "</div>" 

    #ends function and retruns page
    html += "</body></html>"
    return html

if __name__ == "__main__":
    #opens and runs in webbrowser
    import webbrowser
    webbrowser.open("http://localhost:5000")
    #checks page is running correctly
    app.run(debug=True, use_reloader=False)