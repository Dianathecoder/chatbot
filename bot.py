import pandas as pd
import requests
import os
from rapidfuzz import process

# PREPROCESSING ---

def get_vocabulary_from_json(json_path):
    """Extracts dish names from the JSON to use for fuzzy matching."""
    try:
        df = pd.read_json(json_path)
       
        return df['name'].tolist() 
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

# INITIALIZE VOCABULARY
VOCABULARY = get_vocabulary_from_json("recetas.json")


ERRORS = {
    "spageti": "spaghetti",
    "spagueti": "spaghetti",
    "piza": "pizza",
    "pisa": "pizza",
    "risoto": "risotto",
    "rizoto": "risotto",
    "lasaña": "lasagna",
    "lasagna": "lasagna",
    "gnonchi": "gnocchi",
    "noqui": "gnocchi",
    "tortia": "tortilla",
    "paeya": "paella",
    "recepi": "recipe",
    "ingredents": "ingredients",
    "coook": "cook",
    "how to": "how to"
}

def preprocess_input(text: str) -> str:
    """Cleans and corrects user input before searching."""
    text = text.lower().strip()
    
   
    for typo, correction in ERRORS.items():
        text = text.replace(typo, correction)
    

    words = text.split()
    corrected_words = []
    for word in words:
        
        match, score, _ = process.extractOne(word, VOCABULARIO)
        corrected_words.append(match if score > 85 else word)
        
    return " ".join(corrected_words)

#RAG LOGIC (Context Retrieval) 

def retrieve_json_context(clean_query, json_path):
    """Searches the JSON for the relevant recipe row."""
    try:
        df = pd.read_json(json_path)

        results = df[df.apply(lambda row: row.astype(str).str.contains(clean_query, case=False).any(), axis=1)]
        
        if not results.empty:
      
            return results.to_json(orient="records")
        return None
    except:
        return None

#  LLM CONNECTION (Chef AI)

def consult_chef_ai(context, query):
    """Sends the context and query to LM Studio."""
    url = "http://localhost:1234/v1/chat/completions"
    
    system_prompt = (
        "You are a world-class Professional Chef. "
        "Use ONLY the provided RECIPE CONTEXT to answer the user's question. "
        "Explain the ingredients and the cooking process step-by-step. "
        "If the recipe is not in the context, politely inform the user that it's not in your menu."
    )
    
    user_prompt = f"RECIPE CONTEXT:\n{context}\n\nUSER QUESTION: {query}"
    
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, json=payload, timeout=15)
        return response.json()['choices'][0]['message']['content']
    except:
        return "CONNECTION ERROR: Please make sure LM Studio Server is running on port 1234."

#  4. MAIN PIPELINE ---

def handle_query(user_input):
    """The complete flow: Preprocess -> RAG -> Decision -> Generation."""
    # A. Preprocessing
    clean_text = preprocess_input(user_input)
    
    # B. RAG Search
    context = retrieve_json_context(clean_text, "recetas.json")
    
    # C. Decision / Escalation Logic
    if not context:
        return f"I'm sorry, I don't have the recipe for '{user_input}' in my current database."
    
    # D. AI Response
    return consult_chef_ai(context, clean_text)

# EXECUTION

if __name__ == "__main__":
    print("--- 👨‍🍳 Chef AI Assistant Active ---")
    while True:
        user_query = input("\nAsk for a recipe (or type 'quit'): ")
        if user_query.lower() in ["quit", "exit", "stop"]:
            print("Goodbye! Happy cooking.")
            break
            
        response = handle_query(user_query)
        print(f"\nAI: {response}")