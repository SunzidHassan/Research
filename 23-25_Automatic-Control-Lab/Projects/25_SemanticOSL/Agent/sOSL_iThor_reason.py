from sentence_transformers import SentenceTransformer, util
import pandas as pd

def goalSimilarity(itemDF, text_similarity_model, goal="food burning smell"):
    """
    Updates itemDF with a 'goalSimilarity' score by comparing the detected object names
    with a given goal description using cosine similarity.
    
    :param itemDF: DataFrame containing detected objects.
    :param text_similarity_model: SentenceTransformer model for generating embeddings.
    :param goal: The goal description to compare against.
    :return: Updated DataFrame with similarity scores.
    """
    goal_embedding = text_similarity_model.encode(goal, convert_to_tensor=True)
    
    for idx, row in itemDF.iterrows():
        if row.get("goalSimilarity") in [None, "", [], float('nan')] or pd.isna(row.get("goalSimilarity")):
            object_name = row.get("name")
            if object_name:
                name_embedding = text_similarity_model.encode(object_name, convert_to_tensor=True)
                similarity = util.cos_sim(goal_embedding, name_embedding).item()
                itemDF.at[idx, "goalSimilarity"] = similarity
                
    return itemDF
