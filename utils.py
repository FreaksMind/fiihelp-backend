from random import randrange
import re 
image_dict = {
    "alexandru ioan cuza": ["https://tse2.mm.bing.net/th/id/OIP.KwPs3gpGN8_NrglFKmLWXgHaE5?r=0&rs=1&pid=ImgDetMain", "https://www.laiasi.ro/wp-content/uploads/2017/12/Universitatea-Alexandru-Ioan-Cuza-din-Iasi.jpg"],
    "cantina": ["https://www.bzi.ro/wp-content/uploads/2023/02/Cantina-Titu-Maiorescu-Iasi.jpg", "https://tse1.mm.bing.net/th/id/OIP.PBxsudi4TEs-WHw6rs6gYgHaE8?r=0&rs=1&pid=ImgDetMain"],
    "canteen": ["https://www.bzi.ro/wp-content/uploads/2023/02/Cantina-Titu-Maiorescu-Iasi.jpg", "https://tse1.mm.bing.net/th/id/OIP.PBxsudi4TEs-WHw6rs6gYgHaE8?r=0&rs=1&pid=ImgDetMain"],
}

def generate_ngrams(tokens, max_n=3):
    ngrams = set()
    for n in range(1, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[i:i + n]))
    return ngrams

def identify_relevant_images(query):
    no_punct = re.sub(r'[^\w\s]', '', query)
    query_tokens = no_punct.lower().split()
    ngrams = generate_ngrams(query_tokens)

    matched_urls = []
    for phrase in ngrams:
        if phrase in image_dict:
            matched_urls.extend([image_dict[phrase][randrange(len(image_dict[phrase]))]])
    return set(matched_urls[:2])
