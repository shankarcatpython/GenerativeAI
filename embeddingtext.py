import numpy as np

class WordEmbedding:
    # Initialize the weights of Model
    def __init__(self, vocab, embedding_dim):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.embeddings = {word: np.random.rand(embedding_dim) for word in vocab}

    # Update Embeddings Learning process - Ignore non-keys
    def update_embeddings(self, target_word, context_words, learning_rate=0.01):
        target_vector = self.embeddings.get(target_word)
        context_vectors = [self.embeddings.get(w) for w in context_words]
        
        if target_vector is not None and all(context_vector is not None for context_vector in context_vectors):
            target_vector = np.array(target_vector)
            context_vectors = np.array(context_vectors)
            
            # Move target vector closer to context vectors
            # Model paramter - Mean is chosen to minimize loss ( * also called as loss function)
            self.embeddings[target_word] = target_vector + learning_rate * (np.mean(context_vectors, axis=0) - target_vector)

    # Update Embeddings Learning process - ignore non-keys
    def train(self, corpus, window_size, learning_rate, epochs):
        for epoch in range(epochs):
            for sentence in corpus:
                for i, target_word in enumerate(sentence):
                    context = self.get_context_words(sentence, i, window_size)
                    self.update_embeddings(target_word, context, learning_rate)

    # Traverse corpus sentences within specified window , identify target context ( * collection of words )
    def get_context_words(self, sentence, target_index, window_size):
        start = max(0, target_index - window_size)
        end = min(len(sentence), target_index + window_size + 1)
        context = [sentence[i] for i in range(start, end) if i != target_index]
        return context

    def get_word_vector(self, word):
        return self.embeddings.get(word)

# Software Engineering embedding vocabulary , corpus , Tockenized Input / Output

vocab = ["Requirement","specifications", "Design", "blueprint","Coding","functional", "code" "Testing",  "examination","Deployment","finalized"]
corpus = [["Clear", "and", "concise", "specifications", "pave", "the", "path", "for", "successful", "project", "endeavors.", "Requirement"],
    ["Crafting", "an", "elegant", "blueprint", "ensures", "the", "foundation", "for", "a", "robust", "and", "scalable", "solution.", "Design"],
    ["Transforming", "conceptualized", "designs", "into", "functional", "code", "demands", "precision", "and", "creativity.", "Coding"],
    ["Rigorous", "examination", "guarantees", "the", "reliability", "and", "functionality", "of", "the", "developed", "system.", "Testing"],
    ["Seamlessly", "rolling", "out", "the", "finalized", "product", "heralds", "the", "culmination", "of", "diligent", "efforts", "and", "signifies", "the", "beginning", "of", "its", "journey.", "Deployment"]
]

# Hyperparameters Used in Model
embedding_dim = 5  # Adjust as needed
model = WordEmbedding(vocab, embedding_dim)
window_size = 3
learning_rate = 0.01
epochs = 1

print("-" * 100)
print(f"Hyperparamters: \n Embedding Dimension : {embedding_dim} \n Window Size : {window_size}")
print(f"Learning Rate : {learning_rate} \n epochs : {epochs}")
print("-" * 100)
print(f"Model Parameters:\n Loss function - Arithmetic mean")
print("-" * 100)

# Initial Weights 
for value in vocab:
    word_vector = model.get_word_vector(value)
    print(f"Vector Representation Initial {value}:", word_vector)

# Train the model 
model.train(corpus, window_size, learning_rate, epochs)

print("-" * 100)

# Final weights 
for value in vocab:
    word_vector = model.get_word_vector(value)
    print(f"Vector representation Final {value}:", word_vector)