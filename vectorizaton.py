import  pypdf
from pypdf import PdfReader
import numpy as np
import spacy
import numpy as np

with open('AI_Russell_Norvig.pdf', 'rb') as f:
  pdf_reader =  pypdf.PdfReader(f)
  chunks = []
  # print(pdf_reader.pages)
  for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    text = page.extract_text()
    for i in range(0, len(text), 1000):
      chunks.append(text[i:i+1000])

#print('\n\n' + chunks[1] + '\n\n' + chunks[2] + '\n')


nlp = spacy.load('en_core_web_sm')
embedding_vectors = []

for chunk in chunks:
  doc = nlp(chunk)
  embedding_vector = doc.vector
  array_vec = np.array(embedding_vector)
  embedding_vectors.append(array_vec)

#print('\n',chunks[1] , '\n' ,  embedding_vectors[1] , '\n\n-->-->-->-->-->\n\n' , chunks[2] , embedding_vectors[2],'\n' )


combined_database = [[None,None,None] for i in range(len(chunks))]

for i in range(len(chunks)):
  combined_database[i][0] = 0
  combined_database[i][1] = chunks[i]
  combined_database[i][2] = embedding_vectors[i]

user_input = "what generative artificial intelligence ai could do the world in coming days and their risk associted to it to jobs"
user_doc = nlp(user_input)
user_vector = user_doc.vector

#print(user_vector)

for i in range(len(combined_database)):
  similarity = np.dot(user_vector, combined_database[i][2]) / (np.linalg.norm(user_vector) * np.linalg.norm(combined_database[i][2]))
  combined_database[i][0] = similarity

combined_database.sort(key=lambda x: x[0], reverse=True)

context = '"'
for i in range(2):
  #print(f"\nsimilarityScore: {combined_database[i][0]}\n\n{combined_database[i][1]}\n\n\n{combined_database[i][2]}\n")
  context += combined_database[i][1].replace('"', '')
  context += '\n'
  print('--------------------------------')
context += '"'


summarize = "Summarize below passage a software professional using the below passage in simple terms in bullet points  \n \n \n"
promt_to_LLM = summarize + context
print(promt_to_LLM)

recommend = "Recommend suggestions software professional using the below passage in simple terms in bullet points  \n \n \n"
promt_to_LLM = recommend  + context
print(promt_to_LLM)
