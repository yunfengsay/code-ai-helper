 Get started in a couple lines
 import txtai

 embeddings = txtai.Embeddings()
 embeddings.index(["Correct", "Not what we hoped"])
 embeddings.search("positive", 1)
 #[(0, 0.29862046241760254)]
