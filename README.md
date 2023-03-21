# Library

run start.sh to start frontend and backend simultaneously


dataset folder: https://drive.google.com/drive/folders/19TVG9MG6x3_zHrq2XOO3RS9mQNCISory?usp=sharing


Retrieve results with a query:
1. cd into the backend folder, and cd into the IR folder
2. create a virtual environment, and install from requirements.txt with pip install -r requirements.txt
3. Run the indexing file with: python indexDataset.py
4. Start the backend, and the endpoint should be callable with a query (the endpoint should be called with a post request, with a dict: {"query" : "your query here"}, and to the endpoint "http://localhost:5000/retrieve"
