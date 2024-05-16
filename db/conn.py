from pymongo import MongoClient # type: ignore
import numpy as np
import hnswlib # type: ignore

class MongoConnection:
    def __init__(self):
        # self.client = MongoClient('mongodb://localhost:27017/')
        self.client = MongoClient('mongodb+srv://doadmin:2SHa3o47Nz6U590c@db-mongodb-sgp1-65501-7de17c59.mongo.ondigitalocean.com/admin?authSource=admin')
        self.db = self.client["defaultdb"]
        self.collection = self.db["faces"]

    def insert(self, user, face_encoding):
        self.collection.insert_one({"user": user, "encoding": face_encoding})
    
    def insertBalance(self, userName, balance):
        self.collection.insert_one({"balance_" + userName: balance})

    def insertInvoice(self, userName, invoice):
        self.collection.insert_one({"invoice_" + userName: invoice})

    def getAllInvoiceOfUser(self, userName):
        userName = "invoice_" + userName
        query = self.collection.find({userName: {"$exists": True}})
        invoices = []
        for q in query:
            invoices.append(q[userName])
        return invoices
    
    def getAllInvoice(self):
        query = self.collection.find()
        invoices = []
        for q in query:
            for key in q:
                if key.startswith("invoice_"):
                    invoices.append(q[key])
        return invoices
    
    def getBalance(self, userName):
        userName = "balance_" + userName
        query = self.collection.find({userName: {"$exists": True}})
        for q in query:
            return q[userName]
        return None
    
    def updateBalance(self, userName, balance):
        self.collection.update_one({"balance_" + userName: {"$exists": True}}, {"$set": {"balance_" + userName: balance}})

    def get_all(self):
        query = self.collection.find()
        
        # Assuming a dimension for the embeddings, make sure this matches your actual data
        dimension = 128  # You need to know the dimensionality of your embeddings beforehand

        # Initialize the HNSW index
        index = hnswlib.Index(space="cosine", dim=dimension)  # You can choose 'l2' for Euclidean or 'ip' for inner product
        index.init_index(max_elements=10000, ef_construction=200, M=16)  # Adjust parameters according to your needs

        ids = 0  # Initialize IDs for hnswlib
        users = []
        for q in query:
            try:
                user = q["user"]
                embeddings = q["encoding"]  # Assuming this is a list of embeddings
                
                # Process each embedding
                for v in embeddings:
                    embedding_array = np.array(v)  # Ensure the data type is float32

                    users.append(user)
                    # Add each embedding to the hnswlib index
                    index.add_items(embedding_array, ids)
                    ids += 1  # Increment ID for each embedding
            except Exception as e:
                # print(f"Error processing user: {e}")
                pass

        # Finalize the index after all embeddings are added
        index.set_ef(50)  # Adjust this parameter according to your search requirements

        return users, index

    def delete_all(self):
        self.collection.delete_many({})
