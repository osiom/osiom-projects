from flask import Flask, request

app = Flask("flask")

stores = [
    {
        "name": "My Store",
        "items": [
            {
                "name": "Chair",
                "price": 15.99
            }
        ]
    }
]

# GET
@app.get("/stores") #http://127.0.0.1:5000/stores
def get_store():
    return {"stores": stores}, 201

@app.get("/stores/<string:name>")
def get_single_store(name):
    for store in stores:
        if store["name"] == name:
            return store, 201
    return {"message": "Store not found"}, 404

@app.get("/stores/<string:name>/item")
def get_single_store_items(name):
    for store in stores:
        if store["name"] == name:
            return store["items"], 201
    return {"message": "Store not found"}, 404

# POST
@app.post("/stores") #http://127.0.0.1:5000/stores
def post_store():
    request_data = request.get_json() # internal flask function to get body post json
    new_store = {"name": request_data["name"], "items": []}
    stores.append(new_store)
    return new_store, 201

@app.post("/stores/<string:name>/item")
def create_item(name):
    request_data = request.get_json() # internal flask function to get body post json
    for store in stores:
        if store["name"] == name:
            new_item = {"name": request_data["name"], "price": request_data["price"]}
            store["items"].append(new_item)
            return new_item, 201
    return {"message": "Store not found"}, 404
