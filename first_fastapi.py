from fastapi import FastAPI

app = FastAPI(title="Welcome API")


@app.get("/")
def root():
    return {"message": "Welcome to my FastAPI app!"}


@app.get("/greet/{name}")
def greet(name: str):
    return {"message": f"Hello, {name}!"}