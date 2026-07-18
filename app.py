"""Placeholder entrypoint for build order step 1: confirm the Space boots
before any model logic is added."""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Nobility2")


@app.get("/")
def root():
    return {"status": "ok", "component": "nobility2 scaffold", "stage": "step-1-placeholder"}


@app.get("/health")
def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
