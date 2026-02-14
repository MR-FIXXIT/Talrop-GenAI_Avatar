from fastapi import FastAPI
from db import Base, engine
import models  # noqa: F401  (registers all models)

from routes.orgs import router as orgs_router
from routes.keys import router as keys_router
from routes.chat import router as chat_router
from routes.documents import router as documents_router
from routes.health import router as health_router

app = FastAPI()

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

app.include_router(orgs_router)
app.include_router(keys_router)
app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(health_router)
