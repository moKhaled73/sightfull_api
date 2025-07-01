from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.heatmap import router as heatmap_router
from routes.scanpath import router as scanpath_router
from routes.recommendations import router as recommendations_router 
from routes.scores import router as scores_router

app = FastAPI()

# fastapi middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows the specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(heatmap_router, prefix='/heatmap', tags=['Heatmap'])
app.include_router(scanpath_router, prefix='/scanpath', tags=['Scanpath'])
app.include_router(scores_router, prefix='/scores', tags=['Scores'])
app.include_router(recommendations_router, prefix='/recommendations', tags=["Recommendation"])
