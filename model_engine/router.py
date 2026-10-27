from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile

router = APIRouter(prefix="/classify", tags=["xray-classification"])


@router.get("/models")
def list_models(request: Request):
    ens = request.app.state.xray_ensemble
    return {"models": list(ens.models), "labels": ens.labels, "version": ens.cfg["package_version"]}


@router.post("/pneumonia")
async def classify_pneumonia(
    request: Request,
    file: UploadFile = File(...),
    models: list[str] | None = Query(None, description="Subset of models; default = full ensemble"),
):
    ens = request.app.state.xray_ensemble
    if models and not set(models) <= set(ens.models):
        raise HTTPException(400, f"Unknown model. Available: {list(ens.models)}")
    try:
        return ens.predict(await file.read(), only=models)
    except ValueError as e:
        raise HTTPException(400, str(e))
