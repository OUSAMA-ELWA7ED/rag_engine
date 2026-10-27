# Pneumonia Classification API — Deployment & Integration Guide

## 1. Project Repository

**GitHub repository:** https://github.com/OUSAMA-ELWA7ED/rag_engine

**Branch:** `fix/docker-model-inference`

This branch contains the Docker configuration, inference engine, and model configuration files for the pneumonia classification API.

> **Status:** The branch has been pushed to GitHub, but the Docker build and deployment still need to be verified. The model weights are not included in GitHub and must be provided separately.

## 2. Install Docker

Install Docker Desktop on Windows, or Docker Engine and the Docker Compose plugin on Linux.

- Docker Desktop: https://www.docker.com/products/docker-desktop/
- Docker Engine for Linux: https://docs.docker.com/engine/install/

After installation, start Docker and verify it is running:

```bash
docker --version
docker compose version
```

## 3. Download the Project and Configure It

Clone the required branch:

```bash
git clone -b fix/docker-model-inference https://github.com/OUSAMA-ELWA7ED/rag_engine.git
cd rag_engine
```

Create a local environment file from the provided example.

**Windows PowerShell:**

```powershell
Copy-Item .env.example .env
notepad .env
```

**Linux:**

```bash
cp .env.example .env
nano .env
```

Configure the required environment variables, including the database settings and any required API keys. Do not commit `.env` or share secret API keys.

## 4. Install the Model Weights

The model weights are excluded from GitHub because of their size. Obtain them from the project owner and place them in this directory:

```text
rag_engine/
└── model_package/
    ├── pipeline_config.json
    ├── cxr_inference.py
    ├── requirements.txt
    └── weights/
        ├── resnet50.pt
        ├── densenet121.pt
        └── vit_small.pt
```

The three required checkpoint files are:

- `resnet50.pt`
- `densenet121.pt`
- `vit_small.pt`

The Docker Compose configuration mounts `./model_package` into the container at `/models` in read-only mode. The Dockerfile sets `MODEL_PACKAGE_DIR=/models`.

Ensure the files are present before starting the API.

## 5. Start the Containers

From the project root, run:

```bash
docker compose up -d --build
```

Check container status:

```bash
docker compose ps
```

View API logs if needed:

```bash
docker compose logs -f api
```

The configured services are:

- **API:** Port `8000`
- **PostgreSQL with pgvector:** Host port `5433`, container port `5432`

The first build downloads Python dependencies and CPU-compatible PyTorch packages. It requires a working internet connection and may take some time.

## 6. API Base URL

For local development:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

**Production deployment:** Replace `http://localhost:8000` with the actual public server URL, for example `https://your-server-domain.com`. No public deployment URL has been confirmed yet.

The API must be deployed on a reachable server and exposed securely over HTTPS before a frontend hosted elsewhere can call it.

## 7. Pneumonia Classification Endpoint

**Method:** `POST`

**Path:** `/classify/pneumonia`

**Full local URL:**

```text
http://localhost:8000/classify/pneumonia
```

**Content type:** `multipart/form-data`

The endpoint accepts an uploaded chest X-ray image. Use the Swagger UI at `/docs` to confirm the exact multipart field name and accepted image formats before integrating it into the frontend.

### Example request using Python

Replace `file` below with the exact field name shown in Swagger if it differs.

```python
import requests

url = "http://localhost:8000/classify/pneumonia"

with open("chest_xray.jpeg", "rb") as image:
    response = requests.post(
        url,
        files={"file": ("chest_xray.jpeg", image, "image/jpeg")}
    )

print(response.status_code)
print(response.json())
```

Install the client dependency if necessary:

```bash
pip install requests
```

### Example JSON response

A previously successful local inference returned:

```json
{
  "prediction": "PNEUMONIA",
  "probability_pneumonia": 0.9948,
  "threshold": 0.554543137550354,
  "uncertain": false,
  "per_model_probability": {
    "resnet50": 0.9994,
    "densenet121": 0.9929,
    "vit_small": 0.8942
  },
  "tta_views": 9
}
```

This is an example of a previous response, not a guaranteed result for every image.

Response fields:

- `prediction`: predicted class.
- `probability_pneumonia`: estimated pneumonia probability.
- `threshold`: classification threshold used by the model.
- `uncertain`: uncertainty indicator returned by the inference pipeline.
- `per_model_probability`: individual model scores.
- `tta_views`: number of test-time augmentation views used.

These are experimental model predictions and must not be treated as a medical diagnosis.

## 8. Test Through Swagger UI

1. Start the Docker containers.
2. Open `http://localhost:8000/docs`.
3. Locate `POST /classify/pneumonia`.
4. Click **Try it out**.
5. Select a sample chest X-ray image using the file-upload field.
6. Click **Execute**.
7. Inspect the HTTP status code and JSON response.

If the endpoint is missing, the API fails to start, or model loading fails, inspect the logs:

```bash
docker compose logs --tail=200 api
```

## 9. Frontend Integration Notes

The frontend should:

1. Send an HTTP `POST` request to `/classify/pneumonia`.
2. Submit the image as `multipart/form-data`, using the field name specified by Swagger.
3. Parse the JSON response and display the returned fields.
4. Handle unsuccessful HTTP responses and network errors.
5. Clearly label results as experimental predictions, not diagnoses.

For production use, configure HTTPS, appropriate access controls, upload-size limits, and secure handling of medical images.

## 10. Troubleshooting

**Docker cannot download images or dependencies:** Check the internet connection, DNS, proxy/VPN configuration, and Docker Desktop status.

**Model files are missing:** Verify that all three `.pt` files are inside `model_package/weights/` on the host.

**API container exits:** Run `docker compose logs --tail=200 api`.

**Database connection fails:** Check that PostgreSQL is healthy and the configured database credentials match the Compose configuration.

**Important:** Successful deployment and endpoint behavior must be confirmed by running the build and tests on the target machine.
