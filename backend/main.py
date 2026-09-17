import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from backend.services.transliterator import Transliterator
from backend.services.language_detector import LanguageDetector
from backend.config.languages import LANGUAGES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(title="BhashaYantra API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


DETECTED_TO_INTERNAL = {
    "hin": "hi",
    "ben": "bn",
    "asm": "as",
    "brx": "brx",
    "doi": "doi",
    "guj": "gu",
    "kan": "kn",
    "kas": "ks",
    "gom": "kok",
    "mai": "mai",
    "mal": "ml",
    "mni": "mni",
    "mar": "mr",
    "nep": "ne",
    "ori": "or",
    "pan": "pa",
    "san": "sa",
    "snd": "sd",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}


# Load models once when the backend starts
transliterator = Transliterator()
language_detector = LanguageDetector()


class TransliterationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    language_code: str = Field(..., min_length=2, max_length=5)


class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


class AutoTransliterationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    language_code: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=5
    )


@app.get("/")
def home():
    return {
        "message": "BhashaYantra backend is running!"
    }


@app.get("/languages")
def get_languages():
    return LANGUAGES


@app.post("/detect-language")
def detect_language(request: LanguageDetectionRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    predictions = language_detector.detect(request.text)

    return {
        "input": request.text,
        "predictions": predictions
    }


@app.post("/auto-transliterate")
def auto_transliterate(request: AutoTransliterationRequest):

    logger.info(
    "Auto-transliteration request received: %s",
    request.text
)

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    detector_confidence = None

    # If the user provides a language, use it directly
    if request.language_code is not None:

        internal_code = request.language_code

        if internal_code not in LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language code: {internal_code}"
            )

    # Otherwise, detect the language automatically
    else:

        predictions = language_detector.detect(request.text)

        best_prediction = predictions[0]
        detector_confidence = best_prediction["confidence"]

        confidence_threshold = 0.85

        if detector_confidence < confidence_threshold:
            return {
                "input": request.text,
                "status": "needs_language_selection",
                "message": "The language could not be detected confidently. Please select the language manually.",
                "predictions": predictions
            }

        detected_code = best_prediction["language"].split("_")[0]

        internal_code = DETECTED_TO_INTERNAL.get(detected_code)

        if internal_code is None:
            raise HTTPException(
                status_code=400,
                detail=f"Language '{detected_code}' is not currently supported"
            )

    # Check whether a transliteration model is configured
    model_code = LANGUAGES[internal_code]["indicxlit_code"]

    if model_code is None:
        raise HTTPException(
            status_code=400,
            detail=f"No transliteration model is configured for '{internal_code}'"
        )

    # Transliterate using the selected or detected language
    result = transliterator.roman_to_indic(
        text=request.text,
        language_code=internal_code
    )

    return {
        "input": request.text,
        "status": "success",
        "language": LANGUAGES[internal_code]["name"],
        "language_code": internal_code,
        "detector_confidence": detector_confidence,
        "output": result
    }


@app.post("/transliterate")
def transliterate(request: TransliterationRequest):

    if request.language_code not in LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language code: {request.language_code}"
        )

    result = transliterator.roman_to_indic(
        text=request.text,
        language_code=request.language_code
    )

    return {
        "input": request.text,
        "language": LANGUAGES[request.language_code]["name"],
        "language_code": request.language_code,
        "output": result
    }