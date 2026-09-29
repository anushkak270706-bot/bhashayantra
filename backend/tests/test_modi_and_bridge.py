from backend.engines import modi_map
from backend.engines.script_bridge import ScriptBridge
from backend.services.script_detector import ScriptDetector

WORDS = ["श्री", "शिवाजी", "क्षत्रिय", "प्रिय", "किल्ला", "रीत",
         "कमाविसदार", "सनद", "मराठी भाषा । शिवाजी महाराज ॥ १६७४"]


def test_modi_roundtrip_exact():
    for w in WORDS:
        assert modi_map.modi_to_devanagari(modi_map.devanagari_to_modi(w)) == w


def test_bridge_routes_modi_through_our_table():
    b = ScriptBridge()
    modi = b.convert("श्री गणेशाय नमः", "devanagari", "modi")
    assert b.convert(modi, "modi", "devanagari") == "श्री गणेशाय नमः"


def test_modi_to_other_script_via_devanagari():
    b = ScriptBridge()
    modi = b.convert("नमस्ते", "devanagari", "modi")
    assert b.convert(modi, "modi", "kannada") == b.convert("नमस्ते", "devanagari", "kannada")


def test_detector_finds_modi():
    modi = modi_map.devanagari_to_modi("श्री गणेशाय नमः")
    assert ScriptDetector().detect(modi) == {"script": "modi", "confidence": 1.0}
