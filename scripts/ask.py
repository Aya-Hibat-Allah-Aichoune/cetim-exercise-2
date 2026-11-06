import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.pipeline import RagPipeline  

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "¿De qué tratan los documentos?"
    result = RagPipeline().query(question)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
