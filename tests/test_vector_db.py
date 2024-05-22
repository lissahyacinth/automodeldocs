from automodeldocs.embed.db import LocalVectorDB, local_vector_db
from automodeldocs.embed.embed_files import embed_common_ml_libs


def test_db():
    db: LocalVectorDB
    embed_common_ml_libs()
    with local_vector_db() as db:
        res = db.query(f"Python function source code with name unique_values")
        breakpoint()
